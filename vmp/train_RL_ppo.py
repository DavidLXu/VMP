import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model
from skrl.resources.preprocessors.torch import RunningStandardScaler

# Policy and Value networks with shared architecture
class MLPNetwork(Model):
    def __init__(self, observation_space, action_space, device):
        super().__init__(observation_space, action_space, device)
        
        self.net = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU()
        )
        
        # Policy head (mean for diagonal Gaussian)
        self.policy_head = nn.Linear(512, self.action_space.shape[0])
        # Value head
        self.value_head = nn.Linear(512, 1)
        
        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, inputs, training=False):
        features = self.net(inputs)
        return self.policy_head(features), self.value_head(features)

    def act(self, inputs, training=False):
        features = self.net(inputs)
        return self.policy_head(features)

    def value(self, inputs, training=False):
        features = self.net(inputs)
        return self.value_head(features)

# Configure PPO
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["batch_size"] = 8192 * 32  # As specified
cfg["mini_batch_size"] = 8192 * 8  # As specified
cfg["clip_range"] = 0.2
cfg["gamma"] = 0.99
cfg["gae_lambda"] = 0.95
cfg["epochs"] = 5
cfg["target_kl"] = 0.01
cfg["grad_norm_clip"] = 1.0
cfg["learning_rate"] = 3e-4  # Standard PPO learning rate
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": observation_space.shape, "device": device}
cfg["value_preprocessor"] = None
cfg["num_envs"] = 8192

# Setup memory
memory = RandomMemory(memory_size=cfg["batch_size"], num_envs=cfg["num_envs"], device=device)

# Create policy and value networks
policy = MLPNetwork(observation_space, action_space, device)
value = MLPNetwork(observation_space, action_space, device)

# Create PPO agent
agent = PPO(models={"policy": policy, "value": value},
           memory=memory,
           cfg=cfg,
           observation_space=observation_space,
           action_space=action_space,
           device=device)

# Training loop
num_epochs = 1000
best_reward = float('-inf')
writer = SummaryWriter('runs/ppo_training')

for epoch in range(num_epochs):
    # Collect rollouts
    agent.collect_rollouts()
    
    # Train agent
    logs = agent.train()
    
    # Get metrics
    avg_reward = logs["rollout/ep_rew_mean"]
    avg_length = logs["rollout/ep_len_mean"] 
    policy_loss = logs["train/policy_loss"]
    value_loss = logs["train/value_loss"]
    approx_kl = logs["train/approx_kl"]
    clip_fraction = logs["train/clip_fraction"]
    
    # Log metrics
    writer.add_scalar('Reward/train', avg_reward, epoch)
    writer.add_scalar('Length/train', avg_length, epoch)
    writer.add_scalar('Loss/policy', policy_loss, epoch)
    writer.add_scalar('Loss/value', value_loss, epoch)
    writer.add_scalar('Stats/kl', approx_kl, epoch)
    writer.add_scalar('Stats/clip_fraction', clip_fraction, epoch)
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Average Reward: {avg_reward:.2f}, "
          f"Average Length: {avg_length:.2f}")
    
    # Save best model
    if avg_reward > best_reward:
        best_reward = avg_reward
        checkpoint = {
            'epoch': epoch,
            'policy_state_dict': agent.policy.state_dict(),
            'value_state_dict': agent.value.state_dict(),
            'reward': best_reward,
        }
        torch.save(checkpoint, 'ppo_best.pt')
        print(f"Checkpoint saved at epoch {epoch + 1} with reward {best_reward:.2f}")

writer.close()
