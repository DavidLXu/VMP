import sys
import os
import pathlib
ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import isaacgym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


from ase.utils.config import set_np_formatting, get_args, parse_sim_params, load_cfg


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


args = get_args()
cfg, cfg_train, logdir = load_cfg(args)

# cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))

if args.horovod:
    cfg_train['params']['config']['multi_gpu'] = args.horovod

if args.horizon_length != -1:
    cfg_train['params']['config']['horizon_length'] = args.horizon_length

if args.minibatch_size != -1:
    cfg_train['params']['config']['minibatch_size'] = args.minibatch_size
    
if args.motion_file:
    cfg['env']['motion_file'] = args.motion_file

# Create default directories for weights and statistics
cfg_train['params']['config']['train_dir'] = args.output_path

vargs = vars(args)

print(vargs)
print("===")
print(cfg)
print("===")
print(cfg_train)





# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                               nn.ELU(),
                               nn.Linear(512, 512),
                               nn.ELU(),
                               nn.Linear(512, 512),
                               nn.ELU())

        self.mean_layer = nn.Linear(512, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(512, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}


# load and wrap the Isaac Gym environment
env = load_isaacgym_env_preview4(task_name="HumanoidAMP", num_envs=1024, headless=False,
                                cli_args=[


                                    # "+task.env.cfg_env=ase/data/cfg/humanoid_sword_shield.yaml", 
                                    # "+task.train.params.config=ase/data/cfg/train/rlg/amp_humanoid.yaml",
                                    # "+task.motion_file=ase/data/motions/walk/dataset_reallusion_walk.yaml",
                                ])
env = wrap_env(env, wrapper="isaacgym-preview4")
# python ase/run.py --task HumanoidAMP --cfg_env ase/data/cfg/humanoid_sword_shield.yaml 
# --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/walk/dataset_reallusion_walk.yaml --num_envs 1024 --minibatch_size 1024 --checkpoint /data/ASE/output/Humanoid_09-03-17-33-amp-walk-forward/nn/Humanoid.pth

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=32, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
'''
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 32  # memory_size
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 4  # 8192 * 32 / (8192 * 8)
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 5e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 2.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 200
cfg["experiment"]["checkpoint_interval"] = 2000
cfg["experiment"]["directory"] = f"runs/torch/HumanoidAMP"
'''

agent = PPO(models=models,
            memory=memory,
            cfg=cfg_train,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

cfg_trainer = {"timesteps": 20001, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()