# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from enum import Enum
import numpy as np
import torch

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.humanoid import Humanoid, dof_to_obs
from utils import gym_util
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *

from utils import torch_utils
from RotationContinuity.sanity_test.code.tools import compute_rotation_matrix_from_quaternion

class HumanoidVMP(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2 # this is selected in config
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidVMP.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]

        self._num_vmp_W = cfg["env"]["numVMPWindowHalf"]
        self._num_vmp_obs_steps = self._num_vmp_W * 2 + 1
        self._num_vmp_obs_per_step = cfg["env"]["numVMPObsPerStep"] # 93 in this case

        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0] # torch.Size([128, 140])
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:] # torch.Size([128, 9, 140])

        self._amp_obs_demo_buf = None
        self._vmp_obs_demo_buf = None

        return

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        # NOTE self._amp_obs_buf is shifted by 1 step every time step
        # the change of self._curr_amp_obs_buf and self._hist_amp_obs_buf
        # directly affects self._amp_obs_buf
        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())

        self.extras["amp_obs"] = amp_obs_flat

        # vmp condition = m_t,z_t
        vmp_obs_t = self.get_vmp_obs() # num_envs, 93
        self.extras["vmp_obs"] = vmp_obs_t

        self.extras["vmp_obs_window"] = self.fetch_vmp_obs_demo()  # better with num_envs, 2W+1, 93
        # print(self.extras["vmp_obs_window"])
        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def fetch_amp_obs_demo(self, num_samples):
        # not used in --test mode (called by amp_agent in training)
        # num_samples is up to amp_batch_size in the config
        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        motion_ids = self._motion_lib.sample_motions(num_samples)
        self._amp_obs_demo_buf_motion_ids = motion_ids
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        self._amp_obs_demo_buf_motion_times0 = motion_times0

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0):
        # not used in --test mode (called by amp_agent in training)
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        # print(motion_ids.shape, motion_times.shape)
        
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        # import time; time.sleep(1)
        # print(motion_ids.shape, motion_times.shape)
        # print(root_pos.shape, root_rot.shape, dof_pos.shape, root_vel.shape, root_ang_vel.shape, dof_vel.shape)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets)
        # import pdb; pdb.set_trace()
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
    
    '''VMP'''

    def get_num_vmp_obs(self):
        return self._num_vmp_obs_steps * self._num_vmp_obs_per_step
    
    def fetch_vmp_obs_demo(self):
        # not used in --test mode (called by vmp_agent in training)
        # num_samples is up to amp_batch_size in the config

        '''do not sample again, use amp sampled motion ids and times'''

        if (self._vmp_obs_demo_buf is None):
            self._build_vmp_obs_demo_buf(self.num_envs)
        else:
            assert(self._vmp_obs_demo_buf.shape[0] == self.num_envs)
        
        # Use the same motion_ids and times as AMP
        motion_ids = self._amp_obs_demo_buf_motion_ids
        motion_times0 = self._amp_obs_demo_buf_motion_times0
        # Store motion_times0 for potential future use
        self._vmp_obs_demo_buf_motion_times0 = motion_times0

        vmp_obs_demo = self.build_vmp_obs_demo(motion_ids, motion_times0)

        return vmp_obs_demo

    def build_vmp_obs_demo(self, motion_ids, motion_times0):
        # not used in --test mode (called by amp_agent in training)
        dt = self.dt

        # Expand motion_ids for all timesteps in window (2W + 1)
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_vmp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)

        # Create time steps centered at t=0, ranging from -W to +W
        # This gives us [-W*dt, ..., -dt, 0, dt, ..., W*dt]
        time_steps = dt * torch.arange(-self._num_vmp_W, self._num_vmp_W + 1, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)

        # print("motion_times", motion_times.shape)
        motion_times = motion_times.view(-1)
        # print("motion_times", motion_times.shape)

        
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)

        # Build VMP observations for each timestep
        # [31232, 1] [31232, 6] [31232, 6] [31232, 31] [31232, 31] [31232, 18]
        # vmp_obs_demo torch.Size([512, 61, 93])

        
        vmp_obs_demo = build_vmp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                            dof_pos, dof_vel, key_pos,
                                            )

        return vmp_obs_demo.view(self.num_envs, 
                                                self._num_vmp_obs_steps, 
                                                self._num_vmp_obs_per_step)

    def _build_vmp_obs_demo_buf(self, num_envs):
        # Buffer shape remains the same, but now represents [-W, ..., 0, ..., W] timesteps
        self._vmp_obs_demo_buf = torch.zeros(
            (num_envs, self._num_vmp_obs_steps, self._num_vmp_obs_per_step), 
            device=self.device, 
            dtype=torch.float32
        )
        return
    

    ''''''
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/amp_humanoid_sword_shield.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 31 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        return

    def _load_motion(self, motion_file):
        # with sword and shield
        # print(self._dof_body_ids) 
        # print(self._dof_offsets) 
        # [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16]
        # [0, 3, 6, 9, 10, 13, 16, 17, 20, 21, 24, 27, 28, 31]
        # without
        # [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]
        # [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
        # exit()

        assert(self._dof_offsets[-1] == self.num_dof)
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return
    
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super()._reset_envs(env_ids)
        self._init_amp_obs(env_ids)

        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidVMP.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidVMP.StateInit.Start
              or self._state_init == HumanoidVMP.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidVMP.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return
    
    def _reset_default(self, env_ids):
        '''not used, use _reset_ref_state_init when '''
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        '''Uses this one'''
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)

        if (self._state_init == HumanoidVMP.StateInit.Random
            or self._state_init == HumanoidVMP.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == HumanoidVMP.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        # print(dof_pos.shape)
        # exit()
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        # print(self._hist_amp_obs_buf.shape, curr_amp_obs.shape)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        # print(self._hist_amp_obs_buf.shape)

        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, 
                                              dof_pos, dof_vel, key_pos, 
                                              self._local_root_obs, self._root_height_obs, 
                                              self._dof_obs_size, self._dof_offsets)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return
    
    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if (env_ids is None):
            self._curr_amp_obs_buf[:] = build_amp_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._dof_offsets)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._rigid_body_pos[env_ids][:, 0, :],
                                                                   self._rigid_body_rot[env_ids][:, 0, :],
                                                                   self._rigid_body_vel[env_ids][:, 0, :],
                                                                   self._rigid_body_ang_vel[env_ids][:, 0, :],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._dof_offsets)
        return


    def rot_to_6d(self, rot_matrix): 
        return torch.concatenate((rot_matrix[:, :, 0], rot_matrix[:, :, 1]),dim=-1)
    
    def get_m_t_hat(self):
        
        '''
        conform to the notations in the VMP paper
        m_t is reference
        m_t hat is simulated result
        '''
        # h, height of the character’s root relative to the ground
        h_t_hat = self._humanoid_root_states[:, 2]
        # theta, orientation of the root, expressed as a 6D vector
        rot_mat_hat = compute_rotation_matrix_from_quaternion(self._humanoid_root_states[:, 3:7])
        theta_t_hat = self.rot_to_6d(rot_mat_hat)
        # v, linear and angular velocity of the root, expressed as a 6D vector
        
        # TODO: verify idx 0 is indeed the root motion in the calculated motion reference
        v_t_hat = torch.cat([self._rigid_body_vel[:, 0, :], self._rigid_body_ang_vel[:, 0, :]], dim=-1)
        
        
        # q, joint angles, expressed as a 31D vector
        q_t_hat = self._dof_pos
        # dq, joint velocities, expressed as a 31D vector
        dq_t_hat = self._dof_vel
        # p, of hands and feet, relative to the root, in the features for frame t.
        p_t_hat = self._rigid_body_pos[:, self._key_body_ids, :]
        p_t_flat_hat = p_t_hat.reshape(p_t_hat.shape[0], -1) # from [batch, 6, 3] to [batch, 18] 
        
        m_t_hat = {
            "h_t": h_t_hat.unsqueeze(-1),  # [batch, 1]
            "theta_t": theta_t_hat,            # [batch, 6] 
            "v_t": v_t_hat,                # [batch, 6]
            "q_t": q_t_hat,                # [batch, 31]
            "dq_t": dq_t_hat,               # [batch, 31]
            "p_t_flat": p_t_flat_hat            # [batch, 18]
        }
        
        return m_t_hat
    
    def get_vmp_obs(self):
        '''
        similar to get_m_t_hat, but with concatenated features
        '''
        # h, height of the character’s root relative to the ground
        h_t_hat = self._humanoid_root_states[:, 2]
        # theta, orientation of the root, expressed as a 6D vector
        rot_mat_hat = compute_rotation_matrix_from_quaternion(self._humanoid_root_states[:, 3:7])
        theta_t_hat = self.rot_to_6d(rot_mat_hat)
        # v, linear and angular velocity of the root, expressed as a 6D vector
        
        # TODO: verify idx 0 is indeed the root motion in the calculated motion reference
        v_t_hat = torch.cat([self._rigid_body_vel[:, 0, :], self._rigid_body_ang_vel[:, 0, :]], dim=-1)
        
        # q, joint angles, expressed as a 31D vector
        q_t_hat = self._dof_pos
        # dq, joint velocities, expressed as a 31D vector
        dq_t_hat = self._dof_vel
        # p, of hands and feet, relative to the root, in the features for frame t.
        p_t_hat = self._rigid_body_pos[:, self._key_body_ids, :]
        p_t_flat_hat = p_t_hat.reshape(p_t_hat.shape[0], -1) # from [batch, 6, 3] to [batch, 18] 
        
        m_t_hat = torch.cat([
            h_t_hat.unsqueeze(-1),     # [batch, 1]
            theta_t_hat,                # [batch, 6]
            v_t_hat,                    # [batch, 6] 
            q_t_hat,                    # [batch, 31]
            dq_t_hat,                   # [batch, 31]
            p_t_flat_hat                # [batch, 18]
        ], dim=-1)
        return m_t_hat
    
    def get_m_t_ref(self, motion_ids, motion_times):
        '''
        Get reference motion state for given motion_ids and times
        Args:
            motion_ids: tensor of motion IDs
            motion_times: tensor of motion times
        Returns:
            Dictionary containing reference motion components
        '''
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
            = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        # h_t: root height
        h_t = root_pos[:, 2]
        
        # theta_t: root orientation (6D)
        rot_mat = compute_rotation_matrix_from_quaternion(root_rot)
        theta_t = self.rot_to_6d(rot_mat)
        
        # v_t: root velocities
        v_t = torch.cat([root_vel, root_ang_vel], dim=-1)
        
        # q_t: joint angles
        q_t = dof_pos
        
        # dq_t: joint velocities
        dq_t = dof_vel
        
        # p_t: end-effector positions relative to root
        root_pos_expand = root_pos.unsqueeze(-2)
        p_t = key_pos - root_pos_expand
        p_t_flat = p_t.reshape(p_t.shape[0], -1)
        
        m_t_ref = {
            "h_t": h_t.unsqueeze(-1),      # [batch, 1]
            "theta_t": theta_t,             # [batch, 6]
            "v_t": v_t,                     # [batch, 6]
            "q_t": q_t,                     # [batch, 31]
            "dq_t": dq_t,                   # [batch, 31]
            "p_t_flat": p_t_flat           # [batch, 18]
        }
        return m_t_ref
    
    def _compute_reward(self, actions):

        m_t_hat=self.get_m_t_hat()
        m_t_ref=self.get_m_t_ref(self._amp_obs_demo_buf_motion_ids, self._amp_obs_demo_buf_motion_times0)
        
        # print(m_t_hat["q_t"].shape, m_t_ref["q_t"].shape)
        self.rew_buf[:] = \
             self.compute_vmp_reward(m_t_hat, m_t_ref)
        return
    
    # TODO move to jit later
    def compute_vmp_reward(self, m_t_hat, m_t_ref):
        # Unpack values from m_t_hat and m_t_ref
        h_t_hat = m_t_hat["h_t"]
        theta_t_hat = m_t_hat["theta_t"] 
        v_t_hat = m_t_hat["v_t"]
        q_t_hat = m_t_hat["q_t"]
        dq_t_hat = m_t_hat["dq_t"]
        p_t_flat_hat = m_t_hat["p_t_flat"]

        h_t_ref = m_t_ref["h_t"]
        theta_t_ref = m_t_ref["theta_t"]
        v_t_ref = m_t_ref["v_t"] 
        q_t_ref = m_t_ref["q_t"]
        dq_t_ref = m_t_ref["dq_t"]
        p_t_flat_ref = m_t_ref["p_t_flat"]

        # Calculate 2-norm differences
        h_t_diff = torch.norm(h_t_hat - h_t_ref, p=2, dim=-1)
        theta_t_diff = torch.norm(theta_t_hat - theta_t_ref, p=2, dim=-1)
        v_t_diff = torch.norm(v_t_hat - v_t_ref, p=2, dim=-1)
        q_t_diff = torch.norm(q_t_hat - q_t_ref, p=2, dim=-1)
        dq_t_diff = torch.norm(dq_t_hat - dq_t_ref, p=2, dim=-1)
        p_t_flat_diff = torch.norm(p_t_flat_hat - p_t_flat_ref, p=2, dim=-1)
        # print("differences",h_t_diff.mean(), theta_t_diff.mean(), v_t_diff.mean(), q_t_diff.mean(), dq_t_diff.mean(), p_t_flat_diff.mean())
        # Sum all differences for tracking reward
        r_track = -(0.5*h_t_diff + theta_t_diff + v_t_diff + q_t_diff + 0.1*dq_t_diff + p_t_flat_diff)
        # print("r_track",r_track.mean())
        r_alive = 6.0
        r_smooth = 0.0
        
        return r_track


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs

@torch.jit.script
def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.tensor(1e-8, device=v.device))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag

    return v

@torch.jit.script
def compute_rotation_matrix_from_quaternion(quaternion):
    batch = quaternion.shape[0]
    
    quat = normalize_vector(quaternion).contiguous()
    
    qw = quat[...,0].contiguous().view(batch, 1)
    qx = quat[...,1].contiguous().view(batch, 1)
    qy = quat[...,2].contiguous().view(batch, 1)
    qz = quat[...,3].contiguous().view(batch, 1)

    # Unit quaternion rotation matrices computation  
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix

@torch.jit.script
def build_vmp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                         ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,) -> Tensor
    """
    Builds VMP observations following the paper's structure:
    - h_t: height of character's root relative to ground
    - theta_t: orientation of root (6D rotation representation)
    - v_t: linear and angular velocity of root (6D vector)
    - q_t: joint angles
    - dq_t: joint velocities
    - p_t: positions of hands and feet relative to root
    """
    
    # h_t: root height (1D)
    h_t = root_pos[:, 2:3]
    
    # theta_t: root orientation (6D)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)    
    # Convert quaternion to 6D rotation representation
    rot_mat = compute_rotation_matrix_from_quaternion(root_rot)
    theta_t = torch.cat([rot_mat[..., :3, 0], rot_mat[..., :3, 1]], dim=-1)
    
    # v_t: root velocities (6D)

    v_linear = root_vel
    v_angular = root_ang_vel
    v_t = torch.cat([v_linear, v_angular], dim=-1)
    
    # q_t: joint angles
    q_t = dof_pos
    
    # dq_t: joint velocities
    dq_t = dof_vel
    
    # p_t: end-effector positions relative to root
    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                            heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    p_t = local_end_pos.view(local_key_body_pos.shape[0], -1)

    
    # Concatenate all features
    obs = torch.cat([
        h_t,        # [batch, 1]
        theta_t,    # [batch, 6]
        v_t,        # [batch, 6]
        q_t,        # [batch, dof_obs_size]
        dq_t,       # [batch, num_dof]
        p_t         # [batch, num_key_bodies * 3]
    ], dim=-1)

    return obs