from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

# Add ase directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
ase_dir = os.path.join(os.path.dirname(current_dir), 'ase')
if ase_dir not in sys.path:
    sys.path.append(ase_dir)

from utils.motion_lib import MotionLib

from RotationContinuity.sanity_test.code.tools import compute_rotation_matrix_from_quaternion

import torch
from torch.utils.data import Dataset, DataLoader

class MotionDataset(Dataset):

    def __init__(self, motion_file, num_motions, window_length):
        self._motion_file = motion_file


        # modify this before changing robot, refer to humanoid_amp.py, print in _load_motion
        self._dof_body_ids = torch.tensor([1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16], device='cuda:0')
        self._dof_offsets = torch.tensor([0, 3, 6, 9, 10, 13, 16, 17, 20, 21, 24, 27, 28, 31], device='cuda:0')
        self._key_body_ids = torch.tensor([5, 10, 13, 16, 6, 9], device='cuda:0')
        self.device = 'cuda:0'

        # TODO: put to cfg
        self.dt = 0.03333333507180214
        
        self._load_motion()

        self.motion_windows = self.get_vmp_obs_window(num_motions, 
                                                      window_length).reshape(num_motions, 
                                                                             window_length, 
                                                                             -1)
        print("sampled motion windows shape:",self.motion_windows.shape)
        # test motion lib data
    def _get_motion_state(self, n, window_length):

        self._num_amp_obs_steps = window_length # window length 
        motion_ids = self._motion_lib.sample_motions(n)
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        # [num of motions] -> [num of motions, window length]
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        # TODO: why time backward?
        time_steps = -self.dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        
        motion_times = motion_times + time_steps
        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        '''
        root_pos: [num of motions * window length, 3]
        root_rot: [num of motions * window length, 4]
        dof_pos: [num of motions * window length, 31]
        root_vel: [num of motions * window length, 3]
        root_ang_vel: [num of motions * window length, 3]
        dof_vel: [num of motions * window length, 31]
        key_pos: [num of motions * window length, 6, 3]
        '''
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos
    
    # https://github.com/papagina/RotationContinuity/issues/3
    def rot_to_6d(self, rot_matrix): 
        return torch.concatenate((rot_matrix[:, :, 0], rot_matrix[:, :, 1]),dim=-1)

    def get_vmp_obs_window(self,n=1, window_length=10):
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._get_motion_state(n, window_length)
        '''conform to the notations in the VMP paper'''
        # h, height of the character’s root relative to the ground
        h_t = root_pos[:, 2]
        # theta, orientation of the root, expressed as a 6D vector
        rot_mat = compute_rotation_matrix_from_quaternion(root_rot)
        theta_t = self.rot_to_6d(rot_mat)
        # v, linear and angular velocity of the root, expressed as a 6D vector
        v_t = torch.cat([root_vel, root_ang_vel], dim=-1)
        # q, joint angles, expressed as a 31D vector
        q_t = dof_pos
        # dq, joint velocities, expressed as a 31D vector
        dq_t = dof_vel
        # p, of hands and feet, relative to the root, in the features for frame t.
        p_t = key_pos
        p_t_flat = p_t.reshape(p_t.shape[0], -1) # from [batch, 6, 3] to [batch, 18] 
        
        obs = torch.cat([
            h_t.unsqueeze(-1),  # [batch, 1]
            theta_t,            # [batch, 6] 
            v_t,                # [batch, 6]
            q_t,                # [batch, 31]
            dq_t,               # [batch, 31]
            p_t_flat            # [batch, 18]
        ], dim=-1)

        return obs
        
        
    def _load_motion(self):
        self._motion_lib = MotionLib(motion_file=self._motion_file,
                                        dof_body_ids=self._dof_body_ids,
                                        dof_offsets=self._dof_offsets,
                                        key_body_ids=self._key_body_ids.cpu().numpy(), 
                                        device=self.device)
        return
    
    def __len__(self):
        return len(self.motion_windows)

    def __getitem__(self, idx):
        return self.motion_windows[idx, ...]



if __name__ == "__main__":

    motion_combo = "/data/ASE/ase/data/motions/walk/dataset_reallusion_walk.yaml"
    motion_clip = "/data/ASE/ase/data/motions/walk/RL_Avatar_WalkForward01_Motion.npy"



    motion_dataset = MotionDataset(motion_file=motion_combo,
                            num_motions=1000,
                            window_length=10)

    motion_dataloader = DataLoader(motion_dataset, batch_size=32, shuffle=True)

    # Iterate through the DataLoader
    for batch in motion_dataloader:
        print("Batch shape:", batch.shape)  # Each batch will be of shape (batch_size, window_size, feature_dim)
        # Feed `batch` into your VAE for training



