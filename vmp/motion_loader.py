

from isaacgym import gymapi
from isaacgym import gymtorch

from torch.utils.data import DataLoader, TensorDataset
import os
import sys

# Add ase directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
ase_dir = os.path.join(os.path.dirname(current_dir), 'ase')
if ase_dir not in sys.path:
    sys.path.append(ase_dir)

from utils.motion_lib import MotionLib
import torch


def prepare_data_loader(motion_data_buffer, batch_size=32):
    # Convert the buffer list to tensor
    motion_data_tensor = torch.stack(motion_data_buffer)
    dataset = TensorDataset(motion_data_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader



class MotionLoader:

    def __init__(self, cfg):
        self.cfg = cfg
        self._setup_character_props()
        
    def _setup_character_props(self):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
            self._dof_obs_size = 72
            self._num_actions = 28
            self._num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
            
        elif (asset_file == "mjcf/amp_humanoid_sword_shield.xml"):
            self._dof_body_ids = [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16]
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 16, 17, 20, 21, 24, 27, 28, 31]
            self._dof_obs_size = 78
            self._num_actions = 31
            self._num_obs = 1 + 17 * (3 + 6 + 3 + 3) - 3

        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        return

    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)
        self._motion_lib = MotionLib(motion_file=motion_file,
                                        dof_body_ids=self._dof_body_ids,
                                        dof_offsets=self._dof_offsets,
                                        key_body_ids=self._key_body_ids.cpu().numpy(), 
                                        device=self.device)
        return

