
# llc pretrain
# ckpt
# /data/ASE/output/Humanoid_08-14-50-58-getup/nn/Humanoid.pth
# humanoid_ase_getup.yaml
python ase/run.py --task HumanoidAMPGetup --cfg_env ase/data/cfg/humanoid_ase_sword_shield_getup.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --headless
python ase/run.py --task HumanoidAMPGetup --cfg_env ase/data/cfg/humanoid_ase_getup.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml


# play
python ase/run.py --test --task HumanoidAMPGetup --num_envs 16 --cfg_env ase/data/cfg/humanoid_ase_getup.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --checkpoint /data/ASE/output/Humanoid_08-21-40-38/nn/Humanoid.pth

# llc play
python ase/run.py --test --task HumanoidAMPGetup --num_envs 16 --cfg_env ase/data/cfg/humanoid_ase_sword_shield_getup.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --checkpoint /data/ASE/output/Humanoid_08-14-50-58-getup/nn/Humanoid.pth




# amp
python ase/run.py --task HumanoidAMP --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/walk/RL_Avatar_WalkForward01_Motion.npy

# /data/ASE/output/Humanoid_09-03-17-33-amp-walk-forward
# to test whether amp can replicate the walk forward motion
python ase/run.py --task HumanoidAMP --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/walk/RL_Avatar_WalkForward01_Motion.npy --num_envs 1024 --minibatch_size 1024

# looks good
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/walk/RL_Avatar_WalkForward01_Motion.npy --checkpoint /data/ASE/output/Humanoid_09-03-17-33-amp-walk-forward/nn/Humanoid.pth

# if load delibrately motion which is not trained, could fall
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/walk/RL_Avatar_WalkRight01_Motion.npy --checkpoint /data/ASE/output/Humanoid_09-03-17-33-amp-walk-forward/nn/Humanoid.pth
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/walk/RL_Avatar_TurnLeft90_Motion.npy --checkpoint /data/ASE/output/Humanoid_09-03-17-33-amp-walk-forward/nn/Humanoid.pth


# now try training a combo of walk motions
python ase/run.py --task HumanoidAMP --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/walk/dataset_reallusion_walk.yaml --num_envs 1024 --minibatch_size 1024 --checkpoint /data/ASE/output/Humanoid_09-03-17-33-amp-walk-forward/nn/Humanoid.pth











