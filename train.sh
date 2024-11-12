
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

# now try training a combo of walk motions
python ase/run.py --task HumanoidAMP --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/walk/dataset_reallusion_walk.yaml --num_envs 1024 --minibatch_size 1024 --checkpoint /data/ASE/output/Humanoid_09-03-17-33-amp-walk-forward/nn/Humanoid.pth


# looks good
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/walk/RL_Avatar_WalkBackward01_Motion.npy --checkpoint /data/ASE/output/Humanoid_09-18-59-36-amp-walk-turn/nn/Humanoid.pth
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/walk/RL_Avatar_WalkRight01_Motion.npy --checkpoint /data/ASE/output/Humanoid_09-18-59-36-amp-walk-turn/nn/Humanoid.pth
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/walk/RL_Avatar_TurnLeft90_Motion.npy --checkpoint /data/ASE/output/Humanoid_09-18-59-36-amp-walk-turn/nn/Humanoid.pth

# 
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/walk/dataset_reallusion_walk.yaml --checkpoint /data/ASE/output/Humanoid_09-18-59-36-amp-walk-turn/nn/Humanoid.pth


# vmp
#  with sword shield
python ase/run.py --task HumanoidVMP \
    --cfg_env ase/data/cfg/humanoid_sword_shield.yaml \
    --cfg_train ase/data/cfg/train/rlg/vmp_humanoid.yaml \
    --motion_file ase/data/motions/walk/dataset_reallusion_walk.yaml
# without sword shield
python ase/run.py --task HumanoidVMP \
    --cfg_env ase/data/cfg/humanoid_vmp.yaml \
    --cfg_train ase/data/cfg/train/rlg/vmp_humanoid.yaml \
    --motion_file '/data/ASE/ase/data/motions/dataset_amp_walk.yaml'

python ase/run.py --test --task HumanoidVMP --num_envs 16 \
    --cfg_env ase/data/cfg/humanoid_vmp.yaml \
    --cfg_train ase/data/cfg/train/rlg/vmp_humanoid.yaml \
    --motion_file ase/data/motions/walk/dataset_reallusion_walk.yaml \
     --checkpoint /data/ASE/output/Humanoid_12-05-25-15/nn/Humanoid.pth

python ase/run.py --test --task HumanoidViewMotion --num_envs 2 \
    --cfg_env ase/data/cfg/humanoid.yaml \
    --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml \
    --motion_file ase/data/motions/walk/dataset_reallusion_walk.yaml

python ase/run.py --test --task HumanoidViewMotion --num_envs 2  --cfg_env ase/data/cfg/humanoid_ase_getup.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml
