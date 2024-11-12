# An implementation of VMP from Disney

[VMP Paper](https://la.disneyresearch.com/publication/vmp-versatile-motion-priors-for-robustly-tracking-motion-on-physical-characters/)

This code is largely based upon the implementation of [ASE](https://github.com/nv-tlabs/ASE), 
utilizing the backbone of amp control, with modifications including the stage 1 VAE pretraining, composition of motion frames and motion window.

## Dataset
Currently only using the `amp_humanoid_jog.npy`, `amp_humanoid_walk.npy` and `amp_humanoid_run.npy` for the amp_humanoid. Retargeted motion data will be added shortly.


## Training
```bash
python ase/run.py --task HumanoidVMP \
    --cfg_env ase/data/cfg/humanoid_vmp.yaml \
    --cfg_train ase/data/cfg/train/rlg/vmp_humanoid.yaml \
    --motion_file '/data/ASE/ase/data/motions/dataset_amp_walk.yaml'
```
## Evaluation
```bash
python ase/run.py --test --task HumanoidVMP --num_envs 16 \
    --cfg_env ase/data/cfg/humanoid_vmp.yaml \
    --cfg_train ase/data/cfg/train/rlg/vmp_humanoid.yaml \
    --motion_file ase/data/motions/dataset_amp_walk.yaml \
    --checkpoint [checkpoint under /data/ASE/output/]
```

## Technical details

### VAE pretraining
See [train_latent_VAE.py](https://github.com/DavidLXu/VMP/blob/main/vmp/train_latent_VAE.py)

### Motion frames and sliding windows
Refer to [humanoid_vmp.py](https://github.com/DavidLXu/VMP/blob/main/ase/env/tasks/humanoid_vmp.py)

`self.get_vmp_obs()` is used to construct motion frame $m_t = \{ h_t, \theta_t, v_t, q_t, \dot{q}_t, p_t \}$ in the paper.
Note that $\theta_t$ is a 6D rotation representation proposed in [On the Continuity of Rotation Representations in Neural Networks](https://arxiv.org/abs/1812.07035)

`self.fetch_vmp_obs_demo()` is used to construct the window slides, where the info is passed into self.extras["vmp_obs_window"] to be used in [vmp_agent.py](https://github.com/DavidLXu/VMP/blob/main/ase/learning/vmp_agent.py)

### Integration of VAE encoder in rl_games
Refer to [vmp_agent.py](https://github.com/DavidLXu/VMP/blob/main/ase/learning/vmp_agent.py), [vmp_models.py](https://github.com/DavidLXu/VMP/blob/main/ase/learning/vmp_models.py), 
[vmp_network_builder.py](https://github.com/DavidLXu/VMP/blob/main/ase/learning/vmp_network_builder.py) and [vmp_players.py](https://github.com/DavidLXu/VMP/blob/main/ase/learning/vmp_players.py)

### Rewards
Refer to `compute_vmp_reward()` in [humanoid_vmp.py](https://github.com/DavidLXu/VMP/blob/main/ase/env/tasks/humanoid_vmp.py). Seems that there is also a lower level reward being used by amp codebase in training. Will be fixed.
