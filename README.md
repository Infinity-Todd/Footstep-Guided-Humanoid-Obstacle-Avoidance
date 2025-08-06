# 🤖 Footstep-Guided Humanoid Obstacle Avoidance
<div align="center">

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MuJoCo 3.3+](https://img.shields.io/badge/MuJoCo-3.3%2B-2A9D8F)](https://mujoco.org/)
[![Gym 0.26+](https://img.shields.io/badge/Gym-0.26%2B-9cf)](https://www.gymlibrary.dev/)
[![TensorBoard 2.20+](https://img.shields.io/badge/TensorBoard-2.20%2B-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/tensorboard)
[![Ray 2.40+](https://img.shields.io/badge/Ray-2.40%2B-028CF0)](https://www.ray.io/)
[![Apple Silicon • MPS](https://img.shields.io/badge/Apple%20Silicon-MPS-000000?logo=apple&logoColor=white)](https://pytorch.org/docs/stable/notes/mps.html)
[![Conda Env](https://img.shields.io/badge/Conda-Env-44A833?logo=anaconda&logoColor=white)](https://docs.conda.io/)
</div>

---
## 🌟 Introduction
This project implements a MuJoCo + PyTorch (SB3) training setup for a **JRVC-1 humanoid** to slalom through obstacles by tracking a **footstep plan**. It supports:

- ✅ Footstep-plan guided locomotion
- ✅ Custom reward shaping (target tracking, balance regulation, collision penalties)
- ✅ Custom observation design (root/feet kinematics, obstacle encoding)
- ✅ Vectorized environments for faster training
- ✅ TensorBoard logging
- ✅ Apple Silicon **MPS** support

A trainable JRVC-1 slalom environment with footstep-guided obstacle avoidance.

<p align="center">
  <img src="media/slalom_demo.gif" alt="JRVC-1 slalom demo" width="720">
  <br><em>JRVC-1 slalom with a footstep plan (MuJoCo)</em>
</p>
 
 ---
 ## 📦 Code Structure

```text
footstep-humanoid/
├── run_experiment.py  # Main entry (train / eval / watch)
├── envs/              # Actions & observations, PD gains, sim step, control decimation, init/reset
├── tasks/             # Reward shaping, terminations/constraints, curricula
├── rl/                # PPO loop, actor/critic networks, observation normalization, callbacks, vectorized env
├── robots/            # JRVC-1 robot configs/parameters
├── models/            # MuJoCo assets: XMLs / meshes / textures
├── utils/             # Footstep-plan generation & small helpers
├── media/             # Demos (e.g., slalom_demo.mp4)
├── logs/              # TensorBoard logs & run metadata
├── requirements.txt   # Requirements
└── README.md          # Project readme
```
---
## 🔧 Requirments:
- **Python version:** 3.12.4
- **pip install:**
  - mujoco==3.3.4
  - torch==2.5.1
  - gym==0.26.2
  - transforms3d==0.4.2
  - scipy==1.16.0
  - tensorboard==2.20.0
  - **optional:**
    - ray==2.40.0
    - mujoco-python-viewer==0.1.4
    - imageio==2.37.0
    - pillow==11.3.0
    - matplotlib==3.10.5
    - tqdm==4.67.1
    - rich==14.0.0
    - pyyaml==6.0.2

---
## 🚀 Usage
#### **To train:** 
```
python run_experiment.py train --logdir <path_to_exp_dir> --num_procs <num_of_cpu_procs> --env jvrc_obstacle --continued <path_to_the_pretrained model>
```  

#### **To play:**
```
mjpython run_experiment.py eval --logdir <path_to_actor_pt>
```

#### **See my Demo:**
```
mjpython run_experiment.py eval --path logs/obstacle_avoidance_v1/actor_best.pt   
```
---
## 🖊️Citation
If you find this work useful in your own research, please cite the following works:

Papers:
- [**Learning Bipedal Walking On Planned Footsteps For Humanoid Robots**](https://arxiv.org/pdf/2207.12644.pdf)

Code:
- [**LearningHumanoidWalking (GitHub)**](https://github.com/rohanpsingh/LearningHumanoidWalking)

```bibtex
@software{singh_learninghumanoidwalking,
  author       = {Rohan P. Singh and contributors},
  title        = {LearningHumanoidWalking},
  year         = {2022},
  url          = {https://github.com/rohanpsingh/LearningHumanoidWalking},
  note         = {GitHub repository, commit: <commit-hash>, accessed: <YYYY-MM-DD>}
}

@inproceedings{singh2022learning,
  title        = {Learning Bipedal Walking on Planned Footsteps for Humanoid Robots},
  author       = {Singh, Rohan P. and Benallegue, Mehdi and Morisawa, Mitsuharu and Cisneros, Rafael and Kanehiro, Fumio},
  booktitle    = {2022 IEEE-RAS International Conference on Humanoid Robots (Humanoids)},
  pages        = {686--693},
  year         = {2022},
  organization = {IEEE}
}
```


