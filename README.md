# PPO-Continuous-Pytorch
**A clean and robust Pytorch implementation of PPO on continuous action space**:


Pendulum | LunarLanderContinuous
:-----------------------:|:-----------------------:|
<img src="https://github.com/XinJingHao/PPO-Continuous-Pytorch/blob/main/render_gif/PV1.gif" width="80%" height="auto">  | <img src="https://github.com/XinJingHao/PPO-Continuous-Pytorch/blob/main/render_gif/lldcV2.gif" width="80%" height="auto">
<img src="https://github.com/XinJingHao/PPO-Continuous-Pytorch/blob/main/render_gif/pv1_ppoc.png" width="80%" height="auto">  | <img src="https://github.com/XinJingHao/PPO-Continuous-Pytorch/blob/main/render_gif/lldc_ppoc.png" width="80%" height="auto">



**Other RL algorithms by Pytorch can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).**

## Dependencies
```python
gymnasium==0.29.1
numpy==1.26.1
pytorch==2.1.0

python==3.11.5
```

## How to use my code
### Train from scratch
```bash
python main.py
```
where the default enviroment is 'Pendulum'.  

### Play with trained model
```bash
python main.py --EnvIdex 0 --render True --Loadmodel True --ModelIdex 100
```
which will render the 'Pendulum'.  

### Change Enviroment
If you want to train on different enviroments, just run 
```bash
python main.py --EnvIdex 1
```
The ```--EnvIdex``` can be set to be 0~5, where
```bash
'--EnvIdex 0' for 'Pendulum-v1'  
'--EnvIdex 1' for 'LunarLanderContinuous-v2'  
'--EnvIdex 2' for 'Humanoid-v4'  
'--EnvIdex 3' for 'HalfCheetah-v4'  
'--EnvIdex 4' for 'BipedalWalker-v3'  
'--EnvIdex 5' for 'BipedalWalkerHardcore-v3' 
```

Note: if you want train on **BipedalWalker, BipedalWalkerHardcore, or LunarLanderContinuous**, you need to install [box2d-py](https://gymnasium.farama.org/environments/box2d/) first. You can install box2d-py via:
```bash
pip install gymnasium[box2d]
```

if you want train on **Humanoid or HalfCheetah**, you need to install [MuJoCo](https://gymnasium.farama.org/environments/mujoco/) first. You can install MuJoCo via:
```bash
pip install mujoco
pip install gymnasium[mujoco]
```

### Visualize the training curve
You can use the [tensorboard](https://pytorch.org/docs/stable/tensorboard.html) to record anv visualize the training curve. 

- Installation (please make sure PyTorch is installed already):
```bash
pip install tensorboard
pip install packaging
```
- Record (the training curves will be saved at '**\runs**'):
```bash
python main.py --write True
```

- Visualization:
```bash
tensorboard --logdir runs
```

### Hyperparameter Setting
For more details of Hyperparameter Setting, please check 'main.py'

### References
[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)  
[Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf)

## Training Curves
![avatar](https://github.com/XinJingHao/PPO-Continuous-Pytorch/blob/main/ppo_result.jpg)  
All the experiments are trained with same hyperparameters (see main.py). 

