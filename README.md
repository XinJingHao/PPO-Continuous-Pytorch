# PPO-Continuous-Pytorch
I found the current implementation of PPO on continuous action space is whether somewhat complicated or not stable, and this is a **clean and robust Pytorch implementation of PPO on continuous action space**. Here is the result:  
  
![avatar](https://github.com/XinJingHao/PPO-Continuous-Pytorch/blob/main/ppo_result.jpg)  
All the experiments are trained with same hyperparameters (see main.py). **Other RL algorithms by Pytorch can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).**


## Dependencies
```bash
gymnasium==0.29.1  (https://gymnasium.farama.org/)
box2d-py==2.3.5  (https://pypi.org/project/box2d-py/)
numpy==1.26.1  (https://numpy.org/)
pytorch==2.1.0  (https://pytorch.org/)
tensorboard==2.15.1  (https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html?highlight=tensorboard)

python==3.11.5
```

## How to use my code
### Train from scratch
```bash
python main.py
```
where the default enviroment is Pendulum-v0.  

### Play with trained model
```bash
python main.py --write False --render True --Loadmodel True --ModelIdex 400
```

### Change Enviroment
If you want to train on different enviroments, just run 
```bash
python main.py --EnvIdex 0
```

The --EnvIdex can be set to be 0~5, where   
```bash
'--EnvIdex 0' for 'BipedalWalker-v3'  
'--EnvIdex 1' for 'BipedalWalkerHardcore-v3'  
'--EnvIdex 2' for 'LunarLanderContinuous-v2'  
'--EnvIdex 3' for 'Pendulum-v0'  
'--EnvIdex 4' for 'Humanoid-v2'  
'--EnvIdex 5' for 'HalfCheetah-v2' 
```

P.S. 

if you want train on BipedalWalker-v3, BipedalWalkerHardcore-v3, or LunarLanderContinuous-v2, you need to install **box2d-py** first.

if you want train on Humanoid-v2 or HalfCheetah-v2, you need to install **MuJoCo** first.

### Visualize the training curve
You can use the [tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html?highlight=tensorboard) to visualize the training curve. History training curve is saved at '\runs'

### Hyperparameter Setting
For more details of Hyperparameter Setting, please check 'main.py'

### References
[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)  
[Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf)

