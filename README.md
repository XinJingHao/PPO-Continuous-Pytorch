# PPO-Continuous-Pytorch
I found the current implementation of PPO on continuous action space is whether somewhat complicated or not stable.  
And this is a **clean and robust Pytorch implementation of PPO on continuous action space**. Here is the result:  
  
![avatar](https://github.com/XinJingHao/PPO-Continuous-Pytorch/blob/main/ppo_result.jpg)  
All the experiments are trained with same hyperparameters. **Other RL algorithms by Pytorch can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).**


## Dependencies
gym==0.18.3  
box2d==2.3.10  
numpy==1.21.2  
pytorch==1.8.1  
tensorboard==2.5.0

## How to use my code
### Train from scratch
run **'python main.py'**, where the default enviroment is Pendulum-v0.  
### Play with trained model
run **'python main.py --write False --render True --Loadmodel True --ModelIdex 400'**  
### Change Enviroment
If you want to train on different enviroments, just run **'python main.py --EnvIdex 0'**.  
The --EnvIdex can be set to be 0~5, where   
'--EnvIdex 0' for 'BipedalWalker-v3'  
'--EnvIdex 1' for 'BipedalWalkerHardcore-v3'  
'--EnvIdex 2' for 'LunarLanderContinuous-v2'  
'--EnvIdex 3' for 'Pendulum-v0'  
'--EnvIdex 4' for 'Humanoid-v2'  
'--EnvIdex 5' for 'HalfCheetah-v2' 

P.S. if you want train on 'Humanoid-v2' or 'HalfCheetah-v2', you need to install **MuJoCo** first.
### Visualize the training curve
You can use the tensorboard to visualize the training curve. History training curve is saved at '\runs'
### Hyperparameter Setting
For more details of Hyperparameter Setting, please check 'main.py'
### References
[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)  
[Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf)

