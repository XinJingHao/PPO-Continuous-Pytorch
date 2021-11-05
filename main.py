import numpy as np
import torch
import gym
from PPO import PPO, device
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=3, help='BWv3, BWHv3, Lch_Cv2, PV0, Humanv2, HCv2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=400, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
parser.add_argument('--distnum', type=int, default=0, help='0:Beta ; 1:GS_ms  ;  2: GS_m')
parser.add_argument('--Max_train_steps', type=int, default=5e7, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=5e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=150, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=2e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=2e-4, help='Learning rate of critic')
parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
parser.add_argument('--a_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of actor')
parser.add_argument('--c_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of critic')
parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
opt = parser.parse_args()
print(opt)


def Action_adapter(a,max_action):
    #from [0,1] to [-max,max]
    return  2*(a-0.5)*max_action

def Reward_adapter(r, EnvIdex):
    # For BipedalWalker
    if EnvIdex == 0 or EnvIdex == 1:
        if r <= -100: r = -1
    # For Pendulum-v0
    elif EnvIdex == 3:
        r = (r + 8) / 8
    return r

def evaluate_policy(env, model, render, steps_per_epoch, max_action, EnvIdex):
    scores = 0
    turns = 3
    for j in range(turns):
        s, done, ep_r, steps = env.reset(), False, 0, 0
        while not (done or (steps >= steps_per_epoch)):
            # Take deterministic actions at test time
            a, logprob_a = model.evaluate(s)
            act = Action_adapter(a, max_action)  # [0,1] to [-max,max]
            s_prime, r, done, info = env.step(act)
            # r = Reward_adapter(r, EnvIdex)

            ep_r += r
            steps += 1
            s = s_prime
            if render:
                env.render()
        scores += ep_r
    return scores/turns

def main():

    write = opt.write   #Use SummaryWriter to record the training.
    render = opt.render

    EnvName = ['BipedalWalker-v3','BipedalWalkerHardcore-v3','LunarLanderContinuous-v2','Pendulum-v0','Humanoid-v2','HalfCheetah-v2']
    BriefEnvName = ['BWv3', 'BWHv3', 'Lch_Cv2', 'PV0', 'Humanv2', 'HCv2']
    Env_With_Dead = [True, True, True, False, True, False]
    EnvIdex = opt.EnvIdex
    env_with_Dead = Env_With_Dead[EnvIdex]  #Env like 'LunarLanderContinuous-v2' is with Dead Signal. Important!
    env = gym.make(EnvName[EnvIdex])
    eval_env = gym.make(EnvName[EnvIdex])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_steps = env._max_episode_steps
    print('Env:',EnvName[EnvIdex],'  state_dim:',state_dim,'  action_dim:',action_dim,
          '  max_a:',max_action,'  min_a:',env.action_space.low[0], 'max_steps', max_steps)
    T_horizon = opt.T_horizon  #lenth of long trajectory


    Dist = ['Beta', 'GS_ms', 'GS_m'] #type of probility distribution
    distnum = opt.distnum

    Max_train_steps = opt.Max_train_steps
    save_interval = opt.save_interval#in steps
    eval_interval = opt.eval_interval#in steps

    random_seed = opt.seed
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    eval_env.seed(random_seed)
    np.random.seed(random_seed)

    if write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BriefEnvName[EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "env_with_Dead":env_with_Dead,
        "gamma": opt.gamma,
        "lambd": opt.lambd,     #For GAE
        "clip_rate": opt.clip_rate,  #0.2
        "K_epochs": opt.K_epochs,
        "net_width": opt.net_width,
        "a_lr": opt.a_lr,
        "c_lr": opt.c_lr,
        "dist": Dist[distnum],
        "l2_reg": opt.l2_reg,   #L2 regulization for Critic
        "a_optim_batch_size":opt.a_optim_batch_size,
        "c_optim_batch_size": opt.c_optim_batch_size,
        "entropy_coef":opt.entropy_coef, #Entropy Loss for Actor: Large entropy_coef for large exploration, but is harm for convergence.
        "entropy_coef_decay":opt.entropy_coef_decay
    }
    # if Dist[distnum] == 'Beta' :
    #     kwargs["a_lr"] *= 2 #Beta dist need large lr|maybe
    #     kwargs["c_lr"] *= 4  # Beta dist need large lr|maybe

    if not os.path.exists('model'): os.mkdir('model')
    model = PPO(**kwargs)
    if opt.Loadmodel: model.load(opt.ModelIdex)

    traj_lenth = 0
    total_steps = 0
    while total_steps < Max_train_steps:
        s, done, steps, ep_r = env.reset(), False, 0, 0

        '''Interact & trian'''
        while not done:
            traj_lenth += 1
            steps += 1

            if render:
                env.render()
                a, logprob_a = model.evaluate(s)
            else:
                a, logprob_a = model.select_action(s)

            act = Action_adapter(a,max_action) #[0,1] to [-max,max]
            s_prime, r, done, info = env.step(act)
            r = Reward_adapter(r, EnvIdex)

            '''distinguish done between dead|win(dw) and reach env._max_episode_steps(rmax); done = dead|win|rmax'''
            '''dw for TD_target and Adv; done for GAE'''
            if done and steps != max_steps:
                dw = True
                #still have exception: dead or win at _max_episode_steps will not be regard as dw.
                #Thus, decide dw according to reward signal of each game is better.  dw = done_adapter(r)
            else:
                dw = False

            model.put_data((s, a, r, s_prime, logprob_a, done, dw))
            s = s_prime
            ep_r += r

            '''update if its time'''
            if not render:
                if traj_lenth % T_horizon == 0:
                    model.train()
                    traj_lenth = 0


            '''record & log'''
            if total_steps % eval_interval == 0:
                score = evaluate_policy(eval_env, model, False, max_steps, max_action, EnvIdex)
                if write:
                    writer.add_scalar('ep_r_insteps', score, global_step=total_steps)
                print('EnvName:',EnvName[EnvIdex],'seed:',random_seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)
            total_steps += 1

            '''save model'''
            if total_steps % save_interval==0:
                model.save(total_steps)

    env.close()

if __name__ == '__main__':
    main()





