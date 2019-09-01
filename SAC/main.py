import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
import os 
import csv
import gym_test


_max_episode_steps=500
env_name = "hand_cube-v0" #name of the environment to run
policy = "Gaussian" #algorithm to use: Gaussian | Deterministic
evaluate = True #Evaluates a policy a policy every 10 episode (default:True)
gamma = 0.99 #discount factor for reward (default: 0.99)
tau = 0.005 #target smoothing coefficient(τ) (default: 0.005)
lr = 0.0003 #learning rate (default: 0.0003)
alpha = 0.2 #Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)
automatic_entropy_tuning = False #Temperature parameter α automaically adjusted
seed = 456 #random seed (default: 456)
batch_size = 256 #batch size (default: 256)
num_steps = 100*(1000001) #maximum number of steps (default: 1000000)
hidden_size = [512,128] #hidden size (default: 256)
updates_per_step = 1 #model updates per simulator step (default: 1)
start_steps = 10000 #Steps sampling random actions (default: 10000)
target_update_interval = 1 #Value target update per no. of updates per step (default: 1)
replay_size = 1000000 #size of replay buffer (default: 10000000)
cuda = "store_true"
rpy_target = [1.6599062326296403, 6.982997797361229e-08, 0.06913593223957717]
# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(env_name)
torch.manual_seed(seed)
np.random.seed(seed)
env.seed(seed)

#House Keeping
rewardFunctionVersion =2
nnVersion =2

old_episode_reward = -1000000000000000000000000000000000000

current_episode = 0

filename = "reward_each_episode_{}_{}_{}.csv".format(env_name,nnVersion,rewardFunctionVersion)
directory = "./"
fileExists = os.path.isfile(directory+filename)
if fileExists:
    with open(filename) as csvfile:
          #print("im inside:::")
          mLines = csvfile.readlines()
          if len(mLines):
              targetline = mLines[-1]
              current_episode = targetline.split(',')[1]
# Agent
agent_args ={"gamma":gamma,"tau":tau,"alpha":alpha,"policy":policy,
             "target_update_interval":target_update_interval,"automatic_entropy_tuning":automatic_entropy_tuning,
             "cuda":cuda,"hidden_size":hidden_size,"lr":lr}
# print("env.action_space",env.action_space)
agent = SAC(env.observation_space.shape[0], env.action_space, agent_args)
"""
#Loading model from last training if it exists #TODO
filename = "PPO_continuous_" +env_name+"_"+"_"+str(rewardFunctionVersion)+"_"+str(nnVersion)+".pth"
directory = "./"
fileExists = os.path.isfile(directory+filename)
print("fileExists:::::",fileExists)
if fileExists:
    agent.load_model(actor_path, critic_path)
"""
#TesnorboardX
writer = SummaryWriter(logdir='runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), env_name,
                                                             policy, "autotune" if automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(replay_size)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()[0]
    
    if total_numsteps !=0 and episode_reward > old_episode_reward: 
        print("\n\n\n")
        print("saving model....")
        
        agent.save_model(env_name,suffix=str(nnVersion)+"_"+str(rewardFunctionVersion))
        print("\n\n\n")

    # action1=[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    if start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
            # print("action::",action)
    else:
        action = agent.select_action(state)  # Sample action from policy
    while not done:
        if start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
            # print("action::",action)
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > batch_size:
            # Number of updates per step in environment
            for i in range(updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        # print("action",action)
        print("----episod-----",i_episode)
        print("---step---",total_numsteps)
        next_state, reward, done,_ = env.step(action,rpy_target) # Step
        next_state = next_state[0]
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        print("------reward------",reward)
                

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == _max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > num_steps:

        break

    #stroing training data
    with open("reward_each_episode_{}_{}_{}.csv".format(env_name,nnVersion,rewardFunctionVersion), "a+", newline ="") as csvfile:

        CSVwriter = csv.writer(csvfile)
        episode = int(current_episode) + i_episode
        CSVwriter.writerow([episode_reward,episode])
    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and eval == True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, eval=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        

        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()

