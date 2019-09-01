import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
import gym_test
import csv
import os
from sklearn import preprocessing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = torch.squeeze(self.actor(state))
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def main():
    ############## Hyperparameters ##############
    env_name="hand_cube-v0"
    render = False
    solved_reward = 4        # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 1000        # max training episodes
    max_timesteps = 1000        # max timesteps in one episode
    
    update_timestep = 4000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0003                 # parameters for Adam optimizer
    old_running_reward=-100000000000
    betas = (0.9, 0.999)
    # rpy_target = [1.6599062326296403, 6.982997797361229e-08, 0.06913593223957717]
    rpy_target = [1.5433787039535989, 0.08179179218080275, 1.5847075734647111]
    random_seed = None
    #############################################
    nnVersion=23
    rewardFunctionVersion=1
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    current_episode = 0
    min_reward=-4
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

    # training loop
    for i_episode in range(1, max_episodes+1):
       
        state = env.reset()

        episode_reward=0
        print("\n\n\n")
        std_scale = preprocessing.StandardScaler().fit(state)
        state= std_scale.transform(state)
        for t in range(max_timesteps):
            print("----episode---------",i_episode)
            print("------time---------",t)
            # std_scale = preprocessing.StandardScaler().fit(state)
            # state = std_scale.transform(state)
            time_step +=1
            # Running policy_old:
            # state=state/norm(state)
            action = ppo.select_action(state, memory)
            # print("\n\n\n")
            # print("action",action)
            # print("\n\n\n")
            state, reward, done, _ = env.step(action,rpy_target)
            if min_reward < reward:
                min_reward = reward
            print("------reward---------",reward)
            # Saving reward:
            memory.rewards.append(reward)
            
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
            
        avg_length += t
        with open("reward_each_episode_{}_{}_{}.csv".format(env_name,nnVersion,rewardFunctionVersion), "a+", newline ="") as csvfile:

            CSVwriter = csv.writer(csvfile)
            episode = int(current_episode) + i_episode
            CSVwriter.writerow([episode_reward,episode])
        with open("reward_ave_episode_{}_{}_{}.csv".format(env_name,nnVersion,rewardFunctionVersion), "a+", newline ="") as csvfile:

            CSVwriter = csv.writer(csvfile)
            episode = int(current_episode) + i_episode
            CSVwriter.writerow([episode_reward/t,episode])
        # CSVwriter.add_scalar('reward/train', running_reward, i_episode)
        

        if running_reward > (log_interval*solved_reward):
        # if reward > solved_reward:
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_hand_solved_{}_{}_{}.pth'.format(i_episode,nnVersion,rewardFunctionVersion))
            break
        if reward > solved_reward:
            torch.save(ppo.policy.state_dict(), './PPO_solve_episode{}_{}_{}.pth'.format(i_episode,nnVersion,reward))
        # save every 500 episodes
        if i_episode % 500== 0:
            torch.save(ppo.policy.state_dict(), './PPO_hand_episode{}_{}_{}.pth'.format(i_episode,nnVersion,rewardFunctionVersion))
        if i_episode % 1000== 0:    
            print("min reward",min_reward)
        # average 
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {}_{}_{} \t Avg length: {} \t Avg reward: {}'.format(i_episode,nnVersion,rewardFunctionVersion, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
    
            
if __name__ == '__main__':
    main()
    
