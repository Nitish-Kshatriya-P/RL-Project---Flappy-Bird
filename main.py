import torch
from torch import nn

import os
from datetime import datetime
import itertools

import flappy_bird_gymnasium
import gymnasium as gym

import random
import yaml

from my_network import DQN 
from exp_replay import ReplayMemory

DATE_FORMAT = "%m-%d %H:%M:%S"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

SAVED_DIR = "Runs"
os.makedirs(SAVED_DIR , exist_ok = True)

class Agent():
        def __init__(self, sel_env):
            path = 'C:/Users/Admin/Desktop/rl_projects/Flappy_Bird/hyperparameters.yaml'
            with open(path,'r') as f:
                all_paramters = yaml.safe_load(f)
                hyperparameters = all_paramters[sel_env]

            self.sel_env = sel_env

            self.env = hyperparameters['env']
            self.set_rp_memory = hyperparameters['set_rp_memory']
            self.sample_size = hyperparameters['sample_size']
            self.epsilon = hyperparameters['epsilon']
            self.epsilon_decay = hyperparameters['epsilon_decay']
            self.epsilon_min = hyperparameters['epsilon_min']
            self.update_target_dqn = hyperparameters['update_target_dqn']
            self.learning_rate = hyperparameters['learning_rate']
            self.discount_factor = hyperparameters['discount_factor']
            self.set_rewards = hyperparameters['set_rewards']

            self.loss_func = nn.MSELoss()
            self.optimizer = None

            self.MODEL_SAVING_LOGS = os.path.join(SAVED_DIR, f'{self.sel_env}.log')
            self.MODEL_SAVED = os.path.join(SAVED_DIR, f'{self.sel_env}.pt')

        def run(self, is_training = True, is_render = False):
            env = gym.make(self.env, render_mode = "human" if is_render else None, use_lidar = False)

            start_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            policy_dqn = DQN(start_dim, action_dim).to(device)
            
            rewards_per_ep = []

            epsilon_history = []
        
            action_counter = 0
            pr = -9999999
            replay_memory = ReplayMemory(self.set_rp_memory)
            epsilon = self.epsilon

            if is_training:

                target_dqn = DQN(start_dim, action_dim).to(device)
                target_dqn.load_state_dict(policy_dqn.state_dict())

                self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)
            else:
                policy_dqn.load_state_dict(torch.load(self.MODEL_SAVED))


            for ep in itertools.count():
                current_state, _ = env.reset()
                current_state = torch.tensor(current_state, dtype = torch.float32, device = device, requires_grad= False)

                terminated = False

                total_reward = 0.0

                while (not terminated and total_reward < self.set_rewards):
                    if is_training and random.random() < epsilon:
                        action = env.action_space.sample()
                        action = torch.tensor(data = action, dtype = torch.long, device = device)
                    else:
                        with torch.no_grad():
                            action = policy_dqn(current_state.unsqueeze(dim = 0)).squeeze().argmax()

                    new_state, reward, terminated, _, info = env.step(action.item())

                    reward = torch.tensor(reward, dtype= torch.float32, device= device) 
                    new_state = torch.tensor(new_state, dtype = torch.float32, device = device)

                    if is_training:
                        replay_memory.add_exp((current_state, int(action), new_state, reward, terminated))

                        action_counter += 1
                
                    current_state = new_state
                    total_reward += reward.item()

                rewards_per_ep.append(total_reward)
                epsilon = max(epsilon - self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                if is_training:
                    if total_reward > pr:
                        log_message = f"Saving model at {datetime.now().strftime(DATE_FORMAT)}: New best reward is {total_reward:0.1f} & increased by {((total_reward-pr)/pr)*100:+0.1f}% at episode {ep}."
                        print(log_message)
                        with open(self.MODEL_SAVING_LOGS, 'w') as file:
                            file.write(log_message + '\n')

                        torch.save(policy_dqn.state_dict(), self.MODEL_SAVED)
                        pr = total_reward
                        if pr == self.set_rewards:
                            break

                if len(replay_memory)>self.sample_size:
                    get_exps = replay_memory.sample(self.sample_size)

                    self.optimize(get_exps, policy_dqn, target_dqn)

                    if action_counter > self.update_target_dqn:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        action_counter = 0

        def optimize(self, get_exps, policy_dqn, target_dqn):
            current_states, actions, new_states, rewards, terminations = zip(*get_exps)

            current_states = torch.stack(current_states)

            actions = torch.tensor(actions, dtype = torch.long, device=device)

            new_states = torch.stack(new_states)

            rewards = torch.stack(rewards).to(device)

            terminations = torch.tensor(terminations).float().to(device)

            with torch.no_grad():
                target_q = rewards + (1 - terminations)*self.discount_factor*target_dqn(new_states).max(dim = 1)[0]

            current_q = policy_dqn(current_states).gather(1, actions.unsqueeze(1)).squeeze()

            loss = self.loss_func(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def train(env:str):
    bird = Agent(env) 
    bird.run(is_training= True)

def test(env:str):
    bird = Agent(env) 
    bird.run(is_training= False, is_render= True)

if __name__ == '__main__':
    test(env= "FlappyBird-v0")