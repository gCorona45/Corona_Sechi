import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

from src.agent import Agent
from src.DQN_Network import DQN_Network


# Device 
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
SEED = 2024

random.seed(2024)
np.random.seed(2024)

class DQNAgent(Agent):
    def __init__(self, env, **hyperparameters):
        super().__init__(env, **hyperparameters)

        # Hyperparameters
        self.gamma = hyperparameters.get("gamma", 0.99)
        self.lr = hyperparameters.get("learning_rate", 1e-3)
        self.batch_size = hyperparameters.get("batch_size", 64)
        self.memory_size = hyperparameters.get("memory_size", 100000)
        self.target_update_freq = hyperparameters.get("target_update_freq", 1000)

        # Epsilon STATIC
        self.epsilon = hyperparameters.get("epsilon", 0.1) # Valore fisso

        self.steps_done = 0
        self.memory = deque(maxlen=self.memory_size)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.policy_net = DQN_Network(self.state_dim, self.action_dim).to(device)
        self.target_net = DQN_Network(self.state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss() 

    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.policy_net(state_t).argmax().item()

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        # --- REWARD SHAPING ---
        adjusted_reward = reward
        
        # Comprobemos si la información necesaria se encuentra en el diccionario info
        if not terminated and "bird" in info and "pipes" in info:
            bird_y = info["bird"]["y"]
            bird_x = info["bird"]["x"]
            pipes = info["pipes"]
            
            # Encuentra el primer tubo que hay delante del pajarito
            upcoming_pipes = [p for p in pipes if p["x"] + 50 > bird_x]
            
            if upcoming_pipes:
                next_pipe = upcoming_pipes[0]
                target_y = next_pipe["bottom"] - 20
                
                dist_v = abs(bird_y - target_y)
                
                # Priorizamos la precisión en relación con este nuevo objetivo
                shaping = max(0, 0.5 - (dist_v / 150.0))
                adjusted_reward += shaping
                        
            #adjusted_reward += 0.1 # Bonus sopravvivenza frame
            
        elif terminated:
            adjusted_reward = -100.0 # Penalización por colisión
        
        # --- GESTIÓN DE LA MEMORIA ---
        done = terminated or truncated
        self.memory.append((obs, action, adjusted_reward, next_obs, done))
        
        self.steps_done += 1

        # --- LÓGICA DE ENTRENAMIENTO ---
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Cálculo de los valores Q actuales
        current_q = self.policy_net(states).gather(1, actions).squeeze()

        # Cálculo de los valores Q objetivo 
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) 
        self.optimizer.step()

        self.training_history.append(loss.item())

        # Update della Target Network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


