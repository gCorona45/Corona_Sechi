import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from src.agent import Agent 
from src.DQN_Network import DQN_Network 

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
    
SEED = 2024

random.seed(2024)
np.random.seed(2024)
    
class SARSASemiGradientAgent(Agent):
    def __init__(self, env: gym.Env, **hyperparameters):
        super().__init__(env)
        
        # Hiperparámetros
        self.lr = hyperparameters.get("learning_rate", 1e-3)
        self.gamma = hyperparameters.get("gamma", 0.99)
        
        # Epsilon STATIC
        self.epsilon = hyperparameters.get("epsilon", 0.1)
        
        # Red y Optimización
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.q_net = DQN_Network(self.state_dim, self.action_dim).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss() 
        self.last_next_action = None 

    def get_action(self, state):
        # Utilizza epsilon statico
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.q_net(state_t).argmax().item()

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        # --- 1. REWARD SHAPING AVANZATO ---
        adjusted_reward = reward
        
        # Controlliamo se le informazioni necessarie sono presenti nel dizionario info
        if not terminated and "bird" in info and "pipes" in info:
            bird_y = info["bird"]["y"]
            bird_x = info["bird"]["x"]
            pipes = info["pipes"]
            
            # Trova il primo tubo davanti all'uccellino
            upcoming_pipes = [p for p in pipes if p["x"] + 50 > bird_x]
            
            if upcoming_pipes:
                next_pipe = upcoming_pipes[0]
                
                target_y = next_pipe["bottom"] - 20 # Resta 20 pixel sopra il tubo di sotto
                
                dist_v = abs(bird_y - target_y)
                
                # Premiamo la precisione rispetto a questo nuovo target
                shaping = max(0, 0.5 - (dist_v / 150.0))
                adjusted_reward += shaping
                        
            #adjusted_reward += 0.1 # Bonus sopravvivenza frame
            
        elif terminated:
            adjusted_reward = -20.0 # Penalità per collisione
        
        
        # Scegliamo A' basato su epsilon statico
        next_action = self.get_action(next_obs)
        self.last_next_action = next_action 
        
        # Aggiunta unsqueeze(0) per coerenza con la rete (Batch Size = 1)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
        
        q_values = self.q_net(obs_t)
        q_value = q_values[0, action] # Accesso indicizzato corretto per (1, action_dim)
        
        with torch.no_grad():
            if terminated:
                target = torch.tensor(adjusted_reward, dtype=torch.float32).to(device)
            else:
                next_q = self.q_net(next_obs_t)[0, next_action]
                target = adjusted_reward + self.gamma * next_q
        
        loss = self.criterion(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_history.append(loss.item())