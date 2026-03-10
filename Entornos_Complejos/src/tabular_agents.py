import numpy as np
from collections import defaultdict
import random
from src.agent import Agent  # classe base
SEED = 2024

random.seed(2024)
np.random.seed(2024)

class TabularAgent(Agent):
    def __init__(self, env, learning_rate=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(env)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table inizializzata a zero per ogni stato-azione
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n)) 

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # Invece di np.argmax, scegli casualmente tra i massimi
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        actions_with_max_q = np.where(q_values == max_q)[0]
        return np.random.choice(actions_with_max_q)

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        raise NotImplementedError("It must be implemented in a subclass.")
        
class MonteCarloOnPolicyAgent(TabularAgent):
    """
    Non imparano passo dopo passo, ma aspettano 
    la fine dell'episodio per vedere com'è andata a finire.
    """
    
    def __init__(self, env, gamma=0.99, epsilon=1.0):
        super().__init__(env, learning_rate=0.1, gamma=gamma, epsilon=epsilon)
        self.episode_memory = []
        self.returns_count = defaultdict(lambda: np.ones(env.action_space.n)*5)

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        # aggiorna la lista degli episodi con la nuova transizione
        self.episode_memory.append((obs, action, reward)) 

    def end_episode(self):
        G = 0
        visited = set()
        for state, action, reward in reversed(self.episode_memory): # parte dall'ultima azione fatta
            G = self.gamma * G + reward
            # First-visit check
            if (state, action) not in visited:
                visited.add((state, action))
                self.returns_count[state][action] += 1
                N = self.returns_count[state][action]
                old_Q = self.q_table[state][action]
                # Più volte vedi un risultato, più la tua stima di quel risultato diventa solida.
                self.q_table[state][action] += (1/N) * (G - old_Q)
        self.episode_memory = []
               
class MonteCarloOffPolicyAgent(TabularAgent):
    def __init__(self, env, gamma=0.99, epsilon=1.0):
        super().__init__(env, learning_rate=0.0, gamma=gamma, epsilon=epsilon)
        self.episode_memory = []
        self.C = defaultdict(lambda: np.ones(env.action_space.n)*5)

    def get_target_action(self, state):
        return np.argmax(self.q_table[state])

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        self.episode_memory.append((obs, action, reward))

    def end_episode(self):
        G = 0.0
        W = 1.0 
        for state, action, reward in reversed(self.episode_memory):
            G = self.gamma * G + reward
            self.C[state][action] += W
            
            # Aggiornamento incrementale con Weighted Importance Sampling
            self.q_table[state][action] += (W / self.C[state][action]) * (G - self.q_table[state][action])
            
            # Se l'azione presa dalla behavior policy non è quella che avrebbe preso la target (greedy)
            # la probabilità della target policy diventa 0 e il peso W crolla a 0.
            if action != self.get_target_action(state):
                break
            
            # Calcolo probabilità della behavior policy (Epsilon-Greedy)
            # n_actions = self.env.action_space.n
            # prob = (1 - self.epsilon) + (self.epsilon / n_actions)
            prob_behavior = (1 - self.epsilon) + (self.epsilon / self.env.action_space.n)
            
            W = W * (1.0 / prob_behavior)
            
        self.episode_memory = []

class SarsaAgent(TabularAgent):
    # Aggiunto 'info' per matchare la firma della classe base
    def update(self, obs, action, next_obs, next_action, reward, terminated, truncated, info=None):
        done = terminated or truncated
        next_q = 0.0 if done else self.q_table[next_obs][next_action]
        
        # TD Target = R + gamma * Q(s', a')
        target = reward + self.gamma * next_q
        self.q_table[obs][action] += self.lr * (target - self.q_table[obs][action])
        
class QLearningAgent(TabularAgent):
    
    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        done = terminated or truncated
        
        max_next_q = 0.0 if done else np.max(self.q_table[next_obs])
        
        target = reward + self.gamma * max_next_q
        error = target - self.q_table[obs][action]
        self.q_table[obs][action] += self.lr * error