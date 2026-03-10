import gymnasium as gym
from typing import Any, Dict, Tuple

class Agent:
    def __init__(self, env:gym.Env, **hyperparameters):
        """Inicializa todo lo necesario para el aprendizaje"""
        self.env = env
        self.hyperparameters = hyperparameters
        # Lista para guardar el historial de errores o recompensas durante el entrenamiento
        self.training_history = [] 
    
    def get_action(self, state) -> Any:
        """
        Indicará qué acción realizar de acuerdo al estado.
        Responde a la política del agente.
        Construir tantas funciones de este tipo como políticas se quieran usar.
        """
        raise NotImplementedError("El método 'get_action' debe ser implementado por la subclase.")
    
    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """
        Aplica el algoritmo de aprendizaje para actualizar el conocimiento del agente.
        Recibe la tupla de experiencia (s, a, s', r).
        
        Args:
            obs: Estado actual.
            action: Acción tomada.
            next_obs: Siguiente estado resultante.
            reward: Recompensa recibida.
            terminated: Booleano, indica si el episodio terminó (estado terminal).
            truncated: Booleano, indica si el episodio se cortó por límite de tiempo.
            info: Diccionario con información extra.
        """
        raise NotImplementedError("El método 'update' debe ser implementado por la subclase.")