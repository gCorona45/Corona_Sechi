import os
import random
import gc
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
from gymnasium.wrappers import RecordVideo

SEED = 2024

def set_global_seed(seed: int = SEED):
    """
    Establece la semilla global para garantizar la reproducibilidad completa del experimento.
    Sigue las indicaciones de la sección 5.4 del documento de la práctica.
    
    Args:
        seed (int): El número entero a usar como semilla (por defecto 2024).
    """
    print(f"Configurando semilla global (Global Seed): {seed}")

    # 1. Configuración del entorno de Python (Hashing)
    # Esto evita que los diccionarios y sets de Python tengan un orden aleatorio.
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. Configuración del módulo 'random' de Python
    random.seed(seed)

    # 3. Configuración de NumPy
    # Fundamental para cualquier operación matricial o generación de números en CPU.
    np.random.seed(seed)
    
    # 4. Configuración de PyTorch (CPU y GPU)
    # Se fija la semilla para la CPU.
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        # Si hay una GPU disponible (como en Colab), fijamos la semilla también ahí.
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Para configuraciones multi-GPU
        
        # Opciones críticas para garantizar que las operaciones de convolución 
        # sean deterministas (puede ser un poco más lento, pero es necesario para la práctica).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Variable de entorno para depuración de errores en CUDA (opcional pero útil)
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        print(f"Dispositivo GPU detectado y configurado para determinismo.")
    else:
        print("GPU no detectada. Usando CPU.")

def make_env_with_seed(env_name: str, seed: int = SEED):
    """
    Crea un entorno de Gymnasium y asegura que su semilla esté fijada correctamente.
    
    Args:
        env_name (str): El ID del entorno (ej. 'CartPole-v1').
        seed (int): La semilla para el entorno.
        
    Returns:
        env: El entorno configurado.
        obs: La observación inicial.
        info: Información adicional inicial.
    """
    # Creamos el entorno
    env = gym.make(env_name)
    
    # IMPORTANTE: En Gymnasium, la semilla se pasa durante el método reset().
    # Esto reinicia el generador de números aleatorios interno del entorno.
    obs, info = env.reset(seed=seed)
    
    return env, obs, info

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Debug CUDA se necessario
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ==========================================================
# FUNZIONI DI ANALISI METODI TABULARI
# ==========================================================

def analyze_performance_tabular(results_dict):
    summary_data = []
    for name, stats in results_dict.items():
        rewards = np.array(stats['rewards'])
        summary_data.append({
            "Algoritmo": name,
            "Best Reward": np.max(rewards),
            "Avg last 100": f"{np.mean(rewards[-100:]):.2f}",
            "Std Dev": f"{np.std(rewards[-100:]):.2f}"
        })
    return pd.DataFrame(summary_data)

def evaluate_agent(agent, env_name, n_runs=20):
    """Valuta l'agente in modalità puramente deterministica."""
    env = gym.make(env_name)
    eval_rewards = []
    
    for _ in range(n_runs):
        obs, _ = env.reset(options={'start_loc':0, 'goal_loc':63})
        done = False
        total_r = 0
        while not done:
            # Scelta greedy (senza epsilon)
            action = np.argmax(agent.q_table[obs])
            obs, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            done = terminated or truncated
        eval_rewards.append(total_r)
    
    print(f"Resultados de la evaluación {agent.__class__.__name__}:")
    print(f" - AVG: {np.mean(eval_rewards):.2f} | Max: {np.max(eval_rewards)}")
    return eval_rewards

def analyze_q_table(agent):
    """Calcola quanti stati sono stati visitati e quanti hanno valori diversi da zero."""
    visited_states = len(agent.q_table)
    # Conta gli stati dove almeno un'azione ha un valore diverso da 0
    learned_states = sum(1 for state in agent.q_table if np.any(agent.q_table[state] != 0))
    
    return {
        "Estados visitados": visited_states,
        "Estados con conocimiento": learned_states,
        "Cobertura %": (learned_states / visited_states * 100) if visited_states > 0 else 0
    }

def get_agent_trajectory(agent, env_name, grid_size=8):
    """Esegue un episodio e restituisce la lista delle coordinate (riga, col) visitate."""
    env = gym.make(env_name)
    obs, _ = env.reset(options={'start_loc':0, 'goal_loc':63}, seed=SEED)
    
    trajectory = []
    # Aggiungiamo il punto di partenza
    trajectory.append((obs // grid_size, obs % grid_size))
    
    done = False
    max_steps = grid_size * grid_size # Evita loop infiniti se la policy è ciclica
    steps = 0
    
    while not done and steps < max_steps:
        action = np.argmax(agent.q_table[obs])
        obs, reward, terminated, truncated, _ = env.step(action)
        
        row, col = obs // grid_size, obs % grid_size
        trajectory.append((row, col))
        
        done = terminated or truncated
        steps += 1
        
    return trajectory

def analyze_trajectories(all_agents, grid_size=8):
    report = []
    for name, agent in all_agents.items():
        path = get_agent_trajectory(agent, "SimpleGrid-8x8-v0", grid_size)
        length = len(path) - 1 # numero di passi
        
        # Calcoliamo la distanza di Manhattan minima teorica (da 0,0 a 7,7)
        min_dist = (grid_size - 1) + (grid_size - 1)
        
        report.append({
            "Modelo": name,
            "Longitud del recorrido": length,
            "¿Óptimo?": "Sì" if length == min_dist else f"+{length - min_dist} pasos"
        })
    return pd.DataFrame(report)


# ==========================================================
# FUNZIONI DI ANALISI METODI APPROSSIMAZIONE
# ==========================================================

def analyze_performance(results_dict, agents_dict, target_reward=475):
    summary_data = []
    
    for name, rewards in results_dict.items():
        rewards_arr = np.array(rewards)
        
        # Metriche di base
        max_reward = np.max(rewards_arr)
        last_100_avg = np.mean(rewards_arr[-100:]) # Media finale più solida
        
        # Stabilità (Varianza degli ultimi 200 episodi per gestire i 2000 totali)
        stability_std = np.std(rewards_arr[-200:])
        
        # Velocità di Convergenza (Finestra di 100 per evitare falsi positivi)
        window_size = 100
        moving_avg = pd.Series(rewards).rolling(window=window_size).mean().values
        conv_idx = np.where(moving_avg >= target_reward)[0]
        convergence_ep = conv_idx[0] if len(conv_idx) > 0 else "Not reached"
        
        summary_data.append({
            "Algorithm": name,
            "Best Reward": max_reward,
            "Avg last 100": f"{last_100_avg:.2f}",
            "Stability (Std Dev)": f"{stability_std:.2f}",
            "Episode Convergence": convergence_ep
        })
    
    return pd.DataFrame(summary_data)
   
def test_agents(agents_dict, env_name, device, test_episodes=50):
    """
    Valuta gli agenti in modalità deterministica (epsilon=0).
    """
    test_results = []
    # Creiamo l'ambiente per il test
    env = gym.make(env_name)

    for name, agent in agents_dict.items():
        print(f"Testing {name} over {test_episodes} episodes...")
        episode_rewards = []
        
        for ep in range(test_episodes):
            obs, _ = env.reset()
            done = False
            total_r = 0
            
            while not done:
                # Trasformiamo l'osservazione in tensore per la rete neurale
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # Scegliamo SEMPRE l'azione migliore (Greedy)
                    # Per DQN usiamo policy_net, per SARSA q_net
                    if hasattr(agent, 'policy_net'):
                        action = agent.policy_net(state_t).argmax().item()
                    else:
                        action = agent.q_net(state_t).argmax().item()
                
                obs, reward, terminated, truncated, _ = env.step(action)
                total_r += reward
                done = terminated or truncated
            
            episode_rewards.append(total_r)
        
        test_results.append({
            "Algoritmo": name,
            "Recompensa media": np.mean(episode_rewards),
            "Desviación estándar": np.std(episode_rewards),
            "Max Recompensa": np.max(episode_rewards),
            "Min Recompensa": np.min(episode_rewards)
        })

    env.close()
    return pd.DataFrame(test_results)

def record_agent_video(agent, env_name,device, folder="./videos", prefix="test"):
    """
    Registra un intero episodio dell'agente che gioca.
    """
    # Creiamo l'ambiente con il rendering attivo
    env = gym.make(env_name, render_mode="rgb_array")
    
    # Applichiamo il wrapper per la registrazione
    env = RecordVideo(env, video_folder=folder, name_prefix=prefix, 
                      episode_trigger=lambda x: True) # Registra ogni episodio (ne faremo uno solo)

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Trasformiamo l'osservazione in tensore
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Modalità Greedy (Epsilon = 0)
            if hasattr(agent, 'policy_net'):
                action = agent.policy_net(state_t).argmax().item()
            else:
                action = agent.q_net(state_t).argmax().item()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Video recorded! Total reward obtained: {total_reward}")
    env.close()