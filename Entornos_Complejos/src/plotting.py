import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import get_agent_trajectory

# ==========================================================
# FUNZIONI DI VISUALIZZAZIONE METODI TABULARI
# ==========================================================

def plot_tabular_results(results_dict):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for name, stats in results_dict.items():
        rewards = pd.Series(stats['rewards'])
        # Media Mobile (SMA)
        means = rewards.rolling(window=100).mean()
        # Deviazione Standard per l'ombra
        stds = rewards.rolling(window=100).std()
        
        ax.plot(means, label=f"{name}", linewidth=2)
        ax.fill_between(range(len(means)), means - stds, means + stds, alpha=0.1)
    
    ax.set_title("Comparación de métodos tabulares", fontsize=14)
    ax.set_xlabel("Episodios")
    ax.set_ylabel("Recompensa (Media móvil 100 ép)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

def plot_training_results(stats,name, smoothing_window=50):
    """
    Visualizza l'andamento del training: Reward e Epsilon.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Plot dei Reward (Asse Sinistro) ---
    color_reward = 'tab:blue'
    ax1.set_xlabel('Episodios')
    ax1.set_ylabel('Recompensa Totale', color=color_reward)
    
    # Calcolo media mobile per pulire il rumore
    rewards_smoothed = np.convolve(stats['rewards'], np.ones(smoothing_window)/smoothing_window, mode='valid')
    
    ax1.plot(stats['rewards'], alpha=0.3, color=color_reward, label='Recompensa')
    ax1.plot(range(smoothing_window-1, len(stats['rewards'])), rewards_smoothed, 
             color='navy', linewidth=2, label=f'Media móvil ({smoothing_window})')
    ax1.tick_params(axis='y', labelcolor=color_reward)
    ax1.grid(True, alpha=0.3)

    # --- Plot della Epsilon ---
    ax2 = ax1.twinx() 
    color_eps = 'tab:red'
    ax2.set_ylabel('Valor Epsilon', color=color_eps)
    ax2.plot(stats['epsilons'], color=color_eps, linestyle='--', linewidth=2, label='Epsilon Decay')
    ax2.tick_params(axis='y', labelcolor=color_eps)

    plt.title(f"{name}")
    fig.tight_layout()
    plt.show()

def plot_q_values_and_policy(agent, grid_size=8):
    # Prepariamo la matrice per l'intensità (Valore massimo di Q per ogni stato)
    q_intensity = np.zeros((grid_size, grid_size))
    
    # Mapping nomi azioni
    #action_names = {0: 'Su (↑)', 1: 'Giù (↓)', 2: 'Sinistra (←)', 3: 'Destra (→)'}
    
    for state in range(grid_size * grid_size):
        row = state // grid_size
        col = state % grid_size
        if state in agent.q_table:
            # Prendiamo il valore Q massimo per colorare la heatmap
            max_q = np.max(agent.q_table[state])
            q_intensity[row, col] = max_q

    plt.figure(figsize=(10, 8))
    
    # Heatmap basata sull'intensità dei valori Q (scala di rossi)
    # origin='upper' mette lo stato 0 (0,0) in alto a sinistra
    im = plt.imshow(q_intensity, cmap='Reds', origin='upper', interpolation='nearest')
    plt.colorbar(im, label='Valor Q máximo (Confidence)')
    
    # Aggiungiamo le frecce per la politica
    for state in range(grid_size * grid_size):
        row = state // grid_size
        col = state % grid_size
        if state in agent.q_table and np.max(agent.q_table[state]) > 0:
            action = np.argmax(agent.q_table[state])
            dx, dy = 0, 0
            if action == 0: dy = -0.3 # Su (decrementa indice riga)
            elif action == 1: dy = 0.3 # Giù (incrementa indice riga)
            elif action == 2: dx = -0.3 # Sinistra
            elif action == 3: dx = 0.3 # Destra
            
            plt.arrow(col, row, dx, dy, head_width=0.15, head_length=0.15, color='black', alpha=0.6)

    # Legenda delle azioni
    #legend_patches = [mpatches.Patch(color='white', label=action_names[i], ec='black') for i in range(4)]
    #plt.legend(handles=legend_patches, title="Direzioni", bbox_to_anchor=(1.25, 1), loc='upper left')

    plt.title(f"Análisis Q-Table: {agent.__class__.__name__}\n Intensidad del color = Valor Q | Frecce = Política Greedy")
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.show()
        
def plot_all_paths(all_agents, env_name="SimpleGrid-8x8-v0", grid_size=8):
    """Visualizza i percorsi di tutti i modelli su una singola griglia."""
    plt.figure(figsize=(10, 10))
    
    # Creiamo una griglia di sfondo pulita
    background = np.zeros((grid_size, grid_size))
    plt.imshow(background, cmap='Greys', alpha=0.1)
    
    # Colori distinti per ogni modello
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for (name, agent), color in zip(all_agents.items(), colors):
        path = get_agent_trajectory(agent, env_name, grid_size)
        
        if not path: continue
        
        rows, cols = zip(*path)
        
        # Disegniamo la linea del percorso
        # Aggiungiamo un piccolo "jitter" (offset) per non sovrapporre perfettamente le linee
        offset = np.random.uniform(-0.1, 0.1) 
        plt.plot(np.array(cols) + offset, np.array(rows) + offset, 
                 label=name, color=color, marker='o', markersize=4, linewidth=2)
        
        # Segniamo la partenza e l'arrivo solo una volta
        plt.scatter(cols[0], rows[0], color='green', s=100, marker='s', zorder=5) # Start
        plt.scatter(cols[-1], rows[-1], color='red', s=100, marker='*', zorder=5)  # Goal

    plt.title("Confronto Percorsi Reali degli Agenti")
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()



# ==========================================================
# FUNZIONI DI VISUALIZZAZIONE METODI APPROSSIMAZIONE
# ==========================================================


def plot_advanced_analysis_apx(results_dict, agents_dict):
    fig, ax = plt.subplots(2, 1, figsize=(12, 14))
    
    # --- Grafico 1: Loss History ---
    for name, agent in agents_dict.items():
        loss_history = np.array(agent.training_history)
        if len(loss_history) > 1000:
            # Media mobile molto larga per la loss e sottocampionamento ogni 100 step
            window_loss = 100
            smooth_loss = np.convolve(loss_history, np.ones(window_loss)/window_loss, mode='valid')[::100]
            x_loss = np.linspace(0, len(loss_history), len(smooth_loss))
            ax[0].plot(x_loss, smooth_loss, label=f"Trend Loss {name}", alpha=0.8)
    
    ax[0].set_title("Loss Analysis", fontsize=14)
    ax[0].set_yscale('log')
    ax[0].set_xlabel("Updates (Steps)")
    ax[0].set_ylabel("MSE Loss (log)")
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=0.3)

    # --- Grafico 2: Reward con Ombra di Stabilità ---
    for name, rewards in results_dict.items():
        window = 100 # Finestra larga per 2000 episodi
        df_rewards = pd.Series(rewards)
        means = df_rewards.rolling(window=window).mean()
        stds = df_rewards.rolling(window=window).std()
        
        ax[1].plot(means, label=f"Moving Average {name}", linewidth=2)
        ax[1].fill_between(range(len(means)), means - stds, means + stds, alpha=0.2)
    
    ax[1].set_title(f"Performance Reward (Moving Average {window} episodes)", fontsize=14)
    ax[1].set_xlabel("Episodes")
    ax[1].set_ylabel("Reward")
    ax[1].axhline(y=5, color='r', linestyle='--', alpha=0.5, label="Success Threshold")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_results_apx(results_dict, title):
    plt.figure(figsize=(12, 6))
    for name, rewards in results_dict.items():
        window = 100
        smooth_rewards = pd.Series(rewards).rolling(window=window).mean()
        plt.plot(smooth_rewards, label=f"{name} (SMA {window})")
    
    plt.title(title, fontsize=14)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
def plot_epsilon_robustness(results_dict):
    """
    Riceve il dizionario results[n] che contiene i reward.
    Ricostruisce la curva epsilon basata sulla logica della tua run_experiment.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()  # Secondo asse per l'Epsilon

    # Colori per distinguere i metodi
    colors = {'DQN': '#1f77b4', 'SARSA': '#ff7f0e'}
    
    for name, stats in results_dict.items():
        # 'stats' in questo caso è direttamente la lista dei reward restituita da run_experiment
        # Se run_experiment restituisce (rewards, agent), assicurati di passare solo la lista
        rewards_list = stats if isinstance(stats, list) else stats[0]
        rewards = pd.Series(rewards_list)
        n_episodes = len(rewards)
        
        # --- RICOSTRUZIONE LOGICA EPSILON  ---
        eps = 1.0
        eps_min = 0.01
        eps_decay = (eps - eps_min) / (n_episodes * 0.9)
        
        # Generiamo la curva epsilon per ogni episodio
        epsilons = []
        current_eps = eps
        for _ in range(n_episodes):
            epsilons.append(current_eps)
            if current_eps > eps_min:
                current_eps -= eps_decay
        # --------------------------------------------------------------

        # Media Mobile e Deviazione Standard per i Reward
        window = 100 if n_episodes > 100 else 10
        means = rewards.rolling(window=window, min_periods=1).mean()
        stds = rewards.rolling(window=window, min_periods=1).std()
        
        # Plot Reward (Asse Sinistro)
        color = colors.get(name, None)
        line, = ax1.plot(means, label=f"{name} Reward", color=color, linewidth=2)
        ax1.fill_between(range(len(means)), means - stds, means + stds, color=color, alpha=0.1)
        
        # Plot Epsilon ricostruito (Asse Destro - Linea Tratteggiata)
        ax2.plot(epsilons, color=color, linestyle='--', alpha=0.4)

    # Formattazione Grafico
    ax1.set_xlabel("Episodios")
    ax1.set_ylabel("Recompensa (Media Móvil 100)")
    ax2.set_ylabel("Valor de Epsilon")
    ax2.set_ylim(0, 1.05)
    
    ax1.set_title("Estudio de Robustez: Reward vs Epsilon Decay")
    ax1.grid(True, alpha=0.3)
    
    # Legenda Unificata
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    plt.tight_layout()
    plt.show()