"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List, Optional
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from algorithms import Algorithm

def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.
    Intenta detectar automáticamente parámetros comunes como epsilon, c (UCB) o temperature (Softmax).
    
    :param algo: Instancia de un algoritmo.
    :return: Cadena descriptiva para el algoritmo.
    """
    label = algo.__class__.__name__
    
    # Lista de atributos comunes para buscar y añadir a la etiqueta
    params = []
    if hasattr(algo, 'epsilon'):
        params.append(f"eps={algo.epsilon}")
    if hasattr(algo, 'c'): # Para UCB
        params.append(f"c={algo.c}")
    if hasattr(algo, 'temperature'): # Para Softmax
        params.append(f"temp={algo.temperature}")
    if hasattr(algo, 'alpha'): # Para UCB2
        params.append(f"alpha={algo.alpha}")
    
    if params:
        label += f" ({', '.join(params)})"
        
    return label

def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.
    
    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(12, 6))
    
    steps_range = range(steps)
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(steps_range, rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos', loc='best')
    plt.tight_layout()
    plt.show()

def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.
    
    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz (n_algos x steps) de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(12, 6))

    steps_range = range(steps)
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        # Convertimos a porcentaje (0-100%) para la visualización
        plt.plot(steps_range, optimal_selections[idx] * 100, label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('% Selección Óptima', fontsize=14)
    plt.title('Porcentaje de Selección del Brazo Óptimo', fontsize=16)
    plt.ylim(-5, 105) # Fija el límite Y entre 0 y 100 aproximadamente
    plt.legend(title='Algoritmos', loc='best')
    plt.tight_layout()
    plt.show()

def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo.
    
    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz (n_algos x steps) de regret acumulado.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(12, 6))

    steps_range = range(steps)
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(steps_range, regret_accumulated[idx], label=label, linewidth=2)


    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Regret Acumulado', fontsize=14)
    plt.title('Regret Acumulado vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos', loc='best')
    plt.tight_layout()
    plt.show()

def plot_arm_statistics(final_q_values: np.ndarray, action_counts: np.ndarray, algorithms: List, optimal_arm_index: int, true_means: List[float]):
    """
    Muestra las estadísticas finales de cada brazo para cada algoritmo.
    Genera un subplot por algoritmo con histogramas agrupados (Valor Estimado vs Valor Real).
    """
    n_algos = len(algorithms)
    n_arms = final_q_values.shape[1]
    
    cols = 2 if n_algos > 1 else 1
    rows = math.ceil(n_algos / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(9 * cols, 5 * rows), sharey=True)
    if n_algos == 1: axes = [axes]
    axes = np.array(axes).flatten()

    sns.set_theme(style="whitegrid", palette="muted")
    
    x = np.arange(n_arms)
    width = 0.35  # Anchura de las barras

    for idx, algo in enumerate(algorithms):
        ax = axes[idx]
        q_vals = final_q_values[idx]
        counts = action_counts[idx]
        
        # Colores
        est_colors = ['#2ca02c' if i == optimal_arm_index else '#1f77b4' for i in range(n_arms)]
        real_color = '#b0b0b0' 
        
        # Barras Agrupadas
        rects1 = ax.bar(x - width/2, q_vals, width, color=est_colors, edgecolor='black', alpha=0.9, label='Valor Estimado ($\hat{Q}$)')
        rects2 = ax.bar(x + width/2, true_means, width, color=real_color, edgecolor='black', alpha=0.5, hatch='//', label='Valor Real ($\mu$)')
        
        xtick_labels = [f"B{i}\n(N={int(c)})" + ("\n★" if i == optimal_arm_index else "") for i, c in enumerate(counts)]
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, fontsize=10)
        
        # Titolo ripristinato utilizzando esattamente la funzione originale
        ax.set_title(get_algorithm_label(algo), fontsize=14, fontweight='bold')
        
        ax.set_ylabel("Recompensa" if idx % cols == 0 else "")
        
        # Valores solo sobre la estimación
        for bar in rects1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        if idx == 0:
            ax.legend(loc='upper right', fontsize=10)

    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle("Precisión de Aprendizaje: Valores Estimados vs Reales", fontsize=16, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_smoothed_curves(steps: int, 
                         data: np.ndarray, 
                         algorithms: List[Algorithm], 
                         title: str, 
                         ylabel: str, 
                         window_size: int = 50, 
                         ):
    """
    Genera una gráfica suavizada (Media Móvil) para reducir el ruido visual.
    Es especialmente útil para entornos con recompensas binarias (como Bernoulli)
    donde las gráficas estándar pueden verse muy ruidosas.
    
    :param steps: Número total de pasos de tiempo.
    :param data: Matriz de datos (n_algos x steps).
    :param algorithms: Lista de las instancias de algoritmos.
    :param title: Título del gráfico.
    :param ylabel: Etiqueta del eje Y.
    :param window_size: Tamaño de la ventana de suavizado (por defecto 50).
    """
    # Configuración del estilo
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(12, 6))
    
    # Generar una paleta de colores única para cada algoritmo
    colors = sns.color_palette("husl", len(algorithms))

    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        y_data = data[idx]
        
        # Aplicar Media Móvil (Convolution) para suavizar la curva
        if window_size > 1 and len(y_data) >= window_size:
            # El modo 'valid' devuelve solo la parte donde la ventana y los datos se superponen completamente
            y_smoothed = np.convolve(y_data, np.ones(window_size)/window_size, mode='valid')
            # Ajustamos el eje X para centrar la curva suavizada
            x_axis = range(window_size // 2, len(y_smoothed) + window_size // 2)
        else:
            # Si la ventana es 1 o los datos son muy cortos, no suavizamos
            y_smoothed = y_data
            x_axis = range(steps)
            
        # Dibujar la curva suavizada con línea más gruesa
        plt.plot(x_axis, y_smoothed, label=label, color=colors[idx], linewidth=2.5)
        
        # plt.plot(range(steps), y_data, color=colors[idx], alpha=0.1, linewidth=1)


    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f"{title} (Suavizado: w={window_size})", fontsize=16)
    plt.legend(title='Algoritmos', loc='lower right', frameon=True)
    plt.tight_layout()
    plt.show()