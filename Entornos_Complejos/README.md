# Aprendizaje en Entornos Complejos — Partes 2 y 3

## Información

- **Alumnos:** Corona, Giovanni; Sechi, Michele
- **Asignatura:** Aprendizaje por Refuerzo
- **Curso:** 2025/2026
- **Grupo:** CoronaSechi

## Descripción

Este trabajo explora el aprendizaje por refuerzo en entornos MDP (Procesos de Decisión de Markov) completos, abordando dos niveles de complejidad creciente.

En la primera parte se estudian métodos tabulares clásicos (Q-Learning, SARSA, Monte Carlo On-Policy y Off-Policy) en una cuadrícula discreta 8×8, analizando la convergencia hacia la política óptima, la cobertura del espacio de estados y la calidad de las trayectorias aprendidas.

En la segunda parte se escala hacia un entorno de control continuo (Flappy Bird), donde los métodos tabulares resultan inviables. Se implementan y comparan DQN (Deep Q-Network) y SARSA Semi-Gradient, con un estudio sistemático de los hiperparámetros más relevantes: learning rate, target update frequency y tamaño del replay buffer (experience replay).

## Estructura

```
Entornos_Complejos/
│
├── main.ipynb                        # Introducción, análisis y conclusiones generales
├── metodos_tabulares.ipynb           # Experimentos con métodos tabulares en SimpleGrid
├── metodos_aproximados.ipynb         # Experimentos con DQN y SARSA Semi-Gradient en Flappy Bird
│
├── src/
│   ├── agent.py                      # Clase base Agent
│   ├── DQN_Network.py                # Arquitectura de la red neuronal para DQN
│   ├── DQNAgent.py                   # Agente DQN con experience replay y target network
│   ├── SARSASemiGradientAgent.py     # Agente SARSA con aproximación funcional
│   ├── tabular_agents.py             # Q-Learning, SARSA, Monte Carlo On/Off-Policy
│   └── utils.py                      # Utilidades: seed global, limpieza de memoria, make_env
│
├── videos/                           # Grabaciones de episodios de los agentes entrenados
├── requirements.txt                  # Dependencias del proyecto
└── README.md
```

## Instalación y Uso

Los notebooks están diseñados para ejecutarse en orden:

1. `main.ipynb` — Presenta los trabajos realizados y enlace a los estudios.
1. `metodos_tabulares.ipynb` — entrena y evalúa los agentes tabulares
2. `metodos_aproximados.ipynb` — entrena y evalúa DQN y SARSA Semi-Gradient, incluyendo grid search

## Tecnologías Utilizadas

- **Python** 3.12
- **PyTorch** — implementación de las redes neuronales (DQN, SARSA Semi-Gradient)
- **Gymnasium** — framework de entornos de RL
- **flappy-bird-gymnasium** — entorno Flappy Bird para métodos aproximados
- **gym-simplegrid** — entorno de cuadrícula 8×8 para métodos tabulares
- **NumPy / Pandas** — procesamiento de datos y análisis de resultados
- **Matplotlib** — visualización de curvas de aprendizaje y trayectorias
- **tqdm** — barras de progresso durante el entrenamiento