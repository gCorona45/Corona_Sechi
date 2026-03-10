import os
import random
import numpy as np
import torch
import gymnasium as gym

def set_global_seed(seed: int = 2024):
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
        
        # Variable de entorno para depuración de errores en CUDA 
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        print(f"Dispositivo GPU detectado y configurado para determinismo.")
    else:
        print("GPU no detectada. Usando CPU.")

def make_env_with_seed(env_name: str, seed: int):
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