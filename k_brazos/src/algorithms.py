from abc import ABC, abstractmethod
import numpy as np
import math
class Algorithm(ABC):
    def __init__(self, k: int):
        """
        Inicializa el algoritmo con k brazos.
        :param k: Número de brazos.
        """
        self.k: int = k
        # Número de veces que se ha seleccionado cada brazo
        self.counts: np.ndarray = np.zeros(k, dtype=int)
        # Recompensa promedio estimada de cada brazo
        self.values: np.ndarray = np.zeros(k, dtype=float)

    @abstractmethod
    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política del algoritmo.
        :return: Índice del brazo seleccionado.
        """
        raise NotImplementedError("Este método debe ser implementado por la subclase.")

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las recompensas promedio estimadas de cada brazo.
        :param chosen_arm: Índice del brazo que fue tirado.
        :param reward: Recompensa obtenida.
        """
        self.counts[chosen_arm] += 1  # Incrementa el conteo del brazo seleccionado
        n = self.counts[chosen_arm]   # Número de veces que el brazo ha sido seleccionado
        value = self.values[chosen_arm] # Valor actual del brazo seleccionado

        # Actualización incremental de la recompensa promedio
        self.values[chosen_arm] = value + (reward - value) / n

    def reset(self):
        """
        Reinicia el estado del algoritmo.
        """
        self.counts = np.zeros(self.k, dtype=int)
        self.values = np.zeros(self.k, dtype=float)


class EpsilonGreedy(Algorithm):
    """Estrategia Epsilon-Greedy."""
    def __init__(self, k: int, epsilon: float):
        super().__init__(k)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        # Exploración
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        else:
            # Explotación: elegir el brazo con mayor valor estimado
            return np.random.choice(np.flatnonzero(self.values == self.values.max()))

class UCB1(Algorithm):
    """Algoritmo Upper Confidence Bound (UCB1)"""
    def select_arm(self) -> int:
        # Asegurar que cada brazo se accione al menos una vez al principio
        for arm in range(self.k):
            if self.counts[arm] == 0:
                return arm

        total_counts = np.sum(self.counts)
        # Cálculo del valor UCB: Q(a) + bonificación de incertidumbre
        bonus = np.sqrt((2 * np.log(total_counts)) / self.counts)
        ucb_values = self.values + bonus
        
        return np.argmax(ucb_values)
class UCB1Tuned(Algorithm):
    """
    Algoritmo UCB1-Tuned (Auer et al., 2002).
    Ajusta el término de exploración utilizando la varianza empírica de cada brazo.
    """
    def __init__(self, k: int):
        super().__init__(k)
        # Suma de los cuadrados de las recompensas para calcular la varianza
        self.sum_sq_rewards = np.zeros(k, dtype=float)

    def update(self, chosen_arm: int, reward: float):
        super().update(chosen_arm, reward)
        self.sum_sq_rewards[chosen_arm] += reward**2

    def select_arm(self) -> int:
        # 1. Fase de inicialización: probar cada brazo una vez
        for arm in range(self.k):
            if self.counts[arm] == 0:
                return arm

        total_counts = np.sum(self.counts)
        ucb_tuned_values = np.zeros(self.k)
        
        for arm in range(self.k):
            n_a = self.counts[arm]
            mean_a = self.values[arm]
            
            # Estimación de la varianza
            variance = (self.sum_sq_rewards[arm] / n_a) - (mean_a**2)
            
            # Cálculo del término V (mínimo entre la varianza más un margen, y 0.25)
            # 0.25 es la varianza máxima posible para variables en [0,1]
            V = variance + math.sqrt((2 * math.log(total_counts)) / n_a)
            V = min(0.25, V)
            
            # Cálculo del índice UCB1-Tuned
            bonus = math.sqrt((math.log(total_counts) / n_a) * V)
            ucb_tuned_values[arm] = mean_a + bonus
            
        return int(np.argmax(ucb_tuned_values))
        
    def reset(self):
        super().reset()
        self.sum_sq_rewards = np.zeros(self.k, dtype=float)


class UCB2(Algorithm):
    """
    Algoritmo UCB2 (Auer et al., 2002).
    Introduce el concepto de "épocas" para reducir la frecuencia de actualización
    y mejorar las cotas teóricas de regret.
    """
    def __init__(self, k: int, alpha: float = 0.5):
        super().__init__(k)
        assert alpha > 0, "El parámetro alpha debe ser > 0"
        self.alpha = alpha
        self.r = np.zeros(k, dtype=int) # Contador de épocas por brazo
        
        # Variables para gestionar la época actual
        self.current_arm = -1
        self.plays_left_in_epoch = 0

    def _tau(self, r: int) -> int:
        """Calcula la longitud de la época r."""
        return int(math.ceil((1 + self.alpha)**r))

    def select_arm(self) -> int:
        # Si estamos a mitad de una época, seguimos jugando el mismo brazo
        if self.plays_left_in_epoch > 0:
            self.plays_left_in_epoch -= 1
            return self.current_arm

        # Fase de inicialización
        for arm in range(self.k):
            if self.counts[arm] == 0:
                self.current_arm = arm
                self.r[arm] += 1
                # Las épocas iniciales suelen durar 1 jugada
                self.plays_left_in_epoch = self._tau(self.r[arm]) - self._tau(self.r[arm] - 1) - 1
                return arm

        # Si no estamos en una época y ya inicializamos, elegimos un nuevo brazo
        total_counts = np.sum(self.counts)
        ucb2_values = np.zeros(self.k)
        
        for arm in range(self.k):
            tau_r = self._tau(self.r[arm])
            mean_a = self.values[arm]
            
            # Cálculo del índice UCB2
            bonus = math.sqrt(((1 + self.alpha) * math.log(math.e * total_counts / tau_r)) / (2 * tau_r))
            ucb2_values[arm] = mean_a + bonus
            
        best_arm = int(np.argmax(ucb2_values))
        
        # Iniciamos una nueva época para el brazo ganador
        self.current_arm = best_arm
        self.r[best_arm] += 1
        
        # Calculamos cuántas veces seguidas debemos jugarlo
        plays_in_new_epoch = self._tau(self.r[best_arm]) - self._tau(self.r[best_arm] - 1)
        self.plays_left_in_epoch = max(0, plays_in_new_epoch - 1)
        
        return best_arm

    def reset(self):
        super().reset()
        self.r = np.zeros(self.k, dtype=int)
        self.current_arm = -1
        self.plays_left_in_epoch = 0
class UCBV(Algorithm):
    """
    Algoritmo UCB-V (Upper Confidence Bound with Variance).
    Utiliza la desigualdad de Bernstein para ajustar la exploración en función de la varianza empírica.
    """
    def __init__(self, k: int, b: float = 15.0, c: float = 1.0):
        """
        Inicializa el algoritmo UCB-V delegando atributos base a la superclase.
        
        :param k: Número de brazos.
        :param b: Cota superior (o amplitud máxima esperada) de las recompensas.
        :param c: Constante empírica para el término de exploración lineal.
        """
        super().__init__(k)
        self.b = b
        self.c = c
        self.M2 = np.zeros(k, dtype=float)  # Para el cálculo incremental de la varianza (Welford)
        self.total_steps = 0

    def select_arm(self) -> int:
        """Selecciona una acción basándose en la regla de decisión de UCB-V."""
        self.total_steps += 1
        
        # Fase de inicialización forzada: evaluar cada brazo al menos una vez
        if self.total_steps <= self.k:
            return self.total_steps - 1
            
        # Cálculo de la varianza empírica
        # Se utiliza M2 / N_a para obtener la varianza poblacional de las muestras actuales
        variances = self.M2 / self.counts
        
        ln_t = np.log(self.total_steps)
        
        # Aplicación de la desigualdad de Bernstein
        term_variance = np.sqrt(2 * variances * ln_t / self.counts)
        term_linear = self.c * self.b * ln_t / self.counts
        
        ucb_values = self.values + term_variance + term_linear
        
        return int(np.argmax(ucb_values))

    def update(self, chosen_arm: int, reward: float):
        """Actualiza las estimaciones de media y varianza de forma incremental."""
        # Almacenamos el valor estimado anterior para el cálculo de Welford
        old_value = self.values[chosen_arm]
        
        # Invocamos la actualización de conteos y promedios de la clase base
        super().update(chosen_arm, reward)
        
        # Algoritmo online de Welford para actualizar la varianza (M2) de forma estable
        delta = reward - old_value
        delta2 = reward - self.values[chosen_arm]
        self.M2[chosen_arm] += delta * delta2

    def reset(self):
        """Reinicia el estado interno extendiendo el reinicio de la superclase."""
        super().reset()
        self.M2 = np.zeros(self.k, dtype=float)
        self.total_steps = 0

    def __str__(self):
        return f"UCB-V (b={self.b}, c={self.c})"
class Softmax(Algorithm):
    """Estrategia Softmax (Exploración de Boltzmann)"""
    def __init__(self, k: int, temperature: float):
        super().__init__(k)
        self.temperature = temperature 

    def select_arm(self) -> int:
        # Restamos el máximo para evitar desbordamiento numérico (overflow)
        exp_values = np.exp((self.values - np.max(self.values)) / self.temperature)
        probs = exp_values / np.sum(exp_values)
        
        return np.random.choice(range(self.k), p=probs)

class EpsilonDecay(Algorithm):
    """Estrategia Epsilon-Greedy con decaimiento."""
    def __init__(self, k: int, epsilon_start=1.0, epsilon_end=0.01, decay_rate=0.999):
        super().__init__(k)
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate

    def select_arm(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.random.choice(np.flatnonzero(self.values == self.values.max()))

    def update(self, chosen_arm: int, reward: float):
        # Llama a la actualización de la clase base
        super().update(chosen_arm, reward)
        # Reduce epsilon multiplicativamente
        self.epsilon = max(self.epsilon_end, self.epsilon * self.decay_rate)
        
    def reset(self):
        super().reset()
        # Restaurar epsilon a su valor inicial
        self.epsilon = self.epsilon_start

class GradientBandit(Algorithm):
    """Algoritmo de Ascenso de Gradiente."""
    def __init__(self, k: int, alpha: float):
        super().__init__(k)
        self.alpha = alpha
        self.preferences = np.zeros(k)
        self.average_reward = 0.0
        self.time_step = 0
        self.probs = np.ones(k) / k

    def select_arm(self) -> int:
        # Probabilidades Softmax sobre las preferencias (H)
        exp_prefs = np.exp(self.preferences - np.max(self.preferences))
        self.probs = exp_prefs / np.sum(exp_prefs)
        return np.random.choice(range(self.k), p=self.probs)

    def update(self, chosen_arm: int, reward: float):
        # Mantenemos counts actualizados llamando al super
        super().update(chosen_arm, reward)
        
        self.time_step += 1
        # R_bar (Línea base)
        self.average_reward += (1 / self.time_step) * (reward - self.average_reward)
        
        # Actualización de preferencias H(a)
        one_hot = np.zeros(self.k)
        one_hot[chosen_arm] = 1
        baseline_term = reward - self.average_reward
        self.preferences += self.alpha * baseline_term * (one_hot - self.probs)

    def reset(self):
        super().reset()
        self.preferences = np.zeros(self.k)
        self.average_reward = 0.0
        self.time_step = 0
        self.probs = np.ones(self.k) / self.k