
import numpy as np
from arm import Arm

class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución Binomial.

        :param n: Número de ensayos (n > 0).
        :param p: Probabilidad de éxito en cada ensayo (0 <= p <= 1).
        """
        assert n > 0, "El número de ensayos n debe ser mayor que 0."
        assert 0.0 <= p <= 1.0, "La probabilidad p debe estar entre 0 y 1."

        self.n = int(n)
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución Binomial.
        Representa el número de éxitos en n ensayos.

        :return: Recompensa obtenida del brazo (0 a n).
        """
        return np.random.binomial(self.n, self.p)

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Binomial.
        E[X] = n * p

        :return: Valor esperado de la distribución.
        """
        return self.n * self.p

    def __str__(self):
        """
        Representación en cadena del brazo Binomial.

        :return: Descripción detallada del brazo.
        """
        return f"ArmBinomial(n={self.n}, p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, n: int = 10, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos con probabilidades p únicas en el rango [p_min, p_max],
        manteniendo n fijo para todos.

        :param k: Número de brazos a generar.
        :param n: Número de ensayos (fijo para todos los brazos).
        :param p_min: Probabilidad mínima.
        :param p_max: Probabilidad máxima.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert n > 0, "El número de ensayos n debe ser mayor que 0."
        assert 0.0 <= p_min < p_max <= 1.0, "Rango de probabilidades inválido."

        # Generar k valores únicos de p con decimales
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 4)
            p_values.add(p)

        p_values = list(p_values)
        
        # Se crean los brazos con n fijo y p variable
        arms = [ArmBinomial(n, p) for p in p_values]

        return arms