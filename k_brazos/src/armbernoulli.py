import numpy as np
from arm import Arm

class ArmBernoulli(Arm):
    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución de Bernoulli.

        :param p: Probabilidad de éxito (0 <= p <= 1).
        """
        assert 0.0 <= p <= 1.0, "La probabilidad p debe estar entre 0 y 1."

        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución de Bernoulli.
        Devuelve 1 (éxito) con probabilidad p, y 0 (fracaso) con probabilidad 1-p.

        :return: Recompensa obtenida del brazo (0 o 1).
        """
        # Binomial con n=1 es equivalente a Bernoulli
        return np.random.binomial(1, self.p)

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución de Bernoulli.

        :return: Valor esperado (p).
        """
        return self.p

    def __str__(self):
        """
        Representación en cadena del brazo Bernoulli.

        :return: Descripción detallada del brazo.
        """
        return f"ArmBernoulli(p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos con probabilidades únicas en el rango [p_min, p_max].

        :param k: Número de brazos a generar.
        :param p_min: Probabilidad mínima.
        :param p_max: Probabilidad máxima.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert 0.0 <= p_min < p_max <= 1.0, "Rango de probabilidades inválido."

        # Generar k valores únicos de p con decimales
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 4)
            p_values.add(p)

        p_values = list(p_values)
        
        arms = [ArmBernoulli(p) for p in p_values]

        return arms