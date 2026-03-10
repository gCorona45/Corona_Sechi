"""
Module: arms/armnormal.py
Description: Contains the implementation of the ArmNormal class for the normal distribution arm.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""


from typing import Optional
import numpy as np

from arm import Arm


class ArmNormal(Arm):
    def __init__(self, mu: float, sigma: float):
        """
        Inicializa el brazo con distribución normal.

        :param mu: Media de la distribución.
        :param sigma: Desviación estándar de la distribución.
        """
        assert sigma > 0, "La desviación estándar sigma debe ser positiva."

        self.mu = mu
        self.sigma = sigma

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución normal.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.normal(self.mu, self.sigma)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución normal.

        :return: Valor esperado de la distribución.
        """

        return self.mu

    def __str__(self):
        """
        Representación en cadena del brazo normal.

        :return: Descripción detallada del brazo normal.
        """
        return f"ArmNormal(mu={self.mu}, sigma={self.sigma})"

    @classmethod
    def generate_arms(cls, k: int, mu_min: float = 1.0, mu_max: float = 10.0, sigma_min: Optional[float] = None, sigma_max: Optional[float] = None):
        """
        Genera k brazos con medias únicas en el rango [mu_min, mu_max].
        Si sigma_min y sigma_max no se especifican, la desviación estándar se fija en 1.0.

        :param k: Número de brazos a generar.
        :param mu_min: Valor mínimo de la media esperada.
        :param mu_max: Valor máximo de la media esperada.
        :param sigma_min: (Opcional) Valor mínimo de la desviación estándar.
        :param sigma_max: (Opcional) Valor máximo de la desviación estándar.
        :return: Lista de brazos generados (ArmNormal).
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert mu_min < mu_max, "El valor de mu_min debe ser menor que mu_max."

        # Generar k-valores únicos de mu con 4 decimales
        mu_values = set()
        while len(mu_values) < k:
            mu = round(np.random.uniform(mu_min, mu_max), 4)
            mu_values.add(mu)
        
        mu_values = list(mu_values)

        # Determinar el arreglo de desviaciones estándar
        if sigma_min is None or sigma_max is None:
            # Modo homogéneo: desviación estándar fija en 1.0 para todos los brazos
            sigma_values = [1.0] * k
        else:
            # Modo heterogéneo: desviación estándar aleatoria en el rango especificado
            assert sigma_min <= sigma_max, "sigma_min no puede ser mayor que sigma_max."
            assert sigma_min >= 0, "La desviación estándar no puede ser negativa."
            sigma_values = [round(np.random.uniform(sigma_min, sigma_max), 4) for _ in range(k)]

        # Construir la lista de brazos acoplando cada mu con su respectivo sigma
        arms = [ArmNormal(mu, sigma) for mu, sigma in zip(mu_values, sigma_values)]

        return arms