# Parte 1: Problema del Bandido de k-brazos

## Información
**Alumnos:** Corona, Giovanni; Sechi, Michele
**Asignatura:** Aprendizaje por Refuerzo
**Curso:** 2025/2026
**Grupo:** Corona_Sechi

## Descripción
Este módulo del proyecto aborda el problema del bandido multibrazo (k-armed bandit), un marco teórico fundamental en el Aprendizaje por Refuerzo para el estudio del dilema entre la exploración y la explotación. El objetivo es maximizar la recompensa acumulada (o minimizar el regret asintótico) seleccionando secuencialmente entre k opciones con distribuciones de probabilidad subyacentes desconocidas.

El presente estudio empírico evalúa el rendimiento de múltiples familias algorítmicas frente a topologías de recompensa de distinta complejidad estocástica. Se han implementado y analizado las siguientes técnicas:
* **Métodos Heurísticos:** Epsilon-Greedy y Epsilon-Decay.
* **Límites de Confianza (UCB):** UCB1, UCB1-Tuned y UCB2.
* **Métodos Probabilísticos/Gradiente:** Softmax y Gradient Bandit.

Los agentes son sometidos a simulaciones de Monte Carlo (500 ejecuciones independientes) sobre entornos con distribuciones Bernoulli, Binomiales y Normales con varianza heterogénea.

## Estructura del Directorio
La arquitectura de esta sección está organizada de la siguiente manera para garantizar la modularidad y reproducibilidad de los experimentos:

* `/src/`: Contiene el código fuente modular orientado a objetos. Incluye las clases base de los agentes, la implementación algorítmica y los generadores de entornos (brazos).
* `/tests/`: Directorio destinado a las pruebas unitarias y validación lógica de los componentes (generación de brazos, actualizaciones Q).
* `main.ipynb`: Cuaderno principal de navegación e introducción teórica a esta parte del proyecto.
* `notebook_bernoulli.ipynb`: Análisis del rendimiento en entornos de recompensa binaria.
* `notebook_binomial.ipynb`: Experimentación en maximización de conversiones sobre lotes discretos.
* `notebook_normal.ipynb`: Estudio de la robustez algorítmica frente a métricas continuas de alta varianza.
* `test_scalability.ipynb`: Prueba de estrés sobre la degradación del rendimiento al escalar la dimensionalidad k.
* `test_ucb_family.ipynb`: Análisis intrafamiliar de optimizaciones estructurales (épocas exponenciales y varianza empírica).
## Instalación y Uso
El código está diseñado para ser ejecutado de forma nativa e ininterrumpida en la plataforma Google Colab, asegurando la reproducibilidad exacta de los resultados mediante la fijación de semillas (seeds) estáticas.

**Ejecución en la nube:**
1. Abra el archivo `main.ipynb` a través de GitHub y haga clic en el botón `Open in Colab`.
2. La primera celda del cuaderno clonará automáticamente el repositorio, instalará las dependencias necesarias y configurará el entorno de trabajo.
3. Puede navegar a los cuadernos de los experimentos específicos mediante los enlaces proporcionados.
4. En cada experimento, seleccione `Entorno de ejecución > Ejecutar todas` para replicar el análisis completo sin intervención manual.

**Ejecución Local:**
Requiere la clonación manual del repositorio y la instalación de las dependencias indicadas en el entorno virtual activo.

## Tecnologías Utilizadas
* **Lenguaje:** Python 3.x
* **Cálculo Numérico y Simulación:** NumPy
* **Visualización de Datos:** Matplotlib, Seaborn
* **Entorno de Desarrollo:** Jupyter Notebook / Google Colab
* **Métricas de Progreso:** tqdm