# Examen: MPPT bajo sombreado parcial mediante métodos de aprendizaje por refuerzo y optimización

## Objetivo general

Se les proporciona el comportamiento de cuatro curvas de potencia-corriente (P-I) correspondientes a un sistema fotovoltaico operando bajo distintas condiciones de sombreado parcial aleatorio. Su tarea consiste en implementar un entorno de Gymnasium que alterne de forma aleatoria entre estos cuatro casos y en desarrollar, entrenar y comparar varios algoritmos de optimización / aprendizaje por refuerzo capaces de encontrar la corriente que maximiza la potencia extraída. El criterio de paro de los algoritmos no debe utilizar el ground-truth (el máximo real de la curva).

Dataset:

https://drive.google.com/drive/folders/1TBmXOfegocxFfZjw4XAXMW0Q6j-el6mg?usp=drive_link

## 1. Material proporcionado
Cuatro archivos NumPy (.npy ) que contienen los datos de las curvas P-I bajo cuatro patrones de sombreado distintos.
Cada archivo permite evaluar la potencia correspondiente a un valor de corriente (o contiene los puntos discretos de la curva).
Documentación mínima del formato de los archivos (se les entregará junto con los archivos).
Los cuatro casos representan condiciones de sombreado parcial que generan curvas P-I con múltiples máximos locales. El algoritmo debe ser capaz de operar sin conocer de antemano cuál de los cuatro casos está activo.

## 2. Entorno de Gymnasium (Gym Env)
Implementar un entorno personalizado que cumpla con la interfaz de Gymnasium y que:

- En cada episodio (o durante la ejecución continua) seleccione aleatoriamente uno de los cuatro casos de sombreado.
- Mantenga el caso activo durante un número de pasos (o un intervalo de tiempo simulado) configurable, tras el cual cambie de forma aleatoria a otro caso.
- Exponga como acción la corriente propuesta por el agente (o un incremento de corriente, según el diseño que elijan).
- Devuelva como observación  el valor de la Potencia.
- Calcule la recompensa en función de la potencia obtenida (o de una señal derivada de ella).
- No proporcione al agente el valor del máximo real de la curva (ground-truth).
El entorno debe ser reutilizable para todos los métodos que se evaluarán.

## 3. Métodos a implementar y evaluar
Deben implementar y entrenar los siguientes algoritmos, de forma independiente:

1. **REINFORCE** (policy gradient puro).
2. **Actor-Critic** (con red de política y red de crítico).
3. **PPO** (Proximal Policy Optimization).
4. **Cross-Entropy Method** (CEM) clásico.
5. **MLP-CEM**: optimización mediante Cross-Entropy Method de los pesos de una red neuronal MLP que produce la acción (corriente). Se debe muestrear conjuntos de pesos, evaluarlos y actualizar la distribución de pesos.

Para cada método deben:
- Usar PyTorch
- Definir una condición de paro que no utilice el ground-truth . Justifiquen la condición elegida.
- Registrar métricas de entrenamiento (recompensa acumulada, potencia media alcanzada, número de evaluaciones, etc.).

## 4. Entregables
### 1. Código por método
- Un script o notebook por algoritmo (o un repositorio bien organizado con un módulo por método).
- El código del entorno de Gymnasium.
- Código de evaluación final que permita correr cada método sobre los cuatro casos y reportar el desempeño alcanzado.
- Debe ser reproducible (semillas fijadas, dependencias listadas).

### 2. GIF de evolución del entrenamiento (uno por método, o un GIF comparativo)
El GIF debe mostrar de forma clara y a una velocidad (FPS) que permita evaluar el desempeño:
- La curva de potencia del caso activo en ese momento.
- Los puntos/muestreos que el algoritmo está evaluando sobre la curva.
- La salida del modelo (corriente propuesta) en cada instante.
- El nombre del método que se está ejecutando.
- Al costado (o en un panel lateral):
  - Valores relevantes del gradiente (norma, componentes principales o media/varianza, según aplique).
  - Salida de la red de política.
  - Salida de la red del crítico (cuando el método lo tenga: Actor-Critic y PPO).
  - Indicación visual del cambio de caso de sombreado cuando este ocurra.
  
El GIF debe permitir observar tanto la exploración sobre la curva como la convergencia del algoritmo y el comportamiento de los gradientes/políticas/críticos.

## 5. Criterios de evaluación (orientativos)
- Correcta implementación y funcionamiento del entorno de Gymnasium con alternancia aleatoria de casos.
- Implementación funcional de los cinco métodos.
- Condición de paro independiente del ground-truth, correctamente justificada y aplicada.
- Calidad y claridad de los GIF (se debe poder apreciar la evolución del entrenamiento y el desempeño relativo).
- Análisis comparativo final (aunque sea breve) indicando qué método se comportó mejor y por qué, bajo las condiciones de sombreado cambiante.
- Calidad del código (legibilidad, modularidad y reproducibilidad).

## 6. Restricciones importantes
- El agente **no puede** consultar el máximo real de la curva en ningún momento del entrenamiento ni para detener el algoritmo.
- Los archivos NumPy deben tratarse como cajas negras: solo se permite evaluar la potencia para un valor de corriente dado (o interpolar la curva si los datos lo permiten). Usar interpolación es permitido
- Se permite el uso de librerías estándar (PyTorch, Gymnasium, NumPy, Matplotlib, imageio/pillow para GIFs, etc.), siempre que el algoritmo principal sea implementado o claramente adaptado por ustedes.

**Formato de entrega:** Repositorio (GitHub/GitLab) + enlace a los GIF (o los GIF incluidos en el repositorio) + breve reporte (máximo 2 páginas) con el análisis comparativo.

Cualquier duda sobre el formato de los archivos NumPy o sobre el entorno debe resolverse en las primeras 48 horas.