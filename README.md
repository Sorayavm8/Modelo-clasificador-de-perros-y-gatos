# Cat and Dog Detection - Image Classification con Data Augmentation

Este proyecto implementa un **clasificador de imágenes** capaz de detectar si una foto contiene un **gato** o un **perro**, usando técnicas de **Deep Learning** y **Data Augmentation**.

## Contenido del proyecto

- `clasificador_perros_y_gatos.ipynb`: Notebook principal con todo el flujo del proyecto:
  1. Preparación y visualización de datos.
  2. Data Augmentation con `albumentations`.
  3. Generador de datos personalizado (`PyDataset`).
  4. Construcción de modelos CNN.
  5. Entrenamiento, validación y evaluación.
  6. Iteraciones con mejoras: EarlyStopping, Dropout y Transfer Learning (VGG16).
  7. Análisis de resultados y conclusiones.

- Dataset: `cats_and_dogs.zip` (no incluido en el repositorio).

## Tecnologías y librerías utilizadas

- Python 3.x
- Keras / TensorFlow
- Albumentations
- OpenCV
- scikit-learn
- Matplotlib / Pandas / NumPy

## Descripción del flujo

1. **Data Augmentation:** Se aplican transformaciones aleatorias (flip, blur, rotaciones, cambios de brillo) para aumentar la variabilidad de los datos y mejorar la generalización del modelo.
2. **Generador de datos:** Clase `CatsDogsDataset` para alimentar la red neuronal con batches de imágenes aumentadas.
3. **Modelos de clasificación:**
   - CNN básica con 2-3 bloques convolucionales y capas densas.
   - Mejoras mediante **Dropout** y **Callbacks** (EarlyStopping y ModelCheckpoint).
   - Transfer Learning usando **VGG16** para la última iteración.
4. **Evaluación:** Se monitorea `loss` y `accuracy` en entrenamiento y validación, y se visualiza la **matriz de confusión** para analizar falsos positivos y negativos.

## Resultados principales

- La última iteración con **VGG16 y Data Augmentation** alcanza más del **90% de precisión** tanto en gatos como en perros.
- El uso de Dropout y Transfer Learning reduce significativamente el **overfitting**.
- Ajuste del threshold a 0.4 para mejorar la detección de perros y reducir falsos negativos.

## Cómo ejecutar

1. Instalar dependencias (ejemplo con pip):
   ```bash
   pip install tensorflow keras albumentations opencv-python scikit-learn matplotlib pandas numpy
