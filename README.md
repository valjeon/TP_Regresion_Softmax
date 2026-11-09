# TP - Clasificación multiclase de texto con Regresión Softmax

Trabajo práctico de **Redes Neuronalesg**. Se clasifican posts del dataset **20 Newsgroups** (~18.000 mensajes de foros de Usenet, 20 categorías) con una regresión softmax implementada en PyTorch, y luego se compara contra un MLP.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `TP_Softmax_Regression.ipynb` | Pipeline completo con regresión softmax (una sola capa lineal): preprocesamiento, TF-IDF, entrenamiento, búsqueda de hiperparámetros y análisis. |
| `TNG+MLP.ipynb` | Misma base, pero el modelo pasa a ser un MLP con capas ocultas. Incluye la búsqueda de capas y unidades ocultas y la comparación con el modelo lineal. |
| `TNG_Regularizacion.ipynb` | Efecto de la regularización L1/L2/L1+L2, Dropout y BatchNorm (cada técnica por separado) sobre el MLP, con búsqueda conjunta en Optuna y evaluación final en test. |
| `mlflow.db` | Base de MLflow con los parámetros y las métricas por época de todas las corridas de `TNG_Regularizacion.ipynb`. |
| `requirements.txt` | Dependencias de Python. |

## Qué hace el trabajo

1. **Datos:** se carga 20 Newsgroups con `fetch_20newsgroups`, quitando `headers`, `footers` y `quotes` para que el problema no sea trivial.
2. **Preprocesamiento configurable:** minúsculas, solo tokens alfabéticos, y opcionalmente remoción de stopwords y normalización (`none`, `stem` o `lemmatize`) con NLTK.
3. **Vectorización:** TF-IDF con `ngram_range`, `max_features` y `min_df` configurables, más normalización de los vectores.
4. **Modelo:** regresión softmax (una capa lineal + `CrossEntropyLoss`) en PyTorch. En `TNG+MLP.ipynb`, un MLP con capas ocultas.
5. **Seguimiento de experimentos:** curvas de loss y accuracy en TensorBoard, y parámetros y modelos en MLflow.
6. **Optimización:** búsqueda de hiperparámetros de preprocesamiento y de entrenamiento (como `lr` y `batch_size`) con Optuna.
7. **Entrenamiento final y evaluación:** con los mejores hiperparámetros, sobre el test set original. Incluye matriz de confusión, `classification_report` y los pares de categorías que más se confunden.
8. **Análisis:** efecto del learning rate, de `remove_stopwords` y `normalization`, y, en el notebook del MLP, de la cantidad de capas y unidades ocultas frente a tiempos de corrida.

## Notebook de regularización (`TNG_Regularizacion.ipynb`)

Sobre el mismo MLP de `TNG+MLP.ipynb` (capas ocultas `[32, 128, 128, 64]`, ReLU, Adam) se incorporan como hiperparámetros:

- **Regularización L1, L2 y L1+L2**, como penalidad explícita sobre los pesos de las capas lineales.
- **Dropout**, con distintas probabilidades.
- **Batch Normalization**, combinada con distintos *learning rates*.

Para cada técnica se compara contra una línea base sin regularización y se registra: accuracy de validación, brecha train-validación, valor absoluto de los pesos (media, máximo y fracción cercana a cero), tiempo por época y ruido de la curva de aprendizaje. Después, Optuna busca la mejor combinación y el modelo final se evalúa una sola vez sobre el test set original.

El notebook se entrega con las salidas guardadas, así que se puede leer sin ejecutarlo.

## Resultados destacados

- Regresión softmax: accuracy de **0.6581 en test** y **0.9723 en train**, lo que indica sobreajuste.
- El MLP no mejoró de forma clara al modelo lineal. Los promedios del `classification_report` del modelo lineal fueron levemente más altos, y en algunas categorías el MLP empeoró. La conclusión del notebook es que, para este dataset, el modelo más simple funciona mejor.

## Cómo ejecutarlo

### Opción 1: Google Colab

Cada notebook tiene un botón **Open in Colab** en la primera celda. Las dependencias que faltan se instalan desde el propio notebook.

### Opción 2: entorno local

```bash
git clone https://github.com/valjeon/TP_Regresion_Softmax.git
cd TP_Regresion_Softmax

python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

Los notebooks descargan recursos de NLTK (`stopwords`, `wordnet` y `punkt`) en la celda de setup. Hace falta conexión a internet la primera vez, tanto para eso como para descargar el dataset.

### Ver los experimentos

```bash
tensorboard --logdir runs
mlflow ui
```

Para explorar las corridas de `TNG_Regularizacion.ipynb` que quedaron guardadas en el repositorio, ejecutar desde la carpeta del repo:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

y abrir `http://localhost:5000`. Se verá el experimento `regularizacion_20newsgroups`, con una corrida por configuración (parámetros y métricas por época). Las curvas de TensorBoard no se incluyen en el repositorio: los mismos gráficos están en el notebook, y si se vuelve a ejecutar se regeneran en `runs/reg`.

## Notas

- El entrenamiento usa GPU si está disponible y, si no, CPU.
- Se fija una semilla para que los resultados sean reproducibles.
- La búsqueda con Optuna puede tardar bastante, sobre todo en el notebook del MLP.
