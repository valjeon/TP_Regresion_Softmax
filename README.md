# TP - Clasificación multiclase de texto con Regresión Softmax

Trabajo práctico de **Redes Neuronales**. Se clasifican posts del dataset **20 Newsgroups** (~18.000 mensajes de foros de Usenet, 20 categorías) con una regresión softmax implementada en PyTorch, y luego se compara contra un MLP.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `TP_softmax/TP_Softmax_Regression.ipynb` | Pipeline completo con regresión softmax (una sola capa lineal): preprocesamiento, TF-IDF, entrenamiento, búsqueda de hiperparámetros y análisis. |
| `TP_mlp/TNG+MLP.ipynb` | Misma base, pero el modelo pasa a ser un MLP con capas ocultas. Incluye la búsqueda de capas y unidades ocultas y la comparación con el modelo lineal. |
| `TP_regularizacion/TNG_Regularizacion.ipynb` | Efecto de la regularización L1/L2/L1+L2, Dropout y BatchNorm (cada técnica por separado) sobre el MLP, con búsqueda conjunta en Optuna y evaluación final en test. |
| `TP_softmax/mlflow.db`, `TP_softmax/mlruns/` | Corridas de MLflow del notebook de regresión softmax (trials de Optuna, análisis y modelo final) y el modelo guardado. |
| `TP_mlp/mlflow.db`, `TP_mlp/mlruns/` | Corridas de MLflow del notebook del MLP (trials, experimentos de arquitectura, búsqueda de sobreajuste y modelos finales) y los modelos guardados. |
| `TP_softmax/runs/`, `TP_mlp/runs/` | Logs de TensorBoard de esas mismas corridas. |
| `TP_regularizacion/mlflow.db` | Base de MLflow con los parámetros y las métricas por época de todas las corridas de `TNG_Regularizacion.ipynb`. |
| `requirements.txt` | Dependencias de Python. |
| `TP_cnn/CNN_en_Skin_Dataset.ipynb` | Clasificación de lesiones de piel con una red convolucional (CNN) sobre el Skin Dataset. |

## Qué hace el trabajo

1. **Datos:** se carga 20 Newsgroups con `fetch_20newsgroups`, quitando `headers`, `footers` y `quotes` para que el problema no sea trivial.
2. **Preprocesamiento configurable:** minúsculas, solo tokens alfabéticos, y opcionalmente remoción de stopwords y normalización (`none`, `stem` o `lemmatize`) con NLTK.
3. **Vectorización:** TF-IDF con `ngram_range`, `max_features` y `min_df` configurables, más normalización de los vectores.
4. **Modelo:** regresión softmax (una capa lineal + `CrossEntropyLoss`) en PyTorch. En `TP_mlp/TNG+MLP.ipynb`, un MLP con capas ocultas.
5. **Seguimiento de experimentos:** curvas de loss y accuracy en TensorBoard, y parámetros y modelos en MLflow.
6. **Optimización:** búsqueda de hiperparámetros de preprocesamiento y de entrenamiento (como `lr` y `batch_size`) con Optuna.
7. **Entrenamiento final y evaluación:** con los mejores hiperparámetros, sobre el test set original. Incluye matriz de confusión, `classification_report` y los pares de categorías que más se confunden.
8. **Análisis:** efecto del learning rate, de `remove_stopwords` y `normalization`, y, en el notebook del MLP, de la cantidad de capas y unidades ocultas frente a tiempos de corrida.

## Notebook de regularización (`TP_regularizacion/TNG_Regularizacion.ipynb`)

Sobre el mismo MLP de `TP_mlp/TNG+MLP.ipynb` (capas ocultas `[32, 128, 128, 64]`, ReLU, Adam) se incorporan como hiperparámetros:

- **Regularización L1, L2 y L1+L2**, como penalidad explícita sobre los pesos de las capas lineales.
- **Dropout**, con distintas probabilidades.
- **Batch Normalization**, combinada con distintos *learning rates*.

Para cada técnica se compara contra una línea base sin regularización y se registra: accuracy de validación, brecha train-validación, valor absoluto de los pesos (media, máximo y fracción cercana a cero), tiempo por época y ruido de la curva de aprendizaje. Después, Optuna busca la mejor combinación y el modelo final se evalúa una sola vez sobre el test set original.

El notebook se entrega con las salidas guardadas, así que se puede leer sin ejecutarlo.


## Cómo ejecutarlo

### Opción 1: Google Colab

Cada notebook tiene un botón **Open in Colab** al principio. La primera celda instala `mlflow` y `optuna` (el resto de las librerías ya viene en Colab). Conviene activar la GPU en *Entorno de ejecución → Cambiar tipo de entorno de ejecución*, aunque los notebooks también corren en CPU.

### Opción 2: entorno local

```bash
git clone https://github.com/valjeon/TP_Regresion_Softmax.git
cd TP_Regresion_Softmax

python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

El entorno virtual (`.venv`) va en la raíz del repo y lo comparten los tres notebooks. No se sube al repositorio (está en `.gitignore`).

Los notebooks descargan recursos de NLTK (`stopwords`, `wordnet` y `punkt`) en la celda de setup. Hace falta conexión a internet la primera vez, tanto para eso como para descargar el dataset.

## Notebook de CNN (`TP_cnn/CNN_en_Skin_Dataset.ipynb`)

Clasificación de imágenes de lesiones de piel (melanoma, nevus melanocítico, queratosis benigna, carcinoma de células escamosas, dermatofibroma, lesión vascular, dermatitis atópica, tiña, entre otras) con una red convolucional en PyTorch.

**Dataset:** no está incluido en este repositorio por su tamaño (~170 MB). Se descarga del repositorio de la cátedra y se copia la carpeta `data/` dentro de `TP_cnn/`:

```bash
git clone https://github.com/cselmo/CNN-en-Skin-Dataset.git
cp -r CNN-en-Skin-Dataset/data TP_cnn/data
```

Tiene que quedar así: `TP_cnn/data/Split_smol/train/<clase>/` y `TP_cnn/data/Split_smol/val/<clase>/`. La carpeta `TP_cnn/data/` está en `.gitignore`, así que no se sube.

### Ver los experimentos

Cada TP tiene su propia carpeta con su propia base de MLflow (`mlflow.db`): cada notebook escribe las corridas en la carpeta desde donde se ejecuta. Para explorar las corridas que quedaron guardadas en el repositorio, ejecutar **desde la carpeta del TP** (`TP_softmax`, `TP_mlp` o `TP_regularizacion`):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
tensorboard --logdir runs      # solo TP_softmax y TP_mlp
```

y abrir `http://localhost:5000` (MLflow) o `http://localhost:6006` (TensorBoard). Los experimentos son `softmax_20newsgroups`, `mlp_20newsgroups` y `regularizacion_20newsgroups`, con los parámetros y las métricas por época de cada corrida. Los modelos guardados de TP_softmax y TP_mlp están en `mlruns/` y se pueden cargar desde MLflow con `mlflow.pytorch.load_model`.

Las bases de TP_softmax y TP_mlp corresponden exactamente a las ejecuciones cuyas salidas quedaron en los notebooks. Las curvas de TensorBoard no se incluyen en el repositorio: los mismos gráficos están en el notebook, y si se vuelve a ejecutar se regeneran en `runs/reg`.

## Notas

- El entrenamiento usa GPU si está disponible y, si no, CPU.
- Se fija una semilla (y se vuelve a fijar antes de crear cada modelo, y en el sampler de Optuna) para que los resultados sean reproducibles en el mismo entorno. Entre CPU y GPU pueden aparecer diferencias numéricas chicas.
- La búsqueda con Optuna puede tardar bastante, sobre todo en el notebook del MLP.
