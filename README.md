# Clasificacion de Enfermedades en Hojas de Papa

Sistema de clasificacion automatica de enfermedades en hojas de papa mediante redes neuronales convolucionales (CNN) con Transfer Learning utilizando **ResNet50** y segmentacion de imagenes con el algoritmo **Watershed** (OpenCV).

El modelo distingue entre tres estados de la planta:

| Clase | Descripcion |
|-------|-------------|
| `Potato___Early_blight` | Tizon temprano |
| `Potato___Late_blight` | Tizon tardio |
| `Potato___healthy` | Hoja saludable |

---

## Arquitectura General

```mermaid
flowchart TD
    A["Dataset Kaggle\nPotato Disease Dataset"] --> B["Descarga y extraccion\nKaggle API"]
    B --> C["Carga de imagenes\ntf.keras.utils.image_dataset_from_directory"]
    C --> D["Data Augmentation\nFlip, Rotation, Zoom, Contrast"]
    D --> E["Optimizacion del pipeline\ncache + shuffle + prefetch"]
    E --> F["Segmentacion Watershed\nOpenCV"]
    E --> G["Entrenamiento ResNet50\nTransfer Learning"]
    G --> H["Modelo entrenado\nmodelo_papas_resnet50.h5"]
```

---

## Arquitectura del Modelo

```mermaid
flowchart TD
    INPUT["Input\n224 x 224 x 3"] --> AUG["Data Augmentation\nRandomFlip, RandomRotation\nRandomZoom, RandomContrast"]
    AUG --> PRE["Lambda\npreprocess_input ResNet50"]
    PRE --> RESNET["ResNet50\nweights=imagenet\ninclude_top=False\ntrainable=False"]
    RESNET --> GAP["GlobalAveragePooling2D"]
    GAP --> DROP["Dropout\nrate=0.3"]
    DROP --> DENSE["Dense\nunits=3, activation=softmax"]
    DENSE --> OUT["Output\nEarly_blight | Late_blight | healthy"]
```

**Configuracion de entrenamiento:**

| Parametro | Valor |
|-----------|-------|
| Optimizador | Adam (lr=0.0001) |
| Funcion de perdida | sparse_categorical_crossentropy |
| Early Stopping | patience=3, monitor=val_loss |
| Epocas maximas | 15 |
| Batch Size | 32 |

---

## Segmentacion Watershed

```mermaid
flowchart TD
    IMG["Imagen original"] --> GRAY["Escala de grises"]
    GRAY --> OTSU["Binarizacion Otsu\nTHRESH_BINARY_INV"]
    OTSU --> OPEN["Opening morfologico\neliminacion de ruido"]
    OPEN --> BG["Dilatacion\nfondo seguro"]
    OPEN --> DT["Distance Transform\nprimer plano seguro"]
    BG --> UNKNOWN["Region desconocida\nsustraccion"]
    DT --> UNKNOWN
    UNKNOWN --> MARKERS["Etiquetado de marcadores\nconnectedComponents"]
    MARKERS --> WS["Watershed\nsegmentacion final"]
    WS --> RESULT["Contornos detectados"]
```

El modulo `Segmentacion.py` implementa el algoritmo Watershed de OpenCV para aislar la hoja del fondo, utilizando binarizacion de Otsu, operaciones morfologicas y transformada de distancia.

---

## Dataset

**Fuente:** [Potato Disease Dataset - Kaggle](https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset)
**Licencia:** CC0 1.0 (Dominio Publico)

| Clase | Imagenes |
|-------|----------|
| Potato___Early_blight | 1,000 |
| Potato___Late_blight | 1,000 |
| Potato___healthy | 152 |
| **Total** | **2,152** |

**Distribucion del pipeline:**

| Conjunto | Imagenes | Porcentaje |
|----------|----------|------------|
| Entrenamiento | 1,722 | 80% |
| Validacion | 430 | 20% |

---

## Estructura del Proyecto

```
PAPA-PROYECTO/
├── Dataset.ipynb                    # Notebook principal: descarga, preprocesamiento,
│                                    # segmentacion, entrenamiento y visualizacion
├── Entrenamiento.py                 # Script standalone del entrenamiento con ResNet50
├── Segmentacion.py                  # Script standalone de segmentacion Watershed
├── modelo_papas_resnet50.h5         # Modelo entrenado (pesos y arquitectura)
├── potato-dataset-metadata.json     # Metadata del dataset (formato Croissant/MLCommons)
└── .gitignore
```

---

## Tecnologias

| Libreria | Uso |
|----------|-----|
| TensorFlow / Keras | Framework de Deep Learning, construccion y entrenamiento del modelo |
| ResNet50 (ImageNet) | Arquitectura base preentrenada (Transfer Learning) |
| OpenCV | Segmentacion de imagenes con algoritmo Watershed |
| NumPy | Operaciones numericas y manipulacion de arrays |
| Matplotlib | Visualizacion de resultados y Data Augmentation |
| Kaggle API | Descarga automatizada del dataset |
| Jupyter Notebook | Entorno de desarrollo interactivo |

---

## Requisitos

- Python 3.13+
- TensorFlow
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- Kaggle API

```bash
pip install tensorflow opencv-python numpy matplotlib kaggle
```

Para la descarga del dataset se requiere el archivo `kaggle.json` con las credenciales de la API de Kaggle en el directorio de trabajo.

---

## Uso

### Notebook principal (recomendado)

El archivo `Dataset.ipynb` contiene el flujo completo: descarga del dataset, preprocesamiento, Data Augmentation, segmentacion Watershed, entrenamiento y guardado del modelo.

```bash
jupyter notebook Dataset.ipynb
```

### Entrenamiento standalone

```bash
python Entrenamiento.py
```

Requiere que `train_dataset`, `validation_dataset`, `class_names` y `data_augmentation` esten definidos previamente en el entorno.

### Segmentacion standalone

```bash
python Segmentacion.py
```

Requiere una imagen de prueba `papa_prueba.jpg` en el directorio de trabajo. La ruta puede modificarse en la funcion `aplicar_watershed()`.

---

## Resultados del Entrenamiento

| Epoca | Accuracy | Loss | Val Accuracy | Val Loss |
|-------|----------|------|--------------|----------|
| 1 | 0.5319 | 0.9528 | 0.7651 | 0.6293 |
| 5 | 0.8682 | 0.3537 | 0.9116 | 0.2580 |
| 10 | 0.9245 | 0.2058 | 0.9512 | 0.1660 |
| 15 | 0.9570 | 0.1436 | 0.9651 | 0.1285 |

**Mejor resultado:** Val Accuracy = **96.51%** | Val Loss = **0.1285**

```mermaid
xychart-beta
    title "Curva de Accuracy durante el Entrenamiento"
    x-axis ["Ep 1", "Ep 2", "Ep 3", "Ep 4", "Ep 5", "Ep 6", "Ep 7", "Ep 8", "Ep 9", "Ep 10", "Ep 11", "Ep 12", "Ep 13", "Ep 14", "Ep 15"]
    y-axis "Accuracy" 0.5 --> 1.0
    line [0.53, 0.69, 0.80, 0.86, 0.87, 0.90, 0.92, 0.92, 0.93, 0.92, 0.95, 0.95, 0.95, 0.95, 0.96]
    line [0.77, 0.86, 0.88, 0.90, 0.91, 0.93, 0.93, 0.94, 0.94, 0.95, 0.96, 0.96, 0.96, 0.96, 0.97]
```

---

## Licencia

Este proyecto utiliza el **Potato Disease Dataset** disponible en Kaggle bajo licencia **CC0 1.0 (Dominio Publico)**.
