# Clasificacion de Enfermedades en Hojas de Papa

Proyecto academico de Computacion Grafica que clasifica hojas de papa como
**tizon temprano**, **tizon tardio** u **hoja sana**. Utiliza Transfer Learning
con ResNet50 y contiene una implementacion experimental de segmentacion HSV +
Watershed con OpenCV.

> El resultado es educativo y no sustituye una evaluacion agronomica.

## Estado actual

- Dataset publico: Potato Disease Dataset de Kaggle, licencia CC0.
- Total: 2 152 imagenes (1 000 early blight, 1 000 late blight y 152 healthy).
- Modelo entregado: `modelo_papas_resnet50.h5`.
- Accuracy de validacion informada: 96.51 %.
- Pendiente para la fase 2: conjunto de prueba independiente y metricas por clase.

El 96.51 % no debe interpretarse como rendimiento garantizado en fotografias de
campo. El modelo fue entrenado con las imagenes originales, sin aplicar la
segmentacion Watershed a su entrada.

## Estructura

```text
PAPA-PROYECTO/
|-- Dataset.ipynb                 Flujo exploratorio original
|-- modelo_papas_resnet50.h5      Modelo entrenado
|-- papa_disease/
|   |-- config.py                 Clases, rutas y limites compartidos
|   |-- data.py                   Carga reproducible del dataset
|   |-- training.py               Arquitectura y entrenamiento base
|   |-- segmentation.py           Segmentacion HSV + Watershed
|   `-- inference.py              Validacion y contrato de prediccion
|-- scripts/
|   |-- train.py                  Entrada para entrenamiento
|   `-- predict.py                Entrada para inferencia local
|-- tests/                        Pruebas del contrato
`-- requirements.txt              Dependencias del proyecto
```

## Contrato del modelo

### Entrada

- Archivo JPG, JPEG o PNG.
- Maximo 10 MB y 25 megapixeles.
- La imagen se decodifica como RGB y se redimensiona a 224 x 224.
- `preprocess_input` de ResNet50 ya forma parte del modelo guardado.

### Orden inmutable de las salidas

| Indice | Identificador | Etiqueta mostrada |
|---:|---|---|
| 0 | `Potato___Early_blight` | Tizon temprano |
| 1 | `Potato___Late_blight` | Tizon tardio |
| 2 | `Potato___healthy` | Hoja sana |

La salida de inferencia contiene `class_id`, `label`, `confidence`, las
probabilidades de las tres clases y un aviso de uso educativo.

## Instalacion

Se utiliza Python 3.13 y un entorno virtual limpio. Las versiones fijadas fueron
seleccionadas para disponer de paquetes compatibles en Windows y deben tratarse
como una unidad al actualizarse.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

No se deben agregar al repositorio `kaggle.json`, credenciales, entornos
virtuales, el dataset descargado ni imagenes privadas.

Las versiones de `requirements.txt` estan fijadas para que el entorno sea
repetible. Si se actualiza alguna dependencia, se debe volver a ejecutar toda la
verificacion, incluida una inferencia real con el modelo.

## Inferencia local

Desde la raiz del proyecto:

```bash
python -m scripts.predict ruta/a/hoja.jpg
```

Watershed no se aplica automaticamente. Su efecto sobre la precision debe
compararse en la fase 2 antes de incorporarlo al flujo de clasificacion.

## Reproducir el entrenamiento base

1. Descargar el dataset en `dataset_papa/` conservando las tres carpetas de
   clases.
2. Ejecutar:

```bash
python -m scripts.train
```

El flujo reproduce el split original 80/20 con semilla 123, batch de 32 y hasta
15 epocas. Reentrenar sobrescribe el modelo si se usa la ruta predeterminada;
se recomienda conservar una copia versionada del modelo validado.

## Segmentacion Watershed

`papa_disease.segmentation.apply_watershed` recibe una imagen OpenCV en BGR y
devuelve mascara, marcadores, contornos y hoja aislada. La mascara HSV actual fue
diseñada para hojas verdes sobre fondos relativamente uniformes y puede fallar
con lesiones extensas, sombras o fondos de campo.

## Verificacion

Las pruebas ligeras, que no requieren TensorFlow, se ejecutan con:

```bash
python -m unittest discover -s tests -v
python -m compileall papa_disease scripts tests
```

La carga e inferencia real requieren instalar las dependencias y usar una imagen
JPG o PNG valida.

El archivo H5 contiene una capa Lambda heredada. Por ese motivo solo debe
cargarse el modelo versionado y confiable del proyecto; nunca un `.h5` enviado
por un usuario o descargado de una fuente no verificada.

## Licencia del dataset

[Potato Disease Dataset](https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset),
publicado bajo CC0 1.0.
