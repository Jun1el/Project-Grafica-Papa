# Clasificacion de Enfermedades en Hojas de Papa

Proyecto academico de Computacion Grafica que clasifica hojas de papa como
**tizon temprano**, **tizon tardio** u **hoja sana**. Utiliza Transfer Learning
con ResNet50 y contiene una implementacion experimental de segmentacion HSV +
Watershed con OpenCV.

> El resultado es educativo y no sustituye una evaluacion agronomica.

El orden de trabajo compartido y los criterios para cerrar cada etapa estan en
[`docs/PLAN_PROYECTO.md`](docs/PLAN_PROYECTO.md).

## Estado actual

- Dataset publico: Potato Disease Dataset de Kaggle, licencia CC0.
- Total: 2 152 imagenes (1 000 early blight, 1 000 late blight y 152 healthy).
- Modelo entregado: `modelo_papas_resnet50.h5`.
- Accuracy de validacion informada: 96.51 %.
- La fase 2 fue completada con un reparto reproducible 70/15/15 y metricas por
  clase. El modelo reentrenado obtuvo 97.83 % de accuracy y 96.65 % de macro F1
  en test.
- Watershed se conserva como demostracion de computacion grafica, pero no se
  aplica al clasificador: redujo el accuracy de test a 66.25 %.
- La fase 3 entrega la demo Gradio `app.py` para inferencia local o en Hugging
  Face Spaces, sin entrenamiento ni almacenamiento de imagenes de usuarios.
- La fase 4 (robustez con imagenes de campo) esta en progreso. Una prueba
  parcial con PlantDoc mostro que el accuracy cae de 97.83 % a 58.26 % fuera del
  dataset de laboratorio; ver
  [`docs/RESULTADOS_FASE_4.md`](docs/RESULTADOS_FASE_4.md).
- Las metricas y decisiones completas estan en
  [`docs/RESULTADOS_FASE_2.md`](docs/RESULTADOS_FASE_2.md).

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
|   |-- inference.py              Validacion y contrato de prediccion
|   `-- evaluation.py             Metricas y comparacion con Watershed
|-- scripts/
|   |-- prepare_split.py          Manifiesto estratificado 70/15/15
|   |-- evaluate.py               Reporte y matrices de confusion
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

## Preparar el reparto sin fugas

Después de descargar el dataset en `dataset_papa/`, crear el manifiesto:

```bash
python -m scripts.prepare_split
```

El reparto es estratificado: 70 % entrenamiento, 15 % validacion y 15 % prueba,
con semilla 123. Cada archivo se identifica mediante SHA-256; si existe contenido
duplicado, la preparacion se detiene para evitar fugas. `split_manifest.json`
contiene rutas locales y no se publica en Git.

## Reentrenar y evaluar

El entrenamiento usa exclusivamente `train` y `validation`:

```bash
python -m scripts.train
```

El modelo se guarda como `modelo_papas_resnet50_split70_15_15.h5`. El artefacto
historico `modelo_papas_resnet50.h5` no se sobrescribe, para poder conservarlo
como linea base. El nuevo modelo permanece ignorado por Git hasta completar y
aprobar su evaluacion.

La particion `test` debe permanecer cerrada hasta finalizar el entrenamiento.
Después se genera el reporte final una sola vez:

```bash
python -m scripts.evaluate --split test
```

El resultado se guarda en `outputs/evaluation/` e incluye accuracy, macro F1,
precision, recall y F1 por clase, matriz de confusion, cobertura del umbral de
confianza y comparacion con/sin Watershed. Los reportes locales no se versionan
hasta confirmar que proceden de un modelo reentrenado con el manifiesto actual.

Para ajustar el umbral de resultado no concluyente:

```bash
python -m scripts.evaluate --split validation --threshold 0.70
```

El umbral debe elegirse sobre validacion y congelarse antes de evaluar `test`.

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
