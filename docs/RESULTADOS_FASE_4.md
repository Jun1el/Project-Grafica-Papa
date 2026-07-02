# Resultados de la fase 4 (parcial)

Prueba de robustez del modelo `modelo_papas_resnet50_split70_15_15.h5` sobre
imagenes fuera del dataset de laboratorio. El objetivo no es lucir una metrica
alta, sino medir cuanto cae el desempeno con fotografias reales.

> Estado: **parcial**. Falta la clase `Potato___healthy` y una revision de
> etiquetas y duplicados. Los numeros aun no son un cierre de fase.

## Origen de las imagenes

| Fuente | Clases aportadas | Imagenes | Licencia |
|---|---|---:|---|
| PlantDoc (train+test) | Tizon temprano, Tizon tardio | 114 + 104 | CC BY 4.0 |
| Pendiente | Hoja sana | 0 | — |

Las imagenes se guardan en `dataset_campo/` (ignorado por Git). PlantDoc no
incluye hoja de papa sana, por eso esa clase queda pendiente.

## Resultado de la ejecucion parcial

Modelo reentrenado, umbral 0.70, 218 imagenes en dos clases.

| Metrica | Campo (parcial) | Test de laboratorio |
|---|---:|---:|
| Accuracy | 58.26 % | 97.83 % |
| Macro F1 (clases presentes) | 58.10 % | 96.65 % |
| Cobertura sobre umbral 0.70 | 68.81 % | 90.40 % |
| Accuracy aceptada | 58.00 % | 99.32 % |

Matriz de confusion (filas = clase real, columnas = prediccion), orden
`[Tizon temprano, Tizon tardio, Hoja sana]`:

```
[48, 66,  0]
[20, 79,  5]
[ 0,  0,  0]
```

Por clase:

| Clase | Soporte | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Tizon temprano | 114 | 70.6 % | 42.1 % | 52.7 % |
| Tizon tardio | 104 | 54.5 % | 76.0 % | 63.5 % |
| Hoja sana | 0 | — | — | — |

## Lectura

- La accuracy cae de 97.83 % a 58.26 %: casi 40 puntos. El 96.51 % del README
  se confirma como rendimiento de laboratorio, no de campo.
- El modelo confunde con frecuencia tizon temprano con tardio (66 casos). El
  recall de tizon temprano se desploma a 42.1 %.
- La cobertura baja del 90 % al 69 %: el modelo esta menos seguro con fotos
  reales, que es el comportamiento deseable del umbral.

## Pendientes para cerrar la fase 4

1. Agregar imagenes de campo de hoja sana (`Potato___healthy`).
2. Verificar etiquetas de tizon temprano vs. tardio (se confunden con facilidad).
3. Descartar duplicados frente al dataset de laboratorio para evitar fugas.
4. Repetir la evaluacion con las tres clases y actualizar esta tabla.

## Reproduccion

```bash
python -m scripts.evaluate_field
```

El reporte completo queda en `outputs/evaluation/field_report.json` y las
matrices en `outputs/evaluation/field_raw.png` (no se versionan).
