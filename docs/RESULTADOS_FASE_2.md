# Resultados de la fase 2

La evaluacion final se ejecuto en Google Colab con GPU el 1 de julio de 2026,
usando el reparto estratificado 70/15/15, semilla 123 y umbral de confianza
0.70. El modelo fue reentrenado exclusivamente con `train` y `validation`; el
conjunto `test` se evaluo al finalizar.

## Resultados aprobados

| Entrada | Split | Accuracy | Macro F1 | Cobertura | Accuracy aceptada |
|---|---:|---:|---:|---:|---:|
| Imagen original | Validation | 96.28 % | 95.45 % | 86.38 % | 98.92 % |
| Imagen original | Test | 97.83 % | 96.65 % | 90.40 % | 99.32 % |
| Watershed | Validation | 68.73 % | 48.92 % | 74.61 % | 76.76 % |
| Watershed | Test | 66.25 % | 46.67 % | 71.52 % | 80.52 % |

En test, la clase minoritaria `Potato___healthy` obtuvo precision 91.67 %,
recall 95.65 % y F1 93.62 % sobre 23 imagenes. La matriz de confusion sin
segmentacion fue `[[147, 3, 0], [1, 147, 2], [0, 1, 22]]`.

## Decisiones

- Se aprueba el modelo `modelo_papas_resnet50_split70_15_15.h5` como base para
  la demo de la fase 3.
- No se aplica Watershed automaticamente porque degrada claramente todas las
  metricas. Se mantiene solo como tecnica experimental y visual.
- Las predicciones con confianza menor a 0.70 se consideran no concluyentes.
- El dataset, el manifiesto local, el modelo y el ZIP de respaldo no se
  versionan en Git. Cada colaborador descarga el dataset publico en Colab y
  regenera el manifiesto mediante `python -m scripts.prepare_split`.

El ZIP local conserva el modelo, el manifiesto, los reportes JSON y las matrices
PNG de la misma ejecucion. El modelo debe publicarse como artefacto externo
controlado para la fase 3, no como archivo ordinario del repositorio.
