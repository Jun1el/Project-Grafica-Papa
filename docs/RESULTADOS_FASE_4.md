# Resultados de la fase 4 (parcial)

Prueba de robustez del modelo `modelo_papas_resnet50_split70_15_15.h5` sobre
imagenes fuera del dataset de laboratorio. El objetivo no es lucir una metrica
alta, sino medir cuanto cae el desempeno con fotografias reales.

> Estado: **parcial**. Las tres clases ya tienen imagenes, pero falta curar la
> clase `Potato___healthy` (ver advertencia) y revisar etiquetas y duplicados.
> Los numeros aun no son un cierre de fase.

## Origen de las imagenes

| Fuente | Clases aportadas | Imagenes | Licencia |
|---|---|---:|---|
| PlantDoc (train+test) | Tizon temprano, Tizon tardio | 114 + 104 | CC BY 4.0 |
| iNaturalist (research grade) | Hoja sana | 120 | CC0 / CC-BY / CC-BY-SA |

Las imagenes se guardan en `dataset_campo/` (ignorado por Git). La atribucion
por archivo de la clase sana esta en
`dataset_campo/Potato___healthy/ATTRIBUTION.txt`.

## Resultado de la ejecucion (3 clases)

Modelo reentrenado, umbral 0.70, 338 imagenes.

| Metrica | Campo | Test de laboratorio |
|---|---:|---:|
| Accuracy | 37.87 % | 97.83 % |
| Macro F1 (clases presentes) | 31.18 % | 96.65 % |
| Cobertura sobre umbral 0.70 | 67.16 % | 90.40 % |

Matriz de confusion (filas = clase real, columnas = prediccion), orden
`[Tizon temprano, Tizon tardio, Hoja sana]`:

```
[48, 66,  0]
[20, 79,  5]
[31, 88,  1]
```

Por clase:

| Clase | Soporte | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Tizon temprano | 114 | 48.5 % | 42.1 % | 45.1 % |
| Tizon tardio | 104 | 33.9 % | 76.0 % | 46.9 % |
| Hoja sana | 120 | 16.7 % | 0.8 % | 1.6 % |

## Lectura

- La accuracy cae de 97.83 % a 37.87 %: unos 60 puntos. El 96.51 % del README
  se confirma como rendimiento de laboratorio, no de campo.
- La clase `Potato___healthy` colapsa: recall 0.8 % (solo 1 de 120). El modelo
  etiqueta casi toda hoja sana real como enferma. En un uso real esto seria un
  exceso de falsos positivos de enfermedad.
- Sesgo marcado hacia tizon tardio: 88 de 120 hojas sanas y 66 de 114 tizones
  tempranos se predicen como tardio.

## Advertencia sobre la clase sana

Las imagenes sanas provienen de observaciones de *Solanum tuberosum* en
iNaturalist: incluyen plantas enteras, flores, tuberculos y cultivos completos,
no solo primeros planos de una hoja como en el entrenamiento. Parte de la caida
de la clase sana se debe a esa diferencia de composicion (cambio de dominio),
no solo a la salud de la hoja. Antes de un cierre formal hay que **curar el
conjunto a primeros planos de follaje sano** para una comparacion justa.

## Pendientes para cerrar la fase 4

1. Curar la clase sana a primeros planos de hoja (descartar flores, tuberculos y
   tomas de cultivo completo).
2. Verificar etiquetas de tizon temprano vs. tardio (se confunden con facilidad).
3. Descartar duplicados frente al dataset de laboratorio para evitar fugas.
4. Repetir la evaluacion y actualizar esta tabla.

## Reproduccion

```bash
python -m scripts.evaluate_field
```

El reporte completo queda en `outputs/evaluation/field_report.json` y las
matrices en `outputs/evaluation/field_raw.png` (no se versionan).
