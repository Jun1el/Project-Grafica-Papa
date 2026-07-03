# Avance de la Fase 7: Flutter Android

## Implementación terminada

La aplicación **Papa Sana** vive en `mobile/` y realiza inferencia local con el
artefacto TFLite aprobado en la Fase 6. Permite usar cámara o galería, corrige la
orientación, redimensiona a RGB 224 x 224 y presenta las tres probabilidades con
el umbral no concluyente de 0.70. No contiene API, cuentas, almacenamiento ni
envío de imágenes.

## Verificación realizada

- Flutter 3.44.4 y Dart 3.12.2.
- Android SDK 36 y emulador `Papa_API_36` (Android 16, x86_64).
- `flutter analyze`: sin problemas.
- `flutter test`: 4 pruebas aprobadas.
- Pruebas Python: 16 ejecutadas, 15 aprobadas y 1 omitida por TensorFlow no
  instalado en el intérprete global.
- `python -m compileall papa_disease scripts tests`: aprobado.
- APK debug generado e instalado correctamente en el emulador.
- Arranque de la aplicación y carga del modelo confirmados en el emulador.

## Validación pendiente para cerrar la fase

Antes de marcar la fase como completada se debe probar cámara, galería e
inferencia con imágenes verificadas de las tres clases en al menos un celular
Android físico. Registrar modelo del teléfono, versión Android y latencias. Esta
prueba no puede sustituirse por el emulador porque debe medir cámara, memoria y
rendimiento reales.
