# Papa Sana para Android

Aplicación Flutter que clasifica hojas de papa completamente offline mediante
`modelo_papas.tflite`. Las fotografías permanecen en el dispositivo.

## Preparación

1. Instalar Flutter estable y Android SDK 36.
2. Ejecutar `flutter doctor` y aceptar las licencias Android.
3. Desde esta carpeta, ejecutar:

   ```bash
   flutter pub get
   flutter analyze
   flutter test
   flutter run
   ```

El emulador local preparado para el proyecto se llama `Papa_API_36`. Para usar
un celular físico, habilitar las opciones de desarrollador y la depuración USB,
conectarlo y confirmar que aparezca en `flutter devices`.

## Contrato de inferencia

- Recurso: `assets/models/modelo_papas.tflite`.
- Entrada: `[1, 224, 224, 3]`, RGB `float32`, píxeles de `0` a `255`.
- Salida: `[1, 3]`, `float32`.
- Orden: tizón temprano, tizón tardío y hoja sana.
- Confianza menor a `0.70`: resultado no concluyente.
- Archivos aceptados: JPG/PNG, máximo 10 MB y 25 megapíxeles.

El APK de prueba se genera con `flutter build apk --debug` y queda en
`build/app/outputs/flutter-apk/app-debug.apk`. La carpeta `build/` no se
versiona.
