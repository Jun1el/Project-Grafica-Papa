"""Arquitectura y entrenamiento de la linea base ResNet50."""

from pathlib import Path

from .config import CLASS_NAMES, IMAGE_SIZE, MODEL_PATH


def build_model():
    """Crea la arquitectura utilizada para entrenar el modelo entregado."""
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )
    base_model = ResNet50(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            augmentation,
            layers.Lambda(preprocess_input, name="resnet50_preprocessing"),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(train_dataset, validation_dataset, output_path: Path = MODEL_PATH):
    """Entrena hasta 15 epocas y guarda el mejor estado restaurado."""
    import tensorflow as tf

    model = build_model()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=15,
        callbacks=[early_stopping],
    )
    model.save(Path(output_path))
    return model, history


def build_finetune_model(model):
    """Descongela los ultimos bloques de ResNet50 para fine-tuning."""
    import tensorflow as tf

    # Buscamos la capa base_model (resnet50)
    base_model = None
    for layer in model.layers:
        if layer.name == "resnet50":
            base_model = layer
            break

    if base_model is None:
        raise ValueError("No se encontro la capa 'resnet50' en el modelo")

    base_model.trainable = True

    # Congelar todas las capas excepto las ultimas 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Recompilar con un learning rate muy bajo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def do_finetune(train_dataset, validation_dataset, input_model_path: Path, output_path: Path):
    """Ejecuta el fine-tuning sobre un modelo pre-entrenado."""
    import tensorflow as tf
    from .config import IMAGE_SIZE

    # Reconstruimos la arquitectura en limpio para evitar el bug de pickling 
    # de Keras al intentar guardar una capa Lambda reconstruida desde el H5.
    model = build_model()
    
    # Construir el modelo explícitamente para que inicialice todas sus capas internas
    model.build(input_shape=(None, *IMAGE_SIZE, 3))
    
    # Usar by_name=True asegura que asigne los pesos a la capa correcta (resnet50, dense)
    # sin importar si la jerarquía de guardado del H5 difiere de la instancia no entrenada.
    model.load_weights(input_model_path, by_name=True)
    
    model = build_finetune_model(model)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=10,
        callbacks=[early_stopping],
    )
    
    model.save(Path(output_path))
    return model, history
