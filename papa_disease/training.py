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

