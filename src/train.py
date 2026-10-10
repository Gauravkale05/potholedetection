import os
import time
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

def build_model(input_shape=(224, 224, 3)):
    """
    Builds MobileNetV2 architecture with custom classification head.
    Class 0: Normal, Class 1: Pothole.
    """
    # Data Augmentation layer
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")

    # Base pretrained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model initially

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid', name="output_pothole")(x)

    model = models.Model(inputs, outputs, name="Pothole_MobileNetV2")
    return model, base_model

def train_pothole_model(
    data_dir: str = "dataset",
    model_save_path: str = "models/pothole_mobilenetv2.keras",
    epochs_phase1: int = 8,
    epochs_phase2: int = 5,
    batch_size: int = 32,
    img_size: tuple = (224, 224)
):
    print("=" * 60)
    print("STARTING MOBILENETV2 MODEL TRAINING")
    print("=" * 60)
    
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "validation"
    
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Training or Validation directory not found in {data_dir}. Run prepare_data.py first.")
        
    print(f"Loading datasets from {data_dir}...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        label_mode='binary'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode='binary'
    )
    
    class_names = train_ds.class_names
    print(f"Class labels identified: {class_names} (0: {class_names[0]}, 1: {class_names[1]})")
    
    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    
    # Build Model
    model, base_model = build_model(input_shape=(*img_size, 3))
    model.summary()
    
    # Ensure models dir exists
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Compile Phase 1
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    callbacks_p1 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("\n--- Phase 1: Training Classification Head (Base Frozen) ---")
    start_time = time.time()
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        callbacks=callbacks_p1
    )
    
    # Phase 2: Fine-Tuning
    print("\n--- Phase 2: Fine-Tuning Top Layers of MobileNetV2 ---")
    base_model.trainable = True
    # Freeze all layers before layer 120 (out of ~154)
    fine_tune_at = 120
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    callbacks_p2 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    total_epochs = epochs_phase1 + epochs_phase2
    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=len(history_p1.epoch),
        callbacks=callbacks_p2
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/60:.2f} minutes.")
    print(f"Best model saved to: {model_save_path}")

if __name__ == "__main__":
    train_pothole_model()

