"""
Plant Disease Detection System - Model Training Script
======================================================
Trains a CNN model using MobileNetV2 transfer learning on the PlantVillage dataset
for plant disease classification.

Usage:
    python train_model.py

Prerequisites:
    - Download the PlantVillage dataset and organize it as:
      dataset/PlantVillage/<ClassName>/<images>
    - Install dependencies: pip install -r requirements.txt
"""

import os
import json
import random
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving plots
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Set seeds for reproducibility BEFORE importing TensorFlow
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================
DATASET_DIR = os.path.join('dataset', 'PlantVillage')
MODEL_SAVE_PATH = os.path.join('models', 'plant_disease_model.keras')
CLASS_NAMES_PATH = os.path.join('models', 'class_names.json')
CONFUSION_MATRIX_PATH = os.path.join('models', 'confusion_matrix.png')
TRAINING_HISTORY_PATH = os.path.join('models', 'training_history.png')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 15          # Transfer learning (frozen base)
EPOCHS_PHASE2 = 10          # Fine-tuning (top layers unfrozen)
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 1e-5
VALIDATION_SPLIT = 0.10
TEST_SPLIT = 0.10


def check_dataset():
    """Check if dataset directory exists and has class subdirectories."""
    if not os.path.isdir(DATASET_DIR):
        print("=" * 70)
        print("ERROR: Dataset directory not found!")
        print(f"Expected path: {os.path.abspath(DATASET_DIR)}")
        print()
        print("Please download the PlantVillage dataset and organize it as:")
        print(f"  {DATASET_DIR}/")
        print(f"    Apple___Apple_scab/")
        print(f"      image1.jpg")
        print(f"      image2.jpg")
        print(f"    Apple___Black_rot/")
        print(f"      ...")
        print(f"    Tomato___healthy/")
        print(f"      ...")
        print()
        print("Download from:")
        print("  https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("  https://github.com/spMohanty/PlantVillage-Dataset")
        print()
        print("See dataset/README.md for detailed instructions.")
        print("=" * 70)
        return False

    classes = [d for d in os.listdir(DATASET_DIR)
               if os.path.isdir(os.path.join(DATASET_DIR, d))]
    if len(classes) == 0:
        print("ERROR: No class subdirectories found in", DATASET_DIR)
        print("Each subdirectory should be named after its class label.")
        return False

    total_images = 0
    for cls in classes:
        cls_path = os.path.join(DATASET_DIR, cls)
        imgs = [f for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(imgs)

    print(f"[OK] Dataset found: {len(classes)} classes, {total_images:,} images")
    return True


def load_datasets():
    """Load and split dataset into train, validation, and test sets."""
    print("\n[1/6] Loading dataset...")

    # Load 80% train + 20% temp (which we split into val + test)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT + TEST_SPLIT,
        subset='training',
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True
    )

    temp_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT + TEST_SPLIT,
        subset='validation',
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"  Classes: {num_classes}")
    print(f"  Class names: {class_names[:5]}{'...' if num_classes > 5 else ''}")

    # Split temp into validation (50%) and test (50%)
    temp_size = tf.data.experimental.cardinality(temp_ds).numpy()
    val_size = temp_size // 2
    val_ds = temp_ds.take(val_size)
    test_ds = temp_ds.skip(val_size)

    print(f"  Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"  Validation batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"  Test batches: {tf.data.experimental.cardinality(test_ds).numpy()}")

    # Normalize pixel values to [0, 1]
    normalization = tf.keras.layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization(x), y))

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names, num_classes


def create_data_augmentation():
    """Create data augmentation pipeline for training."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.1),
    ], name='data_augmentation')


def build_model(num_classes):
    """Build CNN model using MobileNetV2 transfer learning."""
    print("\n[2/6] Building model...")

    # Load MobileNetV2 with ImageNet weights (exclude top classification layer)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model for Phase 1

    # Data augmentation layer
    data_augmentation = create_data_augmentation()

    # Build the complete model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    print(f"  Base model: MobileNetV2 (ImageNet weights, frozen)")
    print(f"  Total params: {model.count_params():,}")
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    print(f"  Trainable params: {trainable_params:,}")

    return model, base_model


def train_phase1(model, train_ds, val_ds):
    """Phase 1: Train with frozen base (transfer learning)."""
    print("\n[3/6] Phase 1 - Transfer Learning (frozen base)...")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks,
        verbose=1
    )

    return history1


def train_phase2(model, base_model, train_ds, val_ds):
    """Phase 2: Fine-tune top layers of base model."""
    print("\n[4/6] Phase 2 - Fine-Tuning (top layers unfrozen)...")

    # Unfreeze the top 30 layers of the base model
    base_model.trainable = True
    fine_tune_at = max(0, len(base_model.layers) - 30)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    print(f"  Unfrozen layers: {len(base_model.layers) - fine_tune_at}")
    print(f"  Trainable params now: {trainable_params:,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE2),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-8,
            verbose=1
        )
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        callbacks=callbacks,
        verbose=1
    )

    return history2


def evaluate_model(model, test_ds, class_names):
    """Evaluate model on test set and print metrics."""
    print("\n[5/6] Evaluating model on test set...")

    # Get predictions
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Overall accuracy
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
    print(f"\n  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Classification report
    if HAS_SKLEARN:
        print("\n  Classification Report:")
        print("-" * 70)
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if HAS_MATPLOTLIB:
            save_confusion_matrix(cm, class_names)
    else:
        print("  [INFO] scikit-learn not available; calculated basic accuracy.")

    return accuracy


def save_confusion_matrix(cm, class_names):
    """Save confusion matrix as an image."""
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 0.5),
                                    max(10, len(class_names) * 0.4)))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    if len(class_names) <= 20:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


def save_training_history(history1, history2):
    """Save training history plots (accuracy and loss)."""
    if not HAS_MATPLOTLIB:
        print("  [INFO] matplotlib not installed; skipping plot generation.")
        return
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    epochs_range = range(1, len(acc) + 1)
    phase1_end = len(history1.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(epochs_range, acc, 'g-', linewidth=2, label='Training Accuracy')
    ax1.plot(epochs_range, val_acc, 'b-', linewidth=2, label='Validation Accuracy')
    ax1.axvline(x=phase1_end, color='r', linestyle='--', alpha=0.7,
                label=f'Fine-tuning starts (epoch {phase1_end})')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(epochs_range, loss, 'g-', linewidth=2, label='Training Loss')
    ax2.plot(epochs_range, val_loss, 'b-', linewidth=2, label='Validation Loss')
    ax2.axvline(x=phase1_end, color='r', linestyle='--', alpha=0.7,
                label=f'Fine-tuning starts (epoch {phase1_end})')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(TRAINING_HISTORY_PATH, dpi=150)
    plt.close()
    print(f"  Training history saved to: {TRAINING_HISTORY_PATH}")


def save_model_and_classes(model, class_names):
    """Save trained model and class names."""
    print("\n[6/6] Saving model and class names...")

    os.makedirs('models', exist_ok=True)

    model.save(MODEL_SAVE_PATH)
    print(f"  Model saved to: {MODEL_SAVE_PATH}")

    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"  Class names saved to: {CLASS_NAMES_PATH}")


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("  Plant Disease Detection System - Model Training")
    print("=" * 70)

    # Check dataset
    if not check_dataset():
        return

    # Create output directory
    os.makedirs('models', exist_ok=True)

    # Load datasets
    train_ds, val_ds, test_ds, class_names, num_classes = load_datasets()

    # Build model
    model, base_model = build_model(num_classes)

    # Phase 1: Transfer learning
    history1 = train_phase1(model, train_ds, val_ds)

    # Phase 2: Fine-tuning
    history2 = train_phase2(model, base_model, train_ds, val_ds)

    # Evaluate
    accuracy = evaluate_model(model, test_ds, class_names)

    # Save training history plots
    save_training_history(history1, history2)

    # Save model and class names
    save_model_and_classes(model, class_names)

    print("\n" + "=" * 70)
    print(f"  Training Complete!")
    print(f"  Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Model: {MODEL_SAVE_PATH}")
    print(f"  Classes: {CLASS_NAMES_PATH}")
    print("=" * 70)


if __name__ == '__main__':
    main()
