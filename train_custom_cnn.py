"""
AI Agronomist — Flower Recognition (Custom CNN)
-----------------------------------------------
Builds and trains a custom Convolutional Neural Network from scratch.
Compared against VGG-16 transfer learning as part of the FYP evaluation.

Author : Ghulam Sarwar
Thesis : AI-Based Agronomist — Sukkur IBA University (2022)
"""

import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
EPOCHS      = 40
LR          = 1e-3
DATASET_DIR = "dataset"
MODEL_OUT   = "models/custom_cnn_flower_recognition.h5"

NUM_CLASSES = len(os.listdir(os.path.join(DATASET_DIR, "train")))


# ── data pipeline ─────────────────────────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.15,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)
val_gen = ImageDataGenerator(rescale=1.0 / 255)

train_ds = train_gen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)
val_ds = val_gen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)


# ── model architecture ────────────────────────────────────────────────────────
def build_custom_cnn(input_shape, num_classes):
    """
    5-block custom CNN:
      Block 1-2 : feature extraction (shallow)
      Block 3-5 : deeper features with increasing filters
      Head      : global avg pool → dense → dropout → softmax
    """
    inp = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # Block 4
    x = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # Block 5
    x = layers.Conv2D(512, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.4)(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs=inp, outputs=out, name="CustomCNN")


model = build_custom_cnn((*IMG_SIZE, 3), NUM_CLASSES)
model.summary()

model.compile(
    optimizer=optimizers.Adam(LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

cb = [
    callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True,
                               monitor="val_accuracy", verbose=1),
    callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1),
]

print("\n=== Training Custom CNN ===")
history = model.fit(train_ds, validation_data=val_ds,
                    epochs=EPOCHS, callbacks=cb)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history["accuracy"],     label="Train")
ax1.plot(history.history["val_accuracy"], label="Val")
ax1.set(title="Custom CNN — Accuracy", xlabel="Epoch", ylabel="Accuracy")
ax1.legend(); ax1.grid(True)

ax2.plot(history.history["loss"],     label="Train")
ax2.plot(history.history["val_loss"], label="Val")
ax2.set(title="Custom CNN — Loss", xlabel="Epoch", ylabel="Loss")
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig("images/custom_cnn_curves.png", dpi=120)

loss, acc = model.evaluate(val_ds)
print(f"\nCustom CNN validation accuracy: {acc*100:.2f}%")
print(f"Model saved to: {MODEL_OUT}")
