"""
AI Agronomist — Flower Recognition (VGG-16 Transfer Learning)
-------------------------------------------------------------
Fine-tunes a pre-trained VGG-16 on a custom flower dataset.
Achieved 97.6% validation accuracy in the final FYP model.

Dataset structure expected:
    dataset/
        train/
            rose/        ← one folder per class
            sunflower/
            tulip/
            ...
        val/
            rose/
            sunflower/
            ...

Author : Ghulam Sarwar
Thesis : AI-Based Agronomist — Sukkur IBA University (2022)
Award  : Winner DICE AFS 2021 (PKR 100,000)
"""

import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)      # VGG-16 native input size
BATCH_SIZE  = 32
EPOCHS_HEAD = 10              # train only the new head first
EPOCHS_FINE = 20              # then fine-tune top conv blocks
LR_HEAD     = 1e-3
LR_FINE     = 1e-5
DATASET_DIR = "dataset"
MODEL_OUT   = "models/vgg16_flower_recognition.h5"

NUM_CLASSES = len(os.listdir(os.path.join(DATASET_DIR, "train")))
print(f"Classes found: {NUM_CLASSES}")


# ── data pipeline ─────────────────────────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.15,
    shear_range=0.1,
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

# Save class names for use in the mobile app
import json
with open("models/class_names.json", "w") as f:
    json.dump(train_ds.class_indices, f, indent=2)
print("Class mapping saved to models/class_names.json")


# ── model ─────────────────────────────────────────────────────────────────────
base = VGG16(weights="imagenet", include_top=False,
             input_shape=(*IMG_SIZE, 3))
base.trainable = False          # freeze all conv blocks initially

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs=base.input, outputs=out)
model.summary()


# ── phase 1 — train head only ─────────────────────────────────────────────────
model.compile(
    optimizer=optimizers.Adam(LR_HEAD),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

cb_head = [
    callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True,
                               monitor="val_accuracy", verbose=1),
    callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                            monitor="val_accuracy"),
]

print("\n=== Phase 1: Training classification head ===")
hist1 = model.fit(train_ds, validation_data=val_ds,
                  epochs=EPOCHS_HEAD, callbacks=cb_head)


# ── phase 2 — fine-tune top conv blocks ───────────────────────────────────────
# Unfreeze the last two conv blocks (block4, block5)
for layer in base.layers:
    layer.trainable = layer.name.startswith(("block4", "block5"))

model.compile(
    optimizer=optimizers.Adam(LR_FINE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

cb_fine = [
    callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True,
                               monitor="val_accuracy", verbose=1),
    callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                            monitor="val_accuracy"),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
]

print("\n=== Phase 2: Fine-tuning top conv blocks ===")
hist2 = model.fit(train_ds, validation_data=val_ds,
                  epochs=EPOCHS_FINE, callbacks=cb_fine)


# ── plot training curves ───────────────────────────────────────────────────────
def plot_history(h1, h2):
    acc  = h1.history["accuracy"]      + h2.history["accuracy"]
    val  = h1.history["val_accuracy"]  + h2.history["val_accuracy"]
    loss = h1.history["loss"]          + h2.history["loss"]
    vloss= h1.history["val_loss"]      + h2.history["val_loss"]
    ep   = range(1, len(acc) + 1)
    split= len(h1.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ep, acc,  "b-",  label="Train acc")
    ax1.plot(ep, val,  "r-",  label="Val acc")
    ax1.axvline(split, color="gray", linestyle="--", label="Fine-tune start")
    ax1.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax1.legend(); ax1.grid(True)

    ax2.plot(ep, loss, "b-",  label="Train loss")
    ax2.plot(ep, vloss,"r-",  label="Val loss")
    ax2.axvline(split, color="gray", linestyle="--", label="Fine-tune start")
    ax2.set(title="Loss", xlabel="Epoch", ylabel="Loss")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig("images/training_curves.png", dpi=120)
    print("Training curves saved to images/training_curves.png")

plot_history(hist1, hist2)

# ── final evaluation ──────────────────────────────────────────────────────────
loss, acc = model.evaluate(val_ds)
print(f"\nFinal validation accuracy: {acc*100:.2f}%")
print(f"Model saved to: {MODEL_OUT}")
