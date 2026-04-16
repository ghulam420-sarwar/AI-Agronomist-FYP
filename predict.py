"""
AI Agronomist — Flower Prediction Script
-----------------------------------------
Load a saved model and predict the flower class from a single image
or a folder of images.

Usage:
    python predict.py --image path/to/flower.jpg --model vgg16
    python predict.py --folder path/to/images/   --model custom_cnn

Author: Ghulam Sarwar
"""

import argparse
import json
import pathlib
import numpy as np
from PIL import Image
import tensorflow as tf


MODEL_PATHS = {
    "vgg16":      ("models/vgg16_flower_recognition.h5",    (224, 224)),
    "custom_cnn": ("models/custom_cnn_flower_recognition.h5", (128, 128)),
}


def load_model_and_classes(model_name):
    path, img_size = MODEL_PATHS[model_name]
    model = tf.keras.models.load_model(path)
    with open("models/class_names.json") as f:
        class_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_idx.items()}
    return model, img_size, idx_to_class


def predict_image(model, img_size, idx_to_class, img_path):
    img = Image.open(img_path).convert("RGB").resize(img_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    top3 = np.argsort(preds)[::-1][:3]
    return [(idx_to_class[i], float(preds[i])) for i in top3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",  help="Path to a single image")
    ap.add_argument("--folder", help="Path to a folder of images")
    ap.add_argument("--model",  choices=["vgg16", "custom_cnn"],
                    default="vgg16")
    args = ap.parse_args()

    model, img_size, idx_to_class = load_model_and_classes(args.model)
    print(f"Model loaded: {args.model}")

    images = []
    if args.image:
        images = [pathlib.Path(args.image)]
    elif args.folder:
        images = list(pathlib.Path(args.folder).glob("*.jpg")) + \
                 list(pathlib.Path(args.folder).glob("*.png"))
    else:
        print("Provide --image or --folder"); return

    for img_path in images:
        results = predict_image(model, img_size, idx_to_class, img_path)
        print(f"\n{img_path.name}")
        for rank, (cls, conf) in enumerate(results, 1):
            print(f"  {rank}. {cls:<20} {conf*100:.1f}%")


if __name__ == "__main__":
    main()
