# Mobile App — AI Agronomist

The trained VGG-16 flower recognition model was deployed as a mobile application, completing the full pipeline from research to a usable product.

## Features

- Point camera at any flower → instant species identification
- Top-3 predictions shown with confidence percentages
- Works offline — model runs on-device
- Gallery mode — identify flowers from saved photos

## How It Works

```
User takes photo
       │
       ▼
Image pre-processed
(resized to 224×224, normalised)
       │
       ▼
On-device CNN inference
       │
       ▼
Top-3 results displayed
with confidence scores
```

## Tech Stack

| Component | Technology |
|---|---|
| Mobile platform | Android / iOS |
| ML inference | TensorFlow Lite (TFLite) |
| Model | VGG-16 converted to .tflite format |
| Camera | Native camera API |

## Model Conversion

The Keras model was converted to TensorFlow Lite for mobile deployment:

```python
import tensorflow as tf

model = tf.keras.models.load_model("models/vgg16_flower_recognition.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("models/model.tflite", "wb") as f:
    f.write(tflite_model)
```

The `.tflite` model is significantly smaller and faster than the full Keras model, making it suitable for real-time on-device inference.
