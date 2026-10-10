import os
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

_MODEL_CACHE = {}

def load_pothole_model(model_path: str = "models/pothole_mobilenetv2.keras"):
    """
    Loads and caches the trained Keras model.
    """
    global _MODEL_CACHE
    model_path_str = str(Path(model_path).resolve())
    if model_path_str not in _MODEL_CACHE:
        if not Path(model_path_str).exists():
            raise FileNotFoundError(f"Model file missing at {model_path_str}")
        _MODEL_CACHE[model_path_str] = tf.keras.models.load_model(model_path_str)
    return _MODEL_CACHE[model_path_str]

def preprocess_image(image: Image.Image, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocesses a PIL image for model input.
    The MobileNetV2 model architecture includes mobilenet_v2.preprocess_input layer,
    so input array should be float32 in range [0.0, 255.0] with shape (1, 224, 224, 3).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(image, dtype=np.float32)
    # Expand dims to batch shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_input, model=None, model_path: str = "models/pothole_mobilenetv2.keras"):
    """
    Predicts whether a given image contains a pothole or normal road.
    
    Returns a dict with:
        - label: "Pothole Detected" or "Normal Road"
        - class_name: "Pothole" or "Normal"
        - confidence: float (0.0 to 100.0)
        - raw_score: float (0.0 to 1.0)
    """
    if model is None:
        model = load_pothole_model(model_path)
        
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise ValueError("image_input must be a file path or PIL Image object.")
        
    processed_img = preprocess_image(pil_img)
    pred = model.predict(processed_img, verbose=0)[0][0]  # Sigmoid output (0: Normal, 1: Pothole)
    
    pothole_prob = float(pred)
    
    if pothole_prob >= 0.5:
        class_name = "Pothole"
        label = "Pothole Detected"
        confidence = pothole_prob * 100.0
    else:
        class_name = "Normal"
        label = "Normal Road"
        confidence = (1.0 - pothole_prob) * 100.0
        
    return {
        "label": label,
        "class_name": class_name,
        "confidence": round(confidence, 2),
        "raw_score": round(pothole_prob, 4)
    }

if __name__ == "__main__":
    # Test script if model exists
    sample_pothole = Path("samples/sample_pothole.jpg")
    sample_normal = Path("samples/sample_normal.jpg")
    
    if sample_pothole.exists():
        res_p = predict_image(sample_pothole)
        print("Pothole Sample Prediction:", res_p)
    if sample_normal.exists():
        res_n = predict_image(sample_normal)
        print("Normal Sample Prediction:", res_n)

