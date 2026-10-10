import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from src.predict import preprocess_image

def generate_gradcam_overlay(image_input, model, alpha: float = 0.4, colormap_name: str = "jet"):
    """
    Generates a Grad-CAM visual explanation overlay on top of the input road image.
    
    Returns:
        overlay_pil (PIL.Image): Blended image with heatmap overlay
        heatmap_norm (np.ndarray): Normalized 2D heatmap array (224, 224)
    """
    if isinstance(image_input, (str, Path)):
        original_img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        original_img = image_input
    else:
        raise ValueError("image_input must be a file path or PIL.Image")

    if original_img.mode != "RGB":
        original_img = original_img.convert("RGB")
        
    img_array = preprocess_image(original_img)
    
    # Extract base mobilenet and head layers
    base_mobilenet = model.get_layer("mobilenetv2_1.00_224")
    
    # Input image to base mobilenet
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = model.get_layer("data_augmentation")(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Base feature extractor
    feature_extractor = tf.keras.Model(
        inputs=base_mobilenet.input,
        outputs=base_mobilenet.get_layer("out_relu").output
    )
    conv_features = feature_extractor(x)

    # Build head submodel from GAP to output_pothole
    head_input = tf.keras.Input(shape=conv_features.shape[1:])
    h = model.get_layer("global_average_pooling2d")(head_input)
    h = model.get_layer("batch_normalization")(h)
    h = model.get_layer("dense")(h)
    h = model.get_layer("dropout")(h)
    preds = model.get_layer("output_pothole")(h)
    head_model = tf.keras.Model(inputs=head_input, outputs=preds)

    with tf.GradientTape() as tape:
        conv_out = feature_extractor(img_array)
        tape.watch(conv_out)
        predictions = head_model(conv_out)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        heatmap = tf.zeros_like(heatmap)
        
    heatmap_np = heatmap.numpy()
    
    # Resize heatmap to target size (224, 224)
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap_np)).resize((224, 224), Image.Resampling.BILINEAR)
    heatmap_norm = np.array(heatmap_pil, dtype=np.float32) / 255.0

    # Apply colormap
    cmap = plt.get_cmap(colormap_name)
    colored_heatmap = (cmap(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(colored_heatmap)

    # Blend with original image resized to (224, 224)
    resized_orig = original_img.resize((224, 224), Image.Resampling.BILINEAR)
    overlay_pil = Image.blend(resized_orig, heatmap_img, alpha=alpha)

    return overlay_pil, heatmap_norm

