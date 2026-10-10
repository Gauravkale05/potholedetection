import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def evaluate_model(
    model_path: str = "models/pothole_mobilenetv2.keras",
    test_dir: str = "dataset/test",
    reports_dir: str = "reports",
    img_size: tuple = (224, 224),
    batch_size: int = 32
):
    print("=" * 60)
    print("STARTING MODEL EVALUATION ON TEST SET")
    print("=" * 60)
    
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found at {test_dir}")
        
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
        
    print(f"Loading trained model from {model_file}...")
    model = tf.keras.models.load_model(model_file)
    
    print(f"Loading test dataset from {test_dir}...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,  # Keep order intact for evaluation
        label_mode='binary'
    )
    
    class_names = test_ds.class_names
    print(f"Class names: {class_names}")
    
    # Extract ground truth labels and predictions
    y_true = []
    y_pred_probs = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().ravel())
        y_pred_probs.extend(preds.ravel())
        
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    # Metrics calculation
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    print("\n--- TEST EVALUATION RESULTS ---")
    print(f"Total Test Samples : {len(y_true)}")
    print(f"Accuracy           : {accuracy * 100:.2f}%")
    print(f"Precision          : {precision * 100:.2f}%")
    print(f"Recall             : {recall * 100:.2f}%")
    print(f"F1-Score           : {f1 * 100:.2f}%")
    print("\nClassification Report:\n", cls_report)
    print("Confusion Matrix:\n", cm)
    
    # Output report dir
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics JSON
    metrics_data = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "total_test_samples": int(len(y_true)),
        "class_mapping": {0: class_names[0], 1: class_names[1]}
    }
    
    with open(rep_path / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    with open(rep_path / "classification_report.txt", "w") as f:
        f.write(f"Road Pothole Detection Model Evaluation Report\n")
        f.write(f"=' * 50\n")
        f.write(f"Accuracy  : {accuracy*100:.2f}%\n")
        f.write(f"Precision : {precision*100:.2f}%\n")
        f.write(f"Recall    : {recall*100:.2f}%\n")
        f.write(f"F1-Score  : {f1*100:.2f}%\n\n")
        f.write(cls_report)
        
    # Plot & Save Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Confusion Matrix - Road Pothole Detection", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Predicted Label", fontsize=12, fontweight='bold')
    plt.ylabel("True Label", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(rep_path / "confusion_matrix.png", dpi=300)
    plt.close()
    
    print(f"\nEvaluation reports saved to: {rep_path.resolve()}")
    return metrics_data

if __name__ == "__main__":
    evaluate_model()

