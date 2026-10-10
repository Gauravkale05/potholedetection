import os
import datetime
import pandas as pd
from PIL import Image
from pathlib import Path

from src.predict import predict_image
from src.severity import analyze_pothole_severity

def process_image_batch(uploaded_files, model):
    """
    Processes a list of uploaded images (or file paths) and returns a summary DataFrame.
    """
    results = []
    
    for idx, file_obj in enumerate(uploaded_files):
        try:
            filename = getattr(file_obj, "name", f"image_{idx+1}.jpg")
            pil_img = Image.open(file_obj)
            
            # Predict
            pred_res = predict_image(pil_img, model=model)
            
            # Analyze Severity & Defect Coverage
            sev_res = analyze_pothole_severity(
                confidence=pred_res["confidence"],
                pil_img=pil_img,
                class_name=pred_res["class_name"]
            )

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            results.append({
                "Inspection ID": f"INC-{idx+1:04d}",
                "Filename": filename,
                "Classification": pred_res["label"],
                "Confidence (%)": pred_res["confidence"],
                "Severity Rating": sev_res["severity_level"],
                "Estimated Defect Area (%)": sev_res["area_percentage"],
                "Recommended Action": sev_res["action_recommended"],
                "Timestamp": timestamp
            })
        except Exception as e:
            results.append({
                "Inspection ID": f"INC-{idx+1:04d}",
                "Filename": getattr(file_obj, "name", f"image_{idx+1}.jpg"),
                "Classification": "Processing Error",
                "Confidence (%)": 0.0,
                "Severity Rating": "N/A",
                "Estimated Defect Area (%)": 0.0,
                "Recommended Action": f"Error: {str(e)}",
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
    df = pd.DataFrame(results)
    return df
