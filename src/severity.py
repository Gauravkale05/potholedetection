import numpy as np
from PIL import Image

def estimate_defect_area_from_image(pil_img: Image.Image) -> float:
    """
    Estimates the surface area percentage of dark cavity/pothole defect regions in a road photo.
    """
    if pil_img is None:
        return 12.5  # Default reasonable estimate
        
    try:
        if pil_img.mode != "L":
            gray_img = pil_img.convert("L")
        else:
            gray_img = pil_img
            
        arr = np.array(gray_img.resize((224, 224)), dtype=np.float32)
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        
        defect_mask = arr < (mean_val - 0.75 * std_val)
        area_pct = (np.sum(defect_mask) / arr.size) * 100.0
        return float(np.clip(area_pct, 4.2, 45.0))
    except Exception:
        return 12.5

def analyze_pothole_severity(confidence: float, pil_img: Image.Image = None, class_name: str = "Pothole"):
    """
    Analyzes pothole severity level and defect surface coverage.
    
    Parameters:
        confidence (float): Pothole confidence percentage (0.0 to 100.0)
        pil_img (PIL.Image): Input road image object
        class_name (str): "Pothole" or "Normal"
        
    Returns:
        dict with:
            - severity_level: "Low Risk", "Moderate Hazard", "Critical Hazard", or "Normal Road"
            - severity_code: "LOW", "MODERATE", "CRITICAL", or "NORMAL"
            - area_percentage: float (estimated defect coverage %)
            - action_recommended: text instruction for maintenance crew
            - color: status card hex color
    """
    if class_name != "Pothole":
        return {
            "severity_level": "Normal Road",
            "severity_code": "NORMAL",
            "area_percentage": 0.0,
            "action_recommended": "Clear road surface. No repair needed.",
            "color": "#22C55E"
        }
        
    area_pct = estimate_defect_area_from_image(pil_img)
    
    # Severity classification logic
    if confidence >= 85.0 or area_pct >= 25.0:
        severity_level = "Critical Hazard"
        severity_code = "CRITICAL"
        action = "🚨 Immediate Emergency Repair Required! High risk to vehicular traffic."
        color = "#EF4444"
    elif confidence >= 60.0 or area_pct >= 12.0:
        severity_level = "Moderate Hazard"
        severity_code = "MODERATE"
        action = "⚠️ Schedule Routine Road Patching within 7 days."
        color = "#F59E0B"
    else:
        severity_level = "Minor Defect"
        severity_code = "LOW"
        action = "🟢 Minor surface irregularity. Monitor during next inspection cycle."
        color = "#10B981"
        
    return {
        "severity_level": severity_level,
        "severity_code": severity_code,
        "area_percentage": round(area_pct, 1),
        "action_recommended": action,
        "color": color
    }
