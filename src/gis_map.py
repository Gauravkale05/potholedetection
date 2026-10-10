import numpy as np
import pandas as pd

CITY_PRESETS = {
    "Mumbai, India": (19.0760, 72.8777),
    "Delhi NCR, India": (28.6139, 77.2090),
    "Bengaluru, India": (12.9716, 77.5946),
    "New York, USA": (40.7128, -74.0060),
    "London, UK": (51.5074, -0.1278),
    "San Francisco, USA": (37.7749, -122.4194),
    "Tokyo, Japan": (35.6762, 139.6503),
    "Sydney, Australia": (-33.8688, 151.2093)
}

def get_sample_pothole_locations(center_lat: float = 19.0760, center_lon: float = 72.8777, count: int = 20, seed: int = 42):
    """
    Generates realistic GPS coordinates for reported road pothole incidents centered at specified coordinates.
    """
    np.random.seed(seed)
    lats = center_lat + np.random.normal(0, 0.015, count)
    lons = center_lon + np.random.normal(0, 0.015, count)
    
    severities = np.random.choice(["Critical Hazard", "Moderate Hazard", "Minor Defect"], size=count, p=[0.35, 0.40, 0.25])
    confidences = np.random.uniform(72.0, 99.8, count).round(1)
    
    df = pd.DataFrame({
        "lat": lats,
        "lon": lons,
        "Severity": severities,
        "Confidence (%)": confidences,
        "Incident ID": [f"PTH-{i+101:03d}" for i in range(count)]
    })
    return df
