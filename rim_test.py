"""Test: rim vs needle at 0.95r brightness"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import cv2, numpy as np

DEMO = Path(__file__).parent / "demo_output"
cx, cy, radius = 320, 240, 200

for fname in ["gauge_135.png", "gauge_200.png", "gauge_330.png"]:
    img = cv2.imread(str(DEMO / fname))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"\n=== {fname} ===")
    
    # Test the rim vs needle angles: check brightness at 0.95r
    for angle_deg in [66, 126, 127, 153, 176, 177, 179, 181, 182, 183]:
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        
        # Brightness at 0.95r
        r_095 = int(radius * 0.95)
        px_095 = int(cx + cos_a * r_095)
        py_095 = int(cy + sin_a * r_095)
        val_095 = int(gray[py_095, px_095]) if (0 <= px_095 < gray.shape[1] and 0 <= py_095 < gray.shape[0]) else 255
        
        # Brightness at 0.85r
        r_085 = int(radius * 0.85)
        px_085 = int(cx + cos_a * r_085)
        py_085 = int(cy + sin_a * r_085)
        val_085 = int(gray[py_085, px_085]) if (0 <= px_085 < gray.shape[1] and 0 <= py_085 < gray.shape[0]) else 255
        
        # Brightness at 0.75r
        r_075 = int(radius * 0.75)
        px_075 = int(cx + cos_a * r_075)
        py_075 = int(cy + sin_a * r_075)
        val_075 = int(gray[py_075, px_075]) if (0 <= px_075 < gray.shape[1] and 0 <= py_075 < gray.shape[0]) else 255
        
        # Brightness at 0.65r
        r_065 = int(radius * 0.65)
        px_065 = int(cx + cos_a * r_065)
        py_065 = int(cy + sin_a * r_065)
        val_065 = int(gray[py_065, px_065]) if (0 <= px_065 < gray.shape[1] and 0 <= py_065 < gray.shape[0]) else 255
        
        # Brightness at 0.55r
        r_055 = int(radius * 0.55)
        px_055 = int(cx + cos_a * r_055)
        py_055 = int(cy + sin_a * r_055)
        val_055 = int(gray[py_055, px_055]) if (0 <= px_055 < gray.shape[1] and 0 <= py_055 < gray.shape[0]) else 255
        
        # Rim feature: 0.95r is also dark (rim is at edge, background = transparent = dark)
        # Needle feature: 0.95r is bright (needle is inside gauge, beyond it = bright background)
        is_rim = val_095 < 100
        is_needle = val_095 > 200
        flag = "NEEDLE" if is_needle else ("RIM" if is_rim else "?")
        
        print(f"  angle={angle_deg:3d}°: 0.55r={val_055} 0.65r={val_065} "
              f"0.75r={val_075} 0.85r={val_085} 0.95r={val_095} [{flag}]")
