"""Read actual grayscale pixel values at needle tip and hub positions"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import cv2, numpy as np

DEMO = Path(__file__).parent / "demo_output"
cx, cy, radius = 320, 240, 200
roi_cx, roi_cy = 280, 240  # after ROI crop

for fname in ["gauge_135.png", "gauge_200.png", "gauge_330.png"]:
    img = cv2.imread(str(DEMO / fname))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"\n=== {fname} ===")

    # Expected needle angles: 202°, 191°, 169°
    # Hub angle: ~205°
    exp_angles = {"gauge_135.png": 202, "gauge_200.png": 191, "gauge_330.png": 169}

    # Scan all angles at multiple radii
    for test_angle in [66, 126, 144, 169, 191, 202, 205]:
        rad = math.radians(test_angle)
        print(f"  Angle {test_angle}°:", end="")
        for r_pct in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90]:
            r_pix = int(radius * r_pct)
            px = int(cx + math.cos(rad) * r_pix)
            py = int(cy + math.sin(rad) * r_pix)
            if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                v = int(gray[py, px])
                print(f" {r_pct:.2f}r={v}", end="")
        print()

    # Show hub center pixel
    print(f"  Hub center ({cx},{cy}): {int(gray[cy,cx])}")
    # Show nearby pixels
    for dy in [-5, -3, -1, 0, 1, 3, 5]:
        for dx in [-5, -3, -1, 0, 1, 3, 5]:
            px, py = cx+dx, cy+dy
            if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                v = int(gray[py, px])
                if v < 200:
                    print(f"    hub+({dx:+d},{dy:+d}): {v}")
    print()

    # Histogram of gray values
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    print(f"  Histogram (brightest peaks):")
    peaks = [(i, int(hist[i])) for i in range(256) if hist[i] > 0]
    peaks.sort(key=lambda x: -x[1])
    for i, count in peaks[:8]:
        print(f"    gray={i}: {count} pixels")
