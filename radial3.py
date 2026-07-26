"""Precise radial scan - test with 1° resolution"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import cv2, numpy as np

DEMO = Path(__file__).parent / "demo_output"
cx, cy, radius = 320, 240, 200

def full_scan(img_path):
    gray = cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY) if hasattr(img_path, 'shape') else cv2.imread(str(img_path))
    if not hasattr(gray, 'shape'):
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    img = img_path if hasattr(img_path, 'shape') else cv2.imread(str(img_path))

    # 1) 1° radial scan: find dark pixels
    dark_candidates = []  # (angle, dark_val, dark_dist)
    for angle_deg in range(0, 360, 1):
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        min_val, min_dist = 255, 0
        for r_pix in range(int(radius*0.45), int(radius*0.96), 1):
            px = int(cx + cos_a * r_pix)
            py = int(cy + sin_a * r_pix)
            if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                v = int(gray[py, px])
                if v < min_val:
                    min_val = v
                    min_dist = r_pix
        if min_val < 200:
            dark_candidates.append((angle_deg, min_val, min_dist))

    # 2) For each dark candidate, compute gradient and range
    scored = []
    for angle_deg, min_val, min_dist in dark_candidates:
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        # Range of gray values from 0.3r to min_dist+0.1r
        vals = []
        for r_pix in range(int(radius*0.30), min(int(min_dist + radius*0.12), int(radius*0.95)) + 1, 2):
            px = int(cx + cos_a * r_pix)
            py = int(cy + sin_a * r_pix)
            if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                vals.append(int(gray[py, px]))
        val_range = max(vals) - min(vals) if vals else 0

        # Radial gradient: at min_dist (dark→bright = needle tip)
        grad_peak = 0
        for r_pix in range(max(0, int(min_dist-radius*0.15)), int(min_dist+radius*0.12)+1, 1):
            px1 = int(cx + cos_a * r_pix)
            py1 = int(cy + sin_a * r_pix)
            px2 = int(cx + cos_a * (r_pix + 4))
            py2 = int(cy + sin_a * (r_pix + 4))
            if (0 <= px1 < gray.shape[1] and 0 <= py1 < gray.shape[0] and
                0 <= px2 < gray.shape[1] and 0 <= py2 < gray.shape[0]):
                grad = (int(gray[py2, px2]) - int(gray[py1, px1])) / 4.0
                if grad > grad_peak:
                    grad_peak = grad

        # Score: combine gradient, range, and distance
        # Range matters: hub shadow has small range, needle has large range
        dist_weight = min_dist / (radius * 0.95)
        score = (grad_peak + val_range / 20) * dist_weight  # val_range/20 normalizes it
        scored.append((score, angle_deg, min_val, min_dist, grad_peak, val_range))

    scored.sort(key=lambda x: -x[0])
    return scored

for fname in ["gauge_135.png", "gauge_200.png", "gauge_330.png"]:
    img = cv2.imread(str(DEMO / fname))
    import re
    exp_p = int(re.search(r'gauge_(\d+)', fname).group(1))/100
    print(f"\n=== {fname} (exp={exp_p}) ===")
    results = full_scan(img)
    print(f"  Top 10 candidates:")
    for score, ang, val, dist, grad, rng in results[:10]:
        print(f"    angle={ang:3d}° dark_val={val} dist={dist}({dist/radius:.2f}r) "
              f"grad={grad:.1f} range={rng} score={score:.1f}")
