"""Precise radial scan with gradient verification"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import cv2, numpy as np

DEMO = Path(__file__).parent / "demo_output"
cx, cy, radius = 320, 240, 200

for fname in ["gauge_135.png", "gauge_200.png"]:
    img = cv2.imread(str(DEMO / fname))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"\n=== {fname} ===")

    # Expected needle angles: 202°, 191°
    for test_angle in [126, 202]:
        rad = math.radians(test_angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        print(f"\n  Scanning angle {test_angle}° (1px steps from 0.50r to 0.95r):")
        min_val, min_dist = 255, 0
        for r_pix in range(int(radius*0.50), int(radius*0.96), 1):
            px = int(cx + cos_a * r_pix)
            py = int(cy + sin_a * r_pix)
            if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                v = int(gray[py, px])
                if v < min_val:
                    min_val = v
                    min_dist = r_pix
                # Only print interesting values
                if v < 220:
                    print(f"    r={r_pix}({r_pix/radius:.2f}r): gray={v}")
        print(f"  Min: dist={min_dist} val={min_val}")

        # Gradient scan at 1px resolution
        print(f"  Gradients (step=3px, ±15% of min_dist):")
        scan_min = max(int(radius*0.45), int(min_dist - radius*0.15))
        scan_max = min(int(radius*0.95), int(min_dist + radius*0.10))
        grad_peak, grad_best_r = 0, 0
        for r_pix in range(scan_min, scan_max + 1, 1):
            px1 = int(cx + cos_a * r_pix)
            py1 = int(cy + sin_a * r_pix)
            px2 = int(cx + cos_a * (r_pix + 3))
            py2 = int(cy + sin_a * (r_pix + 3))
            if (0 <= px1 < gray.shape[1] and 0 <= py1 < gray.shape[0] and
                0 <= px2 < gray.shape[1] and 0 <= py2 < gray.shape[0]):
                grad = (int(gray[py2, px2]) - int(gray[py1, px1])) / 3.0
                if abs(grad) > 5:
                    print(f"    r={r_pix}: ({int(gray[py1,px1])},{int(gray[py2,px2])}) grad={grad:.1f}")
                if grad > grad_peak:
                    grad_peak = grad
                    grad_best_r = r_pix
        print(f"  Positive grad peak: {grad_peak:.1f} at r={grad_best_r}")

        # Also try the 202° direction: needle should have min at 0.85-0.90r
        # Let's scan the ENTIRE angle range with 1° steps and 1px resolution
