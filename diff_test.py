"""Test: brightness drop (0.75r→0.85r) as needle detector"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import cv2, numpy as np

DEMO = Path(__file__).parent / "demo_output"
cx, cy, radius = 320, 240, 200

def get_bright_at_r(gray, angle_deg, r_pct):
    r = int(radius * r_pct)
    px = int(cx + math.cos(math.radians(angle_deg)) * r)
    py = int(cy + math.sin(math.radians(angle_deg)) * r)
    if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
        return int(gray[py, px])
    return 255

print("gauge_135.png (exp_angle=202°, pressure=1.35)")
print("gauge_200.png (exp_angle=191°, pressure=2.00)")
print("gauge_330.png (exp_angle=169°, pressure=3.30)")
print()

for fname in ["gauge_135.png", "gauge_200.png", "gauge_330.png"]:
    img = cv2.imread(str(DEMO / fname))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute 0.75r→0.85r drop for all angles
    drops = []
    for a in range(0, 360, 2):
        v075 = get_bright_at_r(gray, a, 0.75)
        v085 = get_bright_at_r(gray, a, 0.85)
        v090 = get_bright_at_r(gray, a, 0.90)
        drop = v075 - v085  # big drop = needle
        rise = v085 - v090  # rise = needle tip at 0.85r, bright beyond
        
        # Needle: drop >= 100, and v090 > v085 (tip then bright beyond)
        # OR: v085 < 50 (very dark at 0.85r)
        drops.append((drop, rise, a, v075, v085, v090))
    
    # Sort by drop magnitude
    drops.sort(key=lambda x: -x[0])
    
    # Top candidates
    print(f"=== {fname} ===")
    print(f"  Top drop candidates (0.75r→0.85r):")
    for drop, rise, a, v075, v085, v090 in drops[:12]:
        flag = "NEEDLE!" if (drop >= 100 and v085 < 150) else "?"
        print(f"    angle={a:3d}°: 0.75r={v075} 0.85r={v085} 0.90r={v090} "
              f"drop={drop:3d} rise={rise:3d} [{flag}]")
    print()
