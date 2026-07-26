"""Radial scanning strategy: find needle by radial profile"""
import sys, math
from pathlib import Path
import logging
for _l in [logging.root]: _l.setLevel(logging.CRITICAL + 1)
sys.path.insert(0, str(Path(__file__).parent))
import cv2, numpy as np

DEMO = Path(__file__).parent / "demo_output"
fname = "gauge_135.png"
frame = cv2.imread(str(DEMO / fname))
cx, cy, radius = 320, 240, 200

roi_size = int(radius*1.4)
x1, y1 = max(0,cx-roi_size), max(0,cy-roi_size)
x2, y2 = min(frame.shape[1],cx+roi_size), min(frame.shape[0],cy+roi_size)
roi = frame[y1:y2, x1:x2]
roi_cx, roi_cy = cx-x1, cy-y1

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# Simple: just scan at fixed angles
print(f"ROI: ({x1},{y1})-({x2},{y2}), center=({cx},{cy}), roi_center=({roi_cx},{roi_cy})")

# Radial scan at each angle, find darkest pixel distance
results = []
for angle_deg in range(0, 360, 5):
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    # Scan from center to radius*1.1
    min_val = 255
    min_dist = 0
    for t in range(0, int(radius*1.1), 2):
        px = int(roi_cx + dx * t)
        py = int(roi_cy + dy * t)
        if 0 <= px < roi.shape[1] and 0 <= py < roi.shape[0]:
            val = int(gray[py, px])
            if val < min_val:
                min_val = val
                min_dist = t
    if min_val < 200:  # only if there's a dark pixel
        results.append((angle_deg, min_dist, min_val))

results.sort(key=lambda x: x[1])
print("\nTop 15 darkest angles (sorted by distance from center):")
for ang, dist, val in results[:15]:
    print(f"  angle={ang:3d} deg  dist={dist:.0f}  min_val={val}")

# New strategy: radial gradient analysis
# For each angle, look at the gradient in the radial direction
print("\n--- Radial gradient analysis ---")
grad_results = []
for angle_deg in range(0, 360, 3):
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    max_grad = 0
    max_grad_dist = 0
    for t in range(10, int(radius*0.9), 2):
        px1 = int(roi_cx + dx * (t - 2))
        py1 = int(roi_cy + dy * (t - 2))
        px2 = int(roi_cx + dx * (t + 2))
        py2 = int(roi_cy + dy * (t + 2))
        if (0 <= px1 < roi.shape[1] and 0 <= py1 < roi.shape[0] and
            0 <= px2 < roi.shape[1] and 0 <= py2 < roi.shape[0]):
            grad = abs(int(gray[py2, px2]) - int(gray[py1, px1])) / 4.0
            if grad > max_grad:
                max_grad = grad
                max_grad_dist = t
    # needle is a dark-to-bright transition = positive gradient
    if max_grad > 10:
        grad_results.append((angle_deg, max_grad_dist, max_grad))

grad_results.sort(key=lambda x: -x[1])
print("Top 15 gradient peaks:")
for ang, dist, grad in grad_results[:15]:
    print(f"  angle={ang:3d} deg  dist={dist:.0f}  grad={grad:.1f}")

# Combined: find angle with concentrated high gradient at a specific distance
print("\n--- Histogram of gradient peak distances ---")
# Bin distances into ranges
from collections import Counter
dist_bins = Counter()
for ang, dist, grad in grad_results:
    if dist < 200:  # within gauge
        bin_label = (dist // 20) * 20
        dist_bins[bin_label] += grad
for d in sorted(dist_bins.keys()):
    print(f"  dist {d:3d}-{d+19}: score={dist_bins[d]:.0f}")
