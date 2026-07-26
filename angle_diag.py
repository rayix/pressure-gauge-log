"""Map: expected_angle vs detected_angle for all demo images"""
import sys, math
from pathlib import Path
import logging
for _l in [logging.root]: _l.setLevel(logging.CRITICAL + 1)
sys.path.insert(0, str(Path(__file__).parent))
import cv2, numpy as np

DEMO = Path(__file__).parent / "demo_output"

# Expected angle per image: computed from expected pressure
ANGLE_0_DEG = 225.0
SWEEP_RANGE = 270.0
SCALE_MAX = 16.0

def expected_angle(pressure):
    """Given pressure, what's the expected image angle?"""
    ratio = pressure / SCALE_MAX
    offset = ratio * SWEEP_RANGE  # offset in degrees (clockwise from 0-point)
    # angle = (ANGLE_0_DEG - offset) % 360
    return (ANGLE_0_DEG - offset) % 360

def angle_to_pressure(angle_deg):
    offset = (ANGLE_0_DEG - angle_deg) % 360
    if offset > SWEEP_RANGE:
        offset -= 360
    return max(0.0, min(SWEEP_RANGE, offset)) / SWEEP_RANGE * SCALE_MAX

cx, cy, radius = 320, 240, 200
roi_size = int(radius * 1.4)
x1_ = lambda cx=cx: max(0, cx - roi_size)
y1_ = lambda cy=cy: max(0, cy - roi_size)

def get_angles(img_path):
    frame = cv2.imread(str(img_path))
    fname = img_path.name
    import re
    m = re.search(r'gauge_(\d+)\.png', fname)
    exp_pressure = int(m.group(1))/100 if m else None
    exp_angle = expected_angle(exp_pressure) if exp_pressure else None

    x1, y1 = x1_(), y1_()
    x2 = min(frame.shape[1], cx + roi_size)
    y2 = min(frame.shape[0], cy + roi_size)
    roi = frame[y1:y2, x1:x2]
    roi_cx, roi_cy = cx - x1, cy - y1
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    # Strategy 2 radial: collect best angle per threshold
    best_angle, best_score = None, -1
    for t in [40, 60, 80, 100]:
        _, binary = cv2.threshold(blurred, t, 255, cv2.THRESH_BINARY_INV)
        for angle_deg in range(0, 360, 3):
            rad = math.radians(angle_deg)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            min_val, min_dist = 255, 0
            for r_pix in range(int(radius*0.50), int(radius*0.96), 2):
                px = int(roi_cx + cos_a * r_pix)
                py = int(roi_cy + sin_a * r_pix)
                if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                    v = int(gray[py, px])
                    if v < min_val:
                        min_val = v
                        min_dist = r_pix
            if min_val < 180 and min_dist >= radius * 0.45:
                quality = (255 - min_val) * min_dist / radius
                if quality > best_score:
                    best_score = quality
                    best_angle = angle_deg

    return exp_pressure, exp_angle, best_angle

results = []
for p in sorted((DEMO).glob("gauge_*.png")):
    exp_p, exp_a, det_a = get_angles(p)
    results.append((exp_p, exp_a, det_a))
    if exp_a and det_a:
        err = det_a - exp_a
        if err > 180: err -= 360
        if err < -180: err += 360
        computed_p = angle_to_pressure(det_a)
        print(f"{p.name}: exp_angle={exp_a:.1f}° det_angle={det_a:.1f}° "
              f"err={err:.1f}° exp_pressure={exp_p} computed={computed_p:.2f}")
    elif det_a is None:
        print(f"{p.name}: exp_angle={exp_a:.1f}° det_angle=None")
    else:
        print(f"{p.name}: exp_angle={exp_a:.1f}° det_angle={det_a:.1f}°")
