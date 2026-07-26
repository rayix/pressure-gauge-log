"""诊断: 打印每个 demo 图片的 HoughLines 和最远点结果"""
import sys, os, math
from pathlib import Path
import logging
for _l in [logging.root]: _l.setLevel(logging.CRITICAL + 1)
sys.path.insert(0, str(Path(__file__).parent))
import cv2, numpy as np
from gauge_reader import detect_gauge_center, GAUGE_CENTER, GAUGE_RADIUS

DEMO = Path(__file__).parent / "demo_output"
GAUGE_CENTER_v = (320, 240)
GAUGE_RADIUS_v = 200

def analyze(img_path):
    frame = cv2.imread(str(img_path))
    fname = img_path.name
    # extract expected
    import re
    m = re.search(r'gauge_(\d+)\.png', fname)
    expected = int(m.group(1))/100 if m else None

    center, radius = detect_gauge_center(frame)
    cx, cy = center
    roi_size = int(radius * 1.4)
    x1, y1 = max(0, cx - roi_size), max(0, cy - roi_size)
    x2, y2 = min(frame.shape[1], cx + roi_size), min(frame.shape[0], cy + roi_size)
    roi = frame[y1:y2, x1:x2]
    roi_cx, roi_cy = cx - x1, cy - y1

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blurred, 30, 90)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)

    print(f"\n=== {fname} (expected={expected}) ===")
    if lines is None:
        print("  No lines found")
        return

    # Show all lines with their distance to center
    candidates = []
    for row in lines:
        xa, ya, xb, yb = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        line_len = math.hypot(xb-xa, yb-ya)
        if line_len < 15: continue
        d_a = math.hypot(xa-roi_cx, ya-roi_cy)
        d_b = math.hypot(xb-roi_cx, yb-roi_cy)
        near, far = min(d_a,d_b), max(d_a,d_b)
        # angle from center to far point
        far_pt = (xa,ya) if d_a > d_b else (xb,yb)
        angle = math.degrees(math.atan2(far_pt[1], far_pt[0])) % 360
        candidates.append((far, near, line_len, far_pt, angle))

    # Sort by far distance desc
    candidates.sort(key=lambda x: -x[0])
    for far, near, llen, (px,py), ang in candidates[:5]:
        print(f"  near={near:.0f} far={far:.0f} len={llen:.0f} "
              f"angle={ang:.1f} deg  endpoint=({px+x1:.0f},{py+y1:.0f})")

    # Strategy 2: binary far point
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    center_mask = np.zeros_like(binary)
    cv2.circle(center_mask, (roi_cx, roi_cy), int(radius*0.85), 1, -1)
    cv2.circle(center_mask, (roi_cx, roi_cy), int(radius*0.15), 0, -1)
    binary = cv2.bitwise_and(binary, binary, mask=center_mask)
    try:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_pt, best_dist = None, 0
    for cnt in contours:
        pts = cnt.flatten()
        for i in range(0, len(pts)-1, 2):
            px, py = pts[i], pts[i+1]
            dist = math.hypot(px-roi_cx, py-roi_cy)
            if radius*0.2 < dist < radius*0.9 and dist > best_dist:
                best_dist = dist
                best_pt = (px+x1, py+y1)
    if best_pt:
        angle2 = math.degrees(math.atan2(best_pt[1]-cy, best_pt[0]-cx)) % 360
        print(f"  Strategy2: best_pt={best_pt} dist={best_dist:.0f} angle={angle2:.1f} deg")

for p in sorted((DEMO).glob("gauge_*.png")):
    analyze(p)
