"""诊断 HoughLinesP 选出的线段"""
import sys, math
from pathlib import Path
import logging
for _l in [logging.root]: _l.setLevel(logging.CRITICAL + 1)
sys.path.insert(0, str(Path(__file__).parent))
import cv2, numpy as np

DEMO = Path(__file__).parent / "demo_output"
fname = "gauge_200.png"
frame = cv2.imread(str(DEMO / fname))
cx, cy, radius = 320, 240, 200

roi_size = int(radius*1.4)
x1, y1 = max(0,cx-roi_size), max(0,cy-roi_size)
x2, y2 = min(frame.shape[1],cx+roi_size), min(frame.shape[0],cy+roi_size)
roi = frame[y1:y2, x1:x2]
roi_cx, roi_cy = cx-x1, cy-y1

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 5)
edges = cv2.Canny(blurred, 30, 90)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)

print(f"ROI: ({x1},{y1})-({x2},{y2}), roi_cx={roi_cx}, roi_cy={roi_cy}")
print(f"Center in ROI: ({roi_cx},{roi_cy})")
print(f"Found {len(lines)} lines")

best_line_angle = None
best_score = -1
for row in lines:
    xa, ya, xb, yb = int(row[0]), int(row[1]), int(row[2]), int(row[3])
    llen = math.hypot(xb-xa, yb-ya)
    if llen < 15: continue
    d_a = math.hypot(xa-roi_cx, ya-roi_cy)
    d_b = math.hypot(xb-roi_cx, yb-roi_cy)
    near, far = min(d_a,d_b), max(d_a,d_b)
    # My old filter
    ok1 = near < radius*0.35 and far > radius*0.25
    # My new filter
    ok2 = near < radius*0.35 and radius*0.30 < far < radius*0.88
    # Center->far endpoint direction
    if d_a > d_b:
        tip_x, tip_y = xa+x1, ya+y1  # a is far from center
    else:
        tip_x, tip_y = xb+x1, yb+y1
    center_to_tip_angle = math.degrees(math.atan2(tip_y-cy, tip_x-cx)) % 360
    # Line segment direction
    line_dir = math.degrees(math.atan2(yb-ya, xb-xa)) % 360
    if ok1:
        print(f"  near={near:.0f} far={far:.0f} len={llen:.0f}  "
              f"center->tip={center_to_tip_angle:.1f}°  line_dir={line_dir:.1f}°  "
              f"ok1={ok1} ok2={ok2}  pt_a=({xa+x1},{ya+y1})  pt_b=({xb+x1},{yb+y1})")
    if ok1 and far > best_score:
        best_score = far
        best_line_angle = center_to_tip_angle

print(f"\n  best_line_angle (center->tip) = {best_line_angle}")
print(f"  expected: ~179° (gauge_200.png)")

# Also check my strategy with line_dir
best2_score = -1
best2_angle = None
for row in lines:
    xa, ya, xb, yb = int(row[0]), int(row[1]), int(row[2]), int(row[3])
    llen = math.hypot(xb-xa, yb-ya)
    if llen < 15: continue
    d_a = math.hypot(xa-roi_cx, ya-roi_cy)
    d_b = math.hypot(xb-roi_cx, yb-roi_cy)
    near, far = min(d_a,d_b), max(d_a,d_b)
    line_dir = math.degrees(math.atan2(yb-ya, xb-xa)) % 360
    if near < radius*0.35 and radius*0.30 < far < radius*0.88:
        score = far - near
        if score > best2_score:
            best2_score = score
            best2_angle = line_dir

print(f"\n  best2_angle (line_dir) = {best2_angle}")
