"""诊断+可视化: 保存中间图像帮助调试指针检测"""
import sys, os, math
from pathlib import Path
import logging
for _l in [logging.root]: _l.setLevel(logging.CRITICAL + 1)
sys.path.insert(0, str(Path(__file__).parent))
import cv2, numpy as np
from gauge_reader import detect_gauge_center, GAUGE_CENTER, GAUGE_RADIUS

DEMO = Path(__file__).parent / "demo_output"
DEBUG = Path(__file__).parent / "debug_imgs"
DEBUG.mkdir(exist_ok=True)

# 分析不同 Canny/blur 参数组合
def analyze_params(img_path):
    frame = cv2.imread(str(img_path))
    fname = img_path.name
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
    vis = roi.copy()

    # 试不同 blur+参数
    combos = [
        ("blur3_canny20-60", 3, 20, 60),
        ("blur3_canny10-40", 3, 10, 40),
        ("blur3_canny5-30",  3,  5, 30),
        ("noblur_canny20-60", 0, 20, 60),
        ("noblur_canny5-30",  0,  5, 30),
        ("thresh100",         0, 100, 200),  # fixed threshold
        ("thresh80",          0,  80, 160),
    ]

    print(f"\n=== {fname} (expected={expected}) ===")
    best_all = None
    best_all_score = -1

    for name, blur_k, c1, c2 in combos:
        blurred = cv2.medianBlur(gray, blur_k) if blur_k else gray.copy()
        if name.startswith("thresh"):
            _, binary = cv2.threshold(blurred, int(name[6:]), 255, cv2.THRESH_BINARY_INV)
            edges = binary
        else:
            edges = cv2.Canny(blurred, c1, c2)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)

        candidates = []
        if lines is not None:
            for row in lines:
                xa, ya, xb, yb = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                llen = math.hypot(xb-xa, yb-ya)
                if llen < 15: continue
                d_a = math.hypot(xa-roi_cx, ya-roi_cy)
                d_b = math.hypot(xb-roi_cx, yb-roi_cy)
                near, far = min(d_a,d_b), max(d_a,d_b)
                if near < radius*0.35 and far > radius*0.25 and far > radius*0.55:
                    # good: one end near center, one end out (not rim)
                    far_pt = (xa,ya) if d_a > d_b else (xb,yb)
                    angle = math.degrees(math.atan2(far_pt[1]+y1-cy, far_pt[0]+x1-cx)) % 360
                    candidates.append((far, near, llen, far_pt, angle))

        candidates.sort(key=lambda x: -x[0])
        if candidates:
            far_d, near_d, llen, (px, py), ang = candidates[0]
            # Exclude bottom of rim (y near cy+radius)
            is_rim = abs((py+y1) - cy - radius) < 20 and far_d > radius*0.9
            if not is_rim:
                score = far_d - near_d  # long and near-center
                if score > best_all_score:
                    best_all_score = score
                    best_all = (name, ang, far_d, near_d, llen, px, py)
                print(f"  [{name}] far={far_d:.0f} near={near_d:.0f} "
                      f"angle={ang:.1f} deg  pt=({px+x1:.0f},{py+y1:.0f})")
            else:
                print(f"  [{name}] rim-filtered: far={far_d:.0f} near={near_d:.0f}")

    if best_all:
        print(f"  BEST: [{best_all[0]}] angle={best_all[1]:.1f} deg")

        # Save debug visualization
        name, ang, far_d, near_d, llen, px, py = best_all
        blurred = cv2.medianBlur(gray, 3) if "blur3" in name else gray.copy()
        if name.startswith("thresh"):
            _, binary = cv2.threshold(blurred, int(name[6:]), 255, cv2.THRESH_BINARY_INV)
            edges_disp = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            c1, c2 = (20,60) if "20-60" in name else (10,40) if "10-40" in name else (5,30)
            edges_disp = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Draw center
        cv2.circle(edges_disp, (roi_cx, roi_cy), 5, (0,255,0), 2)
        # Draw detected needle
        tip_img = (px+x1-cx, py+y1-cy)
        cv2.line(edges_disp, (roi_cx, roi_cy),
                 (int(px), int(py)), (0,0,255), 2)
        cv2.imwrite(str(DEBUG / f"{fname.replace('.png','')}_edges.png"), edges_disp)
    else:
        print(f"  NO CANDIDATE found for any param combo")
        # Save gray + binary
        blurred = cv2.medianBlur(gray, 3)
        _, binary = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
        edges_disp = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.circle(edges_disp, (roi_cx, roi_cy), 5, (0,255,0), 2)
        cv2.imwrite(str(DEBUG / f"{fname.replace('.png','')}_edges.png"), edges_disp)


for p in sorted((DEMO).glob("gauge_*.png"))[:5]:
    analyze_params(p)
print(f"\nDebug images saved to {DEBUG}")
