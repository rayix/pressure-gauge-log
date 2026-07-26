"""Debug Strategy 2: understand why it gives wrong angle for low-pressure images"""
import sys
from pathlib import Path
import logging
for _l in [logging.root]: _l.setLevel(logging.CRITICAL + 1)
sys.path.insert(0, str(Path(__file__).parent))
import cv2, numpy as np, math

DEMO = Path(__file__).parent / "demo_output"
DEBUG = Path(__file__).parent / "debug_imgs"
DEBUG.mkdir(exist_ok=True)

def debug_s2(img_path):
    frame = cv2.imread(str(img_path))
    fname = img_path.name
    import re
    m = re.search(r'gauge_(\d+)\.png', fname)
    expected = int(m.group(1))/100 if m else None

    cx, cy = 320, 240
    radius = 200
    roi_size = int(radius * 1.4)
    x1, y1 = max(0, cx - roi_size), max(0, cy - roi_size)
    x2, y2 = min(frame.shape[1], cx + roi_size), min(frame.shape[0], cy + roi_size)
    roi = frame[y1:y2, x1:x2]
    roi_cx, roi_cy = cx - x1, cy - y1

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Show Otsu threshold value
    # Find optimal threshold by iterating
    best_thresh = 0
    best_score = 0
    for t in range(0, 255, 5):
        _, b = cv2.threshold(blurred, t, 255, cv2.THRESH_BINARY_INV)
        score = cv2.countNonZero(b)
        if score > best_score:
            best_score = score
            best_thresh = t
    print(f"  Otsu threshold={best_thresh}, nonZero={best_score}")

    center_mask = np.zeros_like(binary)
    cv2.circle(center_mask, (roi_cx, roi_cy), int(radius * 0.85), 1, -1)
    cv2.circle(center_mask, (roi_cx, roi_cy), int(radius * 0.15), 0, -1)
    binary = cv2.bitwise_and(binary, binary, mask=center_mask)

    try:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Collect ALL candidate points (before distance filter)
    all_pts = []
    for cnt in contours:
        pts = cnt.flatten()
        for i in range(0, len(pts)-1, 2):
            px, py = pts[i], pts[i+1]
            dist = math.hypot(px-roi_cx, py-roi_cy)
            angle = math.degrees(math.atan2(py-roi_cy, px-roi_cx)) % 360
            all_pts.append((dist, px, py, angle))

    all_pts.sort(key=lambda x: -x[0])  # farthest first

    print(f"  Total contour pts: {len(all_pts)}")
    print(f"  Top 10 farthest pts:")
    for dist, px, py, ang in all_pts[:10]:
        img_x, img_y = px+x1, py+y1
        print(f"    dist={dist:.0f}  ROI=({px},{py})  img=({img_x},{img_y})  angle={ang:.1f} deg")

    # Try with a fixed lower threshold instead of Otsu
    print(f"  Trying fixed thresholds:")
    for t in [40, 60, 80, 100, 120]:
        _, b2 = cv2.threshold(blurred, t, 255, cv2.THRESH_BINARY_INV)
        b2 = cv2.bitwise_and(b2, b2, mask=center_mask)
        try:
            cnts2, _ = cv2.findContours(b2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            cnts2 = cv2.findContours(b2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_pt2, best_d2 = None, 0
        for cnt in cnts2:
            pts2 = cnt.flatten()
            for i in range(0, len(pts2)-1, 2):
                px2, py2 = pts2[i], pts2[i+1]
                d2 = math.hypot(px2-roi_cx, py2-roi_cy)
                if radius*0.2 < d2 < radius*0.9 and d2 > best_d2:
                    best_d2 = d2
                    best_pt2 = (px2+x1, py2+y1)
        if best_pt2:
            ang2 = math.degrees(math.atan2(best_pt2[1]-cy, best_pt2[0]-cx)) % 360
            print(f"    t={t}: best_pt={best_pt2} dist={best_d2:.0f} angle={ang2:.1f} deg")

    # Visualize
    vis = roi.copy()
    cv2.circle(vis, (roi_cx, roi_cy), 5, (0,255,0), 2)
    for i, (dist, px, py, ang) in enumerate(all_pts[:5]):
        color = (0,0,255) if i==0 else (128,128,255)
        cv2.circle(vis, (int(px),int(py)), 4, color, -1)
    cv2.imwrite(str(DEBUG / f"{fname.replace('.png','_s2.png')}"), vis)
    print(f"  Saved debug image")

for p in sorted((DEMO).glob("gauge_*.png"))[:3]:
    print(f"\n=== {p.name} (expected={int(p.stem.split('_')[1])/100}) ===")
    debug_s2(p)
