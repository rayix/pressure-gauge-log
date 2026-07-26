"""看直方图 top bins"""
import sys, math
from pathlib import Path
import logging
for _l in [logging.root]: _l.setLevel(logging.CRITICAL + 1)
sys.path.insert(0, str(Path(__file__).parent))
import cv2, numpy as np

DEMO = Path(__file__).parent / "demo_output"

def _get_contours(binary):
    try:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def _contour_pts(cnt):
    pts = cnt.flatten()
    return [(pts[i], pts[i+1]) for i in range(0, len(pts)-1, 2)]

N_BINS = 45
BIN_SIZE = 360.0 / N_BINS

def hist_top(binary, roi_cx, roi_cy, radius):
    try:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_d, max_d = int(radius*0.25), int(radius*0.88)
    angles = []
    for cnt in contours:
        for px, py in _contour_pts(cnt):
            d = math.hypot(px-roi_cx, py-roi_cy)
            if min_d < d < max_d:
                ang = math.degrees(math.atan2(py-roi_cy, px-roi_cx)) % 360.0
                angles.append((ang, d))
    hist = [0.0] * N_BINS
    max_d_f = float(max_d)
    for ang, dist in angles:
        weight = dist / max_d_f
        hist[int(ang/BIN_SIZE) % N_BINS] += weight
    # smooth
    hist_s = [(hist[i]+hist[i-1]+hist[(i+1)%N_BINS])/3 for i in range(N_BINS)]
    top5 = sorted(range(N_BINS), key=lambda i: -hist_s[i])[:5]
    print(f"  Points={len(angles)}, top bins:")
    for bi in top5:
        center = (bi+0.5)*BIN_SIZE
        print(f"    bin{bi} center={center:.1f} deg  val={hist_s[bi]:.1f}  raw={hist[bi]:.1f}")

for fname in ["gauge_135.png", "gauge_200.png"]:
    path = DEMO / fname
    frame = cv2.imread(str(path))
    cx, cy, radius = 320, 240, 200
    roi_size = int(radius*1.4)
    x1, y1 = max(0,cx-roi_size), max(0,cy-roi_size)
    x2, y2 = min(frame.shape[1],cx+roi_size), min(frame.shape[0],cy+roi_size)
    roi = frame[y1:y2, x1:x2]
    roi_cx, roi_cy = cx-x1, cy-y1
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    print(f"\n=== {fname} ===")
    for method, desc in [
        ("adaptive", "adaptiveGaussian"),
        ("fixed60",  "thresh=60"),
        ("fixed80",  "thresh=80"),
    ]:
        if method == "adaptive":
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, blockSize=21, C=10)
        else:
            _, binary = cv2.threshold(blurred, int(method.replace("fixed","")), 255, cv2.THRESH_BINARY_INV)
        center_mask = np.zeros_like(binary)
        cv2.circle(center_mask, (roi_cx, roi_cy), int(radius*0.85), 1, -1)
        cv2.circle(center_mask, (roi_cx, roi_cy), int(radius*0.15), 0, -1)
        binary = cv2.bitwise_and(binary, binary, mask=center_mask)
        print(f"  [{desc}]")
        hist_top(binary, roi_cx, roi_cy, radius)
