"""
压力表图像读取器
功能：捕获摄像头图像 → 图像处理识别指针角度 → 换算压力值
"""
import cv2
import numpy as np
import math
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("gauge_reader.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────── 配置区 ───────────
CAMERA_INDEX = 0
MJPEG_URL    = "http://192.168.1.100:8080/mjpeg"
USE_MJPEG    = True

GAUGE_CENTER = (320, 240)
GAUGE_RADIUS = 200
SCALE_MIN    = 0.0
SCALE_MAX    = 16.0
SCALE_UNIT   = "units"

# ── 角度换算（表盘 0-16，弧度 270°） ──
ANGLE_0_DEG = 225.0  # 0刻度在圆周的角度
SWEEP_RANGE = 270.0  # 有效弧度（225°→495° = 读数 0→16）

CIRCLE_PARAM1 = 50
CIRCLE_PARAM2 = 30
# ──────────────────────────────────


# ─────────── 兼容层 ───────────
def _get_contours(binary):
    """兼容 OpenCV 4/5"""
    try:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _contour_pts(cnt):
    """返回 [(x,y), ...]"""
    pts = cnt.flatten()
    return [(pts[i], pts[i+1]) for i in range(0, len(pts)-1, 2)]


# ─────────── 策略①：HoughLinesP 圆心→指针尖 ───────────
def _needle_angle_hough(edges, roi_cx, roi_cy, radius):
    """
    HoughLinesP → 找圆心→指针尖方向
    筛选条件：一端靠近圆心，另一端在表盘中部（排除 rim）
    """
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=30, minLineLength=20, maxLineGap=10)
    if lines is None:
        return None

    best_angle = None
    best_score = -1
    for row in lines:
        xa, ya, xb, yb = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        llen = math.hypot(xb - xa, yb - ya)
        if llen < 15:
            continue
        d_a = math.hypot(xa - roi_cx, ya - roi_cy)
        d_b = math.hypot(xb - roi_cx, yb - roi_cy)
        near, far = min(d_a, d_b), max(d_a, d_b)
        # 一端在内圆，另一端在表盘中部（0.30r–0.85r）
        if near < radius * 0.35 and radius * 0.30 < far < radius * 0.85:
            score = far - near
            if score > best_score:
                best_score = score
                tip_x = xa if d_a > d_b else xb  # ROI 坐标
                tip_y = ya if d_a > d_b else yb
                best_angle = math.degrees(math.atan2(tip_y - roi_cy, tip_x - roi_cx)) % 360.0
    return best_angle


# ─────────── 策略②：径向扫描 + 梯度验证 ───────────
def _needle_angle_radial(gray, roi_cx, roi_cy, radius):
    """
    径向扫描找 needle：
    1. 每 3° 沿半径向外找暗点，候选 = (angle, dark_val, dist)
    2. 对候选做梯度验证：needle 有暗→亮过渡（tip 暗，背景亮）
       hub shadow 只有单一暗点（无过渡）
    3. 返回梯度最大的候选角度
    """
    # 收集每条射线的暗点候选
    candidates = []  # (angle, min_val, min_dist)
    for angle_deg in range(0, 360, 3):
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        min_val, min_dist = 255, 0
        for r_pix in range(int(radius * 0.45), int(radius * 0.96), 2):
            px = int(roi_cx + cos_a * r_pix)
            py = int(roi_cy + sin_a * r_pix)
            if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                v = int(gray[py, px])
                if v < min_val:
                    min_val = v
                    min_dist = r_pix
        if min_val < 200:  # 任意暗点候选
            candidates.append((angle_deg, min_val, min_dist))

    if not candidates:
        return None

    # 梯度验证：对每个候选找最大径向梯度（暗→亮）
    best_angle, best_grad = None, 0
    for angle_deg, min_val, min_dist in candidates:
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        # 扫描 needle tip 区域（min_dist ±30%），找最大梯度
        grad_peak = 0
        scan_min = max(int(radius * 0.45), int(min_dist - radius * 0.15))
        scan_max = min(int(radius * 0.95), int(min_dist + radius * 0.10))
        for r_pix in range(scan_min, scan_max + 1, 2):
            px1 = int(roi_cx + cos_a * r_pix)
            py1 = int(roi_cy + sin_a * r_pix)
            px2 = int(roi_cx + cos_a * (r_pix + 6))
            py2 = int(roi_cy + sin_a * (r_pix + 6))
            if (0 <= px1 < gray.shape[1] and 0 <= py1 < gray.shape[0] and
                0 <= px2 < gray.shape[1] and 0 <= py2 < gray.shape[0]):
                # 梯度：像素沿射线方向的变化率
                grad = (int(gray[py2, px2]) - int(gray[py1, px1])) / 6.0
                if grad > grad_peak:
                    grad_peak = grad

        # 得分：梯度 × 距离权重（越远越可能是 needle tip）
        dist_weight = min_dist / (radius * 0.95)
        score = grad_peak * dist_weight
        if score > best_grad:
            best_grad = score
            best_angle = angle_deg

    return best_angle % 360.0 if best_angle is not None else None


# ─────────── 策略③：环形掩码 + 直方图（备用） ───────────
def _needle_angle_histogram(binary, roi_cx, roi_cy, radius):
    """
    角度直方图：找最拥挤的中环方向
    关键改进：用 0.55r–0.82r 环形区，尽量避开 hub 和 rim
    """
    contours = _get_contours(binary)
    if not contours:
        return None

    min_d = int(radius * 0.55)  # 避开 hub（~0.17r）
    max_d = int(radius * 0.82)  # 避开 rim（~0.89r）

    angles = []
    max_d_f = float(max_d)
    for cnt in contours:
        for px, py in _contour_pts(cnt):
            d = math.hypot(px - roi_cx, py - roi_cy)
            if min_d < d < max_d:
                ang = math.degrees(math.atan2(py - roi_cy, px - roi_cx)) % 360.0
                angles.append((ang, d))

    if len(angles) < 5:
        return None

    # 36 bins × 10°/bin
    N_BINS = 36
    bin_size = 360.0 / N_BINS
    hist = [0.0] * N_BINS
    for ang, dist in angles:
        weight = dist / max_d_f
        hist[int(ang / bin_size) % N_BINS] += weight

    # 平滑 + 找峰
    hist_s = [(hist[i] + hist[i-1] + hist[(i+1) % N_BINS]) / 3.0 for i in range(N_BINS)]
    best_bin = max(range(N_BINS), key=lambda i: hist_s[i])

    # 加权平均（峰 ±1 bin）
    bc = (best_bin + 0.5) * bin_size
    wt = hist_s[best_bin]
    wp = hist_s[(best_bin - 1) % N_BINS]
    wn = hist_s[(best_bin + 1) % N_BINS]
    if wt + wp + wn > 0:
        avg = (bc * wt +
               ((best_bin - 1) * bin_size + bin_size / 2) * wp +
               ((best_bin + 1) * bin_size + bin_size / 2) * wn) / (wt + wp + wn)
    else:
        avg = bc
    return avg % 360.0


# ─────────── 主函数 ───────────
def capture_frame(cam_index: int = 0) -> np.ndarray:
    """从 MJPEG 流或本地摄像头抓取一帧"""
    import urllib.request

    if USE_MJPEG:
        logger.info(f"正在连接 MJPEG 流: {MJPEG_URL}")
        stream = urllib.request.urlopen(MJPEG_URL, timeout=10)
        bytes_data = b""
        while True:
            bytes_data += stream.read(4096)
            a = bytes_data.find(b"\xff\xd8")
            b = bytes_data.find(b"\xff\xd9")
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    break
        logger.info(f"图像尺寸: {frame.shape[1]}x{frame.shape[0]}")
        return frame

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {cam_index}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("摄像头读取失败")
    return frame


def detect_gauge_center(frame: np.ndarray) -> tuple:
    """自动检测表盘圆心（霍夫圆检测），失败时返回默认值"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=CIRCLE_PARAM1, param2=CIRCLE_PARAM2,
        minRadius=int(min(frame.shape) * 0.2),
        maxRadius=int(min(frame.shape) * 0.5)
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        best = max(circles[0], key=lambda c: c[2])
        cx, cy, r = best
        logger.info(f"检测到表盘圆心: ({cx}, {cy})，半径: {r}")
        return (cx, cy), r
    logger.warning("未检测到圆，使用默认圆心/半径")
    return GAUGE_CENTER, GAUGE_RADIUS


def find_needle_angle(frame: np.ndarray, center: tuple, radius: int) -> float:
    """
    检测指针角度（弧度，0=右，逆时针增加）
    顺序：①HoughLinesP ②径向扫描 ③环形直方图
    """
    cx, cy = center
    roi_size = int(radius * 1.4)
    x1 = max(0, cx - roi_size)
    y1 = max(0, cy - roi_size)
    x2 = min(frame.shape[1], cx + roi_size)
    y2 = min(frame.shape[0], cy + roi_size)
    roi = frame[y1:y2, x1:x2]
    roi_cx, roi_cy = cx - x1, cy - y1

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blurred, 30, 90)

    # ── 策略①：HoughLinesP ──
    angle1 = _needle_angle_hough(edges, roi_cx, roi_cy, radius)
    if angle1 is not None:
        logger.info(f"策略① HoughLines → 角度 {angle1:.1f}°")
        return math.radians(angle1)

    # ── 策略②：径向扫描 ──
    angle2 = _needle_angle_radial(blurred, roi_cx, roi_cy, radius)
    if angle2 is not None:
        logger.info(f"策略② 径向扫描 → 角度 {angle2:.1f}°")
        return math.radians(angle2)

    # ── 策略③：多阈值 + 环形直方图 ──
    best_angle, best_nearby = None, 0
    for method, binary in [
        ("adaptive", cv2.adaptiveThreshold(blurred, 255,
              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
              blockSize=21, C=10)),
        ("t40",      *_thresh_fixed(blurred, 40)),
        ("t60",      *_thresh_fixed(blurred, 60)),
        ("t80",      *_thresh_fixed(blurred, 80)),
    ]:
        needle_angle = _needle_angle_histogram(binary, roi_cx, roi_cy, radius)
        if needle_angle is not None:
            # 统计该角度 ±15° 内的有效轮廓点数
            nearby = sum(
                1 for cnt in _get_contours(binary)
                  for px, py in _contour_pts(cnt)
                  if radius * 0.55 < math.hypot(px - roi_cx, py - roi_cy) < radius * 0.82
                  and min(abs((math.degrees(math.atan2(py - roi_cy, px - roi_cx)) % 360) - needle_angle),
                          360 - abs((math.degrees(math.atan2(py - roi_cy, px - roi_cx)) % 360) - needle_angle)) < 15
            )
            if nearby > best_nearby:
                best_nearby = nearby
                best_angle = needle_angle

    if best_angle is not None:
        logger.info(f"策略③ 直方图 → 角度 {best_angle:.1f}°")
        return math.radians(best_angle)

    raise ValueError("未找到指针轮廓")


def _thresh_fixed(gray, t):
    _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
    return _, binary


def angle_to_pressure(angle_rad: float, verbose: bool = False) -> float:
    """指针角度 -> 压力值"""
    angle_deg = math.degrees(angle_rad) % 360.0
    offset = (ANGLE_0_DEG - angle_deg + 360.0) % 360.0
    if offset > SWEEP_RANGE:
        offset = offset - 360.0
    offset = max(0.0, min(SWEEP_RANGE, offset))
    pressure = SCALE_MIN + (offset / SWEEP_RANGE) * (SCALE_MAX - SCALE_MIN)
    if verbose:
        print(f"  angle_deg={angle_deg:.1f} offset={offset:.1f} → {pressure:.2f}")
    return round(pressure, 2)


def read_gauge(cam_index: int = CAMERA_INDEX, image_path: str = None) -> dict:
    """拍照或读取图片，识别并返回结果"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"无法读取图片: {image_path}")
        img_path = image_path
    else:
        frame = capture_frame(cam_index)
        img_path = f"gauge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

    center, radius = detect_gauge_center(frame)
    angle = find_needle_angle(frame, center, radius)
    pressure = angle_to_pressure(angle)

    result = {
        "timestamp": ts,
        "pressure_value": pressure,
        "unit": SCALE_UNIT,
        "angle_deg": round(math.degrees(angle), 1),
        "image_path": img_path,
        "center": center,
        "radius": radius,
    }
    logger.info(f"识别结果: {pressure} {SCALE_UNIT}")
    return result


if __name__ == "__main__":
    import sys, os
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if img_path and os.path.exists(img_path):
        result = read_gauge(image_path=img_path)
    else:
        result = read_gauge()
    print(f"[{result['timestamp']}] 压力 = {result['pressure_value']} {result['unit']} "
          f"(角度={result['angle_deg']}°)")
