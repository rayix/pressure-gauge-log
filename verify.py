"""
verify.py - 用 demo_output 样本图片验证 gauge_reader 算法
用法: python verify.py
"""
import sys, os, re, math, logging
from pathlib import Path

# 静默所有日志（在 import gauge_reader 之前）
for _l in [logging.root, logging.getLogger('gauge_reader')]:
    _l.setLevel(logging.CRITICAL + 1)

sys.path.insert(0, str(Path(__file__).parent))
import cv2
import numpy as np

from gauge_reader import detect_gauge_center, find_needle_angle, angle_to_pressure


DEMO_DIR = Path(__file__).parent / "demo_output"


def parse_expected(filename: str):
    """gauge_XXX.png -> 期望值 XXX/100"""
    m = re.search(r'gauge_(\d+)\.png', filename)
    return int(m.group(1)) / 100.0 if m else None


def verify_image(img_path: str):
    frame = cv2.imread(img_path)
    if frame is None:
        return None, "cannot read"
    try:
        center, radius = detect_gauge_center(frame)
        angle = find_needle_angle(frame, center, radius)
        pressure = angle_to_pressure(angle)
        angle_deg = math.degrees(angle) % 360.0
        return {"pressure": pressure, "angle_deg": round(angle_deg, 1)}, None
    except Exception as e:
        return None, str(e)


def main():
    images = sorted(DEMO_DIR.glob("gauge_*.png"))
    if not images:
        print(f"No images found in {DEMO_DIR}")
        sys.exit(1)

    print(f"{'File':<25} {'Exp':>6} {'Got':>6} {'Diff':>7} {'Angle':>7} Status")
    print("-" * 65)

    errors = []
    for img_path in images:
        fname = img_path.name
        expected = parse_expected(fname)
        result, err = verify_image(str(img_path))
        if err:
            print(f"{fname:<25} {'ERR':>6}   --     --    {err}")
            continue
        detected = result["pressure"]
        angle_deg = result["angle_deg"]
        if expected is not None:
            diff = round(detected - expected, 2)
            ok = abs(diff) <= 0.5
            mark = "[OK]" if ok else "[FAIL]"
            print(f"{fname:<25} {expected:>6.2f} {detected:>6.2f} {diff:>+7.2f} "
                  f"{angle_deg:>6.1f} deg {mark}")
            if not ok:
                errors.append((fname, expected, detected, diff))
        else:
            print(f"{fname:<25} {'--':>6} {detected:>6.2f} {angle_deg:>6.1f} deg")

    print()
    n = len(images)
    if errors:
        print(f"[FAIL] {n - len(errors)}/{n} passed, {len(errors)} failed:")
        for fname, exp, det, diff in errors:
            print(f"   {fname}: expected {exp:.2f}, got {det:.2f}, diff {diff:+.2f}")
    else:
        print(f"[OK] All {n}/{n} passed!")

    # 角度分布诊断
    print("\n-- Angle distribution --")
    results = []
    for img_path in sorted(images):
        result, err = verify_image(str(img_path))
        if err:
            continue
        exp = parse_expected(img_path.name)
        results.append((result["angle_deg"], img_path.name, exp, result["pressure"]))
    results.sort(key=lambda x: x[0])
    for ang, fname, exp, det in results:
        exp_s = f"{exp:.2f}" if exp else "--"
        print(f"  {ang:5.1f} deg  {fname:<25}  exp={exp_s}  got={det:.2f}")


if __name__ == "__main__":
    main()
