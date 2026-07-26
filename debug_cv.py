import sys
sys.stdout.reconfigure(encoding='utf-8')
import cv2, numpy as np

img = cv2.imread(r'C:\Users\Administrator\.qclaw\workspace-agent-74338476\pressure_gauge_logger\demo_output\gauge_200.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 5)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
center_mask = np.zeros_like(binary)
cv2.circle(center_mask, (320,240), 200, 1, -1)
cv2.circle(center_mask, (320,240), 30, 0, -1)
binary = cv2.bitwise_and(binary, binary, mask=center_mask)

ret = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('nret:', len(ret))
if len(ret) == 2:
    contours, hierarchy = ret
else:
    contours, hierarchy, _ = ret + (None,) * (3 - len(ret))
print('contours type=%s' % type(contours).__name__)
print('contours len=%d' % len(contours))
if contours:
    c = contours[0]
    print('contour[0] shape=%s dtype=%s ndim=%d' % (c.shape, c.dtype, c.ndim))
    print('contour[0] raw:', c[:3])
    print('flattened[:6]:', c.flatten()[:6])
