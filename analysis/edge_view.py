# -*- coding: utf-8 -*-
"""Reproduce the tracker's auto-Canny on a raw saved frame and compare thresholds."""
import cv2, glob, os, sys
import numpy as np

frames_dir = sys.argv[1]
out_path = sys.argv[2]
files = sorted(glob.glob(os.path.join(frames_dir, "f*.jpg")))
src = files[len(files) // 2]
bgr = cv2.imread(src)
h, w = bgr.shape[:2]
scale = min(1.0, 640.0 / w)                     # QUALITY_MAX_WIDTH = 640
small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

med = float(np.median(gray))
auto_lo, auto_hi = int(max(0, 0.67 * med)), int(min(255, 1.33 * med))

# Teeth region estimate: bright, low-saturation pixels (white/cream teeth), largest blob
hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
teeth = ((hsv[..., 2] > 150) & (hsv[..., 1] < 60)).astype(np.uint8) * 255
teeth = cv2.morphologyEx(teeth, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

def edges(lo, hi):
    return cv2.Canny(gray, lo, hi)

variants = [("auto (tracker) %d/%d" % (auto_lo, auto_hi), auto_lo, auto_hi),
            ("fixed 40/100", 40, 100),
            ("fixed 25/60", 25, 60)]
print(f"frame: {os.path.basename(src)}  {w}x{h} -> {gray.shape[1]}x{gray.shape[0]}  median gray={med:.1f}", flush=True)
tiles = []
for name, lo, hi in variants:
    e = edges(lo, hi)
    dens_all = (e > 0).mean() * 100
    dens_teeth = (e[teeth > 0] > 0).mean() * 100 if (teeth > 0).any() else float("nan")
    print(f"  {name:24s} edge density: whole={dens_all:5.2f}%  inside bright/teeth-like area={dens_teeth:5.2f}%", flush=True)
    vis = small.copy()
    vis[e > 0] = (0, 255, 255)
    cv2.putText(vis, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    tiles.append(vis)
sheet = np.vstack(tiles)
cv2.imwrite(out_path, sheet)
print("saved", out_path, sheet.shape, flush=True)
