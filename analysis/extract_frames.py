# -*- coding: utf-8 -*-
import cv2, sys, os
import numpy as np

out_dir, src = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
cap = cv2.VideoCapture(src)
if not cap.isOpened():
    print("OPEN FAILED", flush=True); sys.exit(1)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"video: {w}x{h} {fps:.2f} fps  {n} frames  {n/fps:.1f} s", flush=True)
N = 12
tiles = []
for k in range(N):
    i = int(n * (k + 0.5) / N)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = cap.read()
    if not ok:
        continue
    cv2.imwrite(os.path.join(out_dir, f"frame_{k:02d}.png"), f)
    t = cv2.resize(f, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
    cv2.putText(t, f"#{k} t={i/fps:.1f}s", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    tiles.append(t)
cols = 3
rows = (len(tiles) + cols - 1) // cols
th, tw = tiles[0].shape[:2]
sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
for k, t in enumerate(tiles):
    r, c = divmod(k, cols)
    sheet[r*th:(r+1)*th, c*tw:(c+1)*tw] = t
cv2.imwrite(os.path.join(out_dir, "contact_sheet.png"), sheet)
print("sheet:", sheet.shape, "frames:", len(tiles), flush=True)
