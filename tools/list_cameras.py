# -*- coding: utf-8 -*-
"""使えるカメラ番号を調べる（Mac の連係カメラ＝iPhone が何番かを見つけるため。Windows でも使える）。

    python tools/list_cameras.py            # 0〜5 番を試す
    python tools/list_cameras.py --max 3

各番号で 1280x720（M3T_CAMERA_WIDTH/HEIGHT）を要求して 2 秒読み、実際の解像度と fps を表示し、
logs/camera_probe/cam<番号>.jpg に 1 枚保存する。画像を見て iPhone の映像の番号を
m3t_local.env の M3T_CAMERA_INDEX に書く。
Mac では最初の 1 回、ターミナルにカメラの使用許可を求めるダイアログが出る（許可しないと全部「開けない」になる）。
"""
import argparse
import os
import subprocess
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import camera_io  # noqa: E402


def main():
    camera_io.load_local_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=5, help="試す最大の番号")
    ap.add_argument("--seconds", type=float, default=2.0, help="fps を測る秒数")
    args = ap.parse_args()

    out_dir = os.path.join(camera_io.REPO_DIR, "logs", "camera_probe")
    os.makedirs(out_dir, exist_ok=True)
    name, _ = camera_io.camera_backend()
    w_req, h_req = camera_io.camera_size()
    print(f"backend={name}  要求解像度 {w_req}x{h_req}  保存先 {out_dir}")

    if sys.platform == "darwin":
        # 番号と名前の対応は保証されないが、どんなカメラがつながっているかの目安になる
        try:
            names = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True,
                                   timeout=20).stdout
            print("--- system_profiler SPCameraDataType ---")
            print("\n".join(l for l in names.splitlines() if l.strip().endswith(":") and not l.startswith("Camera")))
        except Exception as e:  # noqa: BLE001
            print(f"(カメラ名の取得に失敗: {e})")

    found = []
    for idx in range(args.max + 1):
        try:
            cap, (aw, ah) = camera_io.open_camera(idx, w_req, h_req, verbose=False)
        except RuntimeError:
            print(f"[{idx}] 開けない")
            continue
        n, frame, t0 = 0, None, time.time()
        while time.time() - t0 < args.seconds:
            ok, f = cap.read()
            if ok:
                n += 1
                frame = f
        cap.release()
        if frame is None:
            print(f"[{idx}] 開けたが映像が来ない")
            continue
        fps = n / args.seconds
        path = os.path.join(out_dir, f"cam{idx}.jpg")
        cv2.imwrite(path, frame)
        mark = "" if (aw, ah) == (w_req, h_req) else "  ← 要求と違う解像度"
        print(f"[{idx}] {aw}x{ah}  {fps:4.1f} fps  → {path}{mark}")
        found.append(idx)

    if found:
        print(f"\n映像が取れた番号: {found}。画像を見て iPhone の番号を m3t_local.env の M3T_CAMERA_INDEX=<番号> に書く。")
    else:
        print("\n映像が取れるカメラが無い。Mac ならカメラの使用許可と、iPhone が連係カメラとして見えているか確認。")


if __name__ == "__main__":
    main()
