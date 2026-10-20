import argparse
import glob
import os
import time

import sys

import cv2
import numpy as np

# リポジトリ直下の camera_io.py（OS に合ったカメラのバックエンド、m3t_local.env の読み込み）を使う
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import camera_io  # noqa: E402
camera_io.load_local_env()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_DIR = os.path.join(SCRIPT_DIR, "calibration_images")
DEFAULT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "camera_calibration.npz")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=camera_io.camera_index(),
                        help="Camera index (default: M3T_CAMERA_INDEX or 0). Find it with tools/list_cameras.py.")
    parser.add_argument("--cols", type=int, default=9, help="Inner checkerboard corners per row.")
    parser.add_argument("--rows", type=int, default=6, help="Inner checkerboard corners per column.")
    parser.add_argument("--square-mm", type=float, default=25.0, help="Checkerboard square size in mm.")
    parser.add_argument("--width", type=int, default=None, help="Capture width (e.g. 1280). Must match the tracker's CAMERA_WIDTH.")
    parser.add_argument("--height", type=int, default=None, help="Capture height (e.g. 720). Must match the tracker's CAMERA_HEIGHT.")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true")
    return parser.parse_args()


def capture_images(camera_index, image_dir, pattern_size, width=None, height=None):
    os.makedirs(image_dir, exist_ok=True)

    # Capture at the same resolution the tracker uses, so the saved K matches.
    # Backend: DirectShow on Windows, AVFoundation on macOS (camera_io.py).
    if not (width and height):
        width, height = camera_io.camera_size()
    cap, (aw, ah) = camera_io.open_camera(camera_index, width, height)
    if (aw, ah) != (width, height):
        print("WARNING: camera did not accept the requested resolution; the calibration will be saved for "
              f"{aw}x{ah}. Set M3T_CAMERA_WIDTH/HEIGHT to that size so the tracker uses this K.")

    existing = glob.glob(os.path.join(image_dir, "calib_*.png"))
    next_index = len(existing)

    print("Capture checkerboard images.")
    print("Space: save image with detected corners")
    print("s: save image even if corners are not detected")
    print("q or Esc: finish capture")

    # プレビューのチェッカーボード検出は別スレッドで最新フレームにだけ行う。メインループで毎フレーム検出していたときは
    # （1280x720・高速チェックなし）ボードが見つからないフレームで約 300 ms かかり、映像が溜まって重かった。
    # 検出は半分の大きさ・高速チェック付き（見つからないフレームで約 90 ms）。Space では、検出できたそのフレームを
    # 元の解像度で保存する（計算時に保存画像から検出し直す）。
    import threading
    lock = threading.Lock()
    state = {"pending": None, "result": None, "stop": False}
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK

    def detect_worker():
        while True:
            with lock:
                if state["stop"]:
                    return
                job, state["pending"] = state["pending"], None
            if job is None:
                time.sleep(0.005)
                continue
            frame_bgr = job
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            found, corners = cv2.findChessboardCorners(small, pattern_size, flags)
            with lock:
                state["result"] = (found, corners * 2.0 if found else None, frame_bgr, time.time())

    worker = threading.Thread(target=detect_worker, daemon=True)
    worker.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        with lock:
            state["pending"] = frame        # 検出が追いつかないフレームは捨てて最新だけを渡す
            result = state["result"]
        found = bool(result and result[0] and time.time() - result[3] < 0.5)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, pattern_size, result[1], True)

        cv2.putText(
            display,
            f"saved: {next_index}  corners: {'OK' if found else 'NO'}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0) if found else (0, 0, 255),
            2,
        )
        cv2.imshow("Camera Calibration Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == 32 and found:
            path = os.path.join(image_dir, f"calib_{next_index:03d}.png")
            cv2.imwrite(path, result[2])    # コーナーを検出できたフレームそのもの
            print("saved", path)
            next_index += 1
        elif key == ord("s"):
            path = os.path.join(image_dir, f"calib_{next_index:03d}.png")
            cv2.imwrite(path, frame)
            print("saved without detected corners", path)
            next_index += 1

    with lock:
        state["stop"] = True
    worker.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()


def calibrate_from_images(image_dir, output_path, pattern_size, square_mm):
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not image_paths:
        raise RuntimeError(f"no calibration images found in {image_dir}")

    square_m = square_mm / 1000.0
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_m

    objpoints = []
    imgpoints = []
    image_size = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for image_path in image_paths:
        frame = cv2.imread(image_path)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            print("skip, corners not found:", image_path)
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        print("use:", image_path)

    if len(objpoints) < 8:
        raise RuntimeError(f"need at least 8 good images, found {len(objpoints)}")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    np.savez(
        output_path,
        K=K.astype(np.float32),
        dist=dist.astype(np.float32),
        image_width=np.array([image_size[0]], dtype=np.int32),
        image_height=np.array([image_size[1]], dtype=np.int32),
        rms=np.array([rms], dtype=np.float32),
        square_mm=np.array([square_mm], dtype=np.float32),
        cols=np.array([pattern_size[0]], dtype=np.int32),
        rows=np.array([pattern_size[1]], dtype=np.int32),
    )

    print("calibration saved:", output_path)
    print("rms:", rms)
    print("K:")
    print(K)
    print("dist:")
    print(dist.ravel())


def main():
    args = parse_args()
    pattern_size = (args.cols, args.rows)

    if not args.calibrate_only:
        capture_images(args.camera, args.image_dir, pattern_size, args.width, args.height)
    if not args.capture_only:
        calibrate_from_images(args.image_dir, args.output, pattern_size, args.square_mm)


if __name__ == "__main__":
    main()
