import argparse
import glob
import os

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_DIR = os.path.join(SCRIPT_DIR, "calibration_images")
DEFAULT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "camera_calibration.npz")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
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

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("camera open failed")

    # Capture at the same resolution the tracker uses, so the saved K matches.
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"camera resolution: requested {width}x{height}, actual {aw}x{ah}")
        if (aw, ah) != (width, height):
            print("WARNING: camera did not accept the requested resolution.")

    existing = glob.glob(os.path.join(image_dir, "calib_*.png"))
    next_index = len(existing)

    print("Capture checkerboard images.")
    print("Space: save image with detected corners")
    print("s: save image even if corners are not detected")
    print("q or Esc: finish capture")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, pattern_size, corners, found)

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
            cv2.imwrite(path, frame)
            print("saved", path)
            next_index += 1
        elif key == ord("s"):
            path = os.path.join(image_dir, f"calib_{next_index:03d}.png")
            cv2.imwrite(path, frame)
            print("saved without detected corners", path)
            next_index += 1

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
