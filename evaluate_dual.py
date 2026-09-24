"""
few_teeth_tracking/evaluate_dual.py

フル歯列モデル（グラウンドトゥルース代替）と少数歯クラスタモデルを
同じカメラフレームで同時にトラッキングし、姿勢差（回転誤差・並進誤差）を
フレームごとに計測・CSV 保存する評価スクリプト。

使い方:
    python evaluate_dual.py

    - 左画面: フル歯列モデルの追跡結果
    - 右画面: クラスタモデルの追跡結果
    - 下段: 回転誤差(deg)・並進誤差(mm) のリアルタイムグラフ

操作:
    Space  : 追跡開始/停止
    1〜9   : フル歯列モデルの姿勢スロット切り替え
    Shift+1〜9 : クラスタモデルの姿勢スロット切り替え
    s      : 両モデルの現在姿勢をスロット保存
    v      : 表示モード切替
    q      : 終了
"""

import os
import sys

# few_teeth_tracking は pym3t とは別ディレクトリに配置されているため、
# pym3t.pyd 等の所在地を明示的に指定する。
PYM3T_DIR = r"C:\Users\rytkw\Projects\pym3t"
sys.path.insert(0, PYM3T_DIR)

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import csv
import time
from collections import deque

import cv2
import numpy as np
import pym3t

# ─── 設定 ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FULL_MODEL_PATH    = os.path.join(PYM3T_DIR, "models", "typodont.obj")
CLUSTER_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "cluster.obj")

calibration_path  = os.path.join(PYM3T_DIR, "camera_calibration.npz")
full_poses_dir    = os.path.join(PYM3T_DIR, "initial_poses")
cluster_poses_dir = os.path.join(SCRIPT_DIR, "initial_poses")
log_dir           = os.path.join(SCRIPT_DIR, "logs")

MODEL_UNIT_IN_METER = 0.04
MODEL_SPHERE_RADIUS = 0.20
SHARP_EDGE_ANGLE_DEG = 90.0

QUALITY_LOSS_THRESHOLD   = 0.12
QUALITY_LOSS_CONSECUTIVE = 5

PLOT_HISTORY = 200  # グラフに表示するフレーム数
PANEL_H      = 120  # 下段グラフの高さ

# ─── カメラ ──────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret, frame0 = cap.read()
if not ret:
    raise RuntimeError("カメラ初期化失敗")

h, w = frame0.shape[:2]
K = np.array([[800, 0, w/2], [0, 800, h/2], [0, 0, 1]], dtype=np.float32)
undistort_map = None

if os.path.exists(calibration_path):
    calib = np.load(calibration_path)
    cw, ch = int(calib["image_width"][0]), int(calib["image_height"][0])
    if cw == w and ch == h:
        Kc   = calib["K"].astype(np.float32)
        dist = calib["dist"].astype(np.float32)
        new_K, _ = cv2.getOptimalNewCameraMatrix(Kc, dist, (w, h), 0, (w, h))
        undistort_map = cv2.initUndistortRectifyMap(
            Kc, dist, None, new_K.astype(np.float32), (w, h), cv2.CV_16SC2)
        K = new_K.astype(np.float32)
        print("キャリブレーション読み込み完了")

def prepare_frame(raw):
    if undistort_map is not None:
        return cv2.remap(raw, undistort_map[0], undistort_map[1], cv2.INTER_LINEAR)
    return raw


# ─── OBJ エッジ ──────────────────────────────────────────────────────────

def load_obj_edges(path, max_edges=500, sharp_angle_deg=90.0):
    vertices, faces = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("v "):
                p = s.split()
                if len(p) >= 4:
                    vertices.append([float(p[1]), float(p[2]), float(p[3])])
            elif s.startswith("f "):
                ids = []
                for tok in s.split()[1:]:
                    try: ids.append(int(tok.split("/")[0]) - 1)
                    except: pass
                if len(ids) >= 3:
                    faces.append(ids)
    if not vertices or not faces:
        return []

    verts = np.array(vertices, dtype=np.float32)
    n_v = len(verts)
    tris = [f for f in faces if len(f) == 3]
    if not tris:
        return []
    ta = np.array(tris, dtype=np.int64)
    p0, p1, p2 = verts[ta[:,0]], verts[ta[:,1]], verts[ta[:,2]]
    norms = np.cross(p1-p0, p2-p0)
    nlen  = np.linalg.norm(norms, axis=1, keepdims=True)
    nlen[nlen < 1e-12] = 1.0
    norms = (norms / nlen).astype(np.float32)

    edges = np.stack([ta[:,[0,1]], ta[:,[1,2]], ta[:,[2,0]]], axis=1).reshape(-1, 2)
    fids  = np.repeat(np.arange(len(tris), dtype=np.int64), 3)
    a_can = np.minimum(edges[:,0], edges[:,1])
    b_can = np.maximum(edges[:,0], edges[:,1])
    keys  = a_can * n_v + b_can
    order = np.argsort(keys, kind="stable")
    sk = keys[order]; sf = fids[order]
    chg = np.concatenate([[True], sk[1:] != sk[:-1]])
    gs = np.nonzero(chg)[0]; ge = np.concatenate([gs[1:],[len(sk)]])
    two_starts = gs[(ge-gs) == 2]
    fa, fb = sf[two_starts], sf[two_starts+1]
    cos_a = np.clip(np.sum(norms[fa]*norms[fb], axis=1), -1.0, 1.0)
    dih   = np.degrees(np.arccos(cos_a))
    sharp = sk[two_starts[dih >= sharp_angle_deg]]
    sa = (sharp // n_v).astype(int)
    sb = (sharp % n_v).astype(int)
    result = [(verts[a], verts[b]) for a, b in zip(sa, sb)]
    if len(result) > max_edges:
        step = len(result) / max_edges
        result = [result[int(i*step)] for i in range(max_edges)]
    return result


# ─── M3T セットアップ ─────────────────────────────────────────────────────

for label, path in [("フル歯列", FULL_MODEL_PATH), ("クラスタ", CLUSTER_MODEL_PATH)]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label}モデルが見つかりません: {path}")

tracker_full = pym3t.Tracker(w, h, K, use_region=True, use_depth=False, use_texture=False)
model_full   = pym3t.Model("full", FULL_MODEL_PATH, "", "", MODEL_UNIT_IN_METER, MODEL_SPHERE_RADIUS)
tracker_full.add_model(model_full)
viewer_full  = pym3t.Viewer(tracker_full)

tracker_cls  = pym3t.Tracker(w, h, K, use_region=True, use_depth=False, use_texture=False)
model_cls    = pym3t.Model("cluster", CLUSTER_MODEL_PATH, "", "", MODEL_UNIT_IN_METER, MODEL_SPHERE_RADIUS)
tracker_cls.add_model(model_cls)
viewer_cls   = pym3t.Viewer(tracker_cls)

print("エッジ読み込み中...")
edges_full = load_obj_edges(FULL_MODEL_PATH)
edges_cls  = load_obj_edges(CLUSTER_MODEL_PATH)
print(f"  フル: {len(edges_full)} edges,  クラスタ: {len(edges_cls)} edges")


# ─── 姿勢スロット ─────────────────────────────────────────────────────────

def load_slot(n, d):
    p = os.path.join(d, f"initial_pose_{n}.npy")
    return np.load(p).astype(np.float32) if os.path.exists(p) else None

def save_slot(n, d, pose):
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f"initial_pose_{n}.npy"), pose.astype(np.float32))

default_pose = np.eye(4, dtype=np.float32)
default_pose[2, 3] = 0.4

pose_f = load_slot(1, full_poses_dir)    or default_pose.copy()
pose_c = load_slot(1, cluster_poses_dir) or default_pose.copy()
model_full.reset_pose_cv(pose_f)
model_cls.reset_pose_cv(pose_c)

slot_f = slot_c = 1


# ─── 品質・誤差計算 ──────────────────────────────────────────────────────

def project(pt3d, pose, K_mat):
    p = pose[:3,:3] @ (pt3d * MODEL_UNIT_IN_METER) + pose[:3,3]
    if p[2] <= 0:
        return None
    px = int(K_mat[0,0]*p[0]/p[2] + K_mat[0,2])
    py = int(K_mat[1,1]*p[1]/p[2] + K_mat[1,2])
    return px, py

def edge_quality(dist_map, edges, pose, K_mat, thr=5.0):
    if not edges:
        return 0.0
    hits = 0
    for a, b in edges:
        uv = project((a+b)/2, pose, K_mat)
        if uv:
            x, y = uv
            if 0 <= x < dist_map.shape[1] and 0 <= y < dist_map.shape[0]:
                if dist_map[y, x] < thr:
                    hits += 1
    return hits / len(edges)

def pose_error(pose_a, pose_b):
    """pose_a をグラウンドトゥルース、pose_b を比較対象として誤差を返す。
    Returns: (trans_error_mm, rot_error_deg)
    """
    R_a, t_a = pose_a[:3,:3], pose_a[:3,3]
    R_b, t_b = pose_b[:3,:3], pose_b[:3,3]
    t_err_m = np.linalg.norm(t_b - t_a)
    t_err_mm = t_err_m * 1000.0
    R_rel = R_b @ R_a.T
    trace = np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)
    rot_err_deg = np.degrees(np.arccos(trace))
    return float(t_err_mm), float(rot_err_deg)


# ─── 描画 ────────────────────────────────────────────────────────────────

def draw_wireframe(bgr, edges, pose, K_mat, color=(0, 200, 0)):
    out = bgr.copy()
    for a, b in edges:
        uva = project(a, pose, K_mat)
        uvb = project(b, pose, K_mat)
        if uva and uvb:
            x1,y1 = uva; x2,y2 = uvb
            if (0<=x1<w and 0<=y1<h) or (0<=x2<w and 0<=y2<h):
                cv2.line(out, (x1,y1), (x2,y2), color, 1)
    return out

def draw_error_plot(trans_hist, rot_hist, panel_w, panel_h):
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    def draw_line(hist, color, max_val, label):
        if len(hist) < 2:
            return
        pts = []
        n = len(hist)
        for i, v in enumerate(hist):
            x = int(i / max(n-1, 1) * (panel_w - 1))
            y = int((1.0 - min(v, max_val) / max_val) * (panel_h - 1))
            pts.append((x, y))
        for i in range(len(pts)-1):
            cv2.line(panel, pts[i], pts[i+1], color, 1)
        if pts:
            cv2.putText(panel, f"{label}: {hist[-1]:.2f}", pts[-1],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    draw_line(trans_hist, (0, 200, 255), 20.0,  "Trans(mm)")
    draw_line(rot_hist,   (255, 100, 0), 30.0, "Rot(deg)")
    return panel


# ─── CSV ロガー ──────────────────────────────────────────────────────────

os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"eval_dual_{time.strftime('%Y%m%d_%H%M%S')}.csv")
log_f = open(log_path, "w", newline="", encoding="utf-8")
log_w = csv.writer(log_f)
log_w.writerow([
    "frame", "timestamp",
    "full_quality", "cluster_quality",
    "trans_error_mm", "rot_error_deg",
    "full_lost_streak", "cluster_lost_streak",
])

def log_row(frame_idx, qf, qc, terr, rerr, lsf, lsc):
    log_w.writerow([frame_idx, f"{time.time():.6f}",
                    f"{qf:.4f}", f"{qc:.4f}",
                    f"{terr:.4f}", f"{rerr:.4f}",
                    lsf, lsc])


# ─── メインループ ─────────────────────────────────────────────────────────

tracking_active = False
lost_f = lost_c = 0
trans_hist = deque(maxlen=PLOT_HISTORY)
rot_hist   = deque(maxlen=PLOT_HISTORY)

frame_idx = 0
print(f"ログ: {log_path}")
print("Space=開始/停止  1〜9=フル姿勢スロット  Shift+1〜9=クラスタ姿勢スロット  q=終了")

cv2.namedWindow("evaluate_dual", cv2.WINDOW_NORMAL)

while True:
    ret, raw = cap.read()
    if not ret:
        break
    frame_img = prepare_frame(raw)
    frame_idx += 1

    dist_map = compute_dist_map = cv2.distanceTransform(
        cv2.bitwise_not(cv2.Canny(cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY), 50, 150)),
        cv2.DIST_L2, 5
    )

    if tracking_active:
        tracker_full.step(frame_img)
        tracker_cls.step(frame_img)
        pose_f = model_full.pose_cv.astype(np.float32)
        pose_c = model_cls.pose_cv.astype(np.float32)

        qf = edge_quality(dist_map, edges_full, pose_f, K)
        qc = edge_quality(dist_map, edges_cls,  pose_c, K)

        if qf < QUALITY_LOSS_THRESHOLD:
            lost_f += 1
        else:
            lost_f = 0
        if qc < QUALITY_LOSS_THRESHOLD:
            lost_c += 1
        else:
            lost_c = 0

        terr, rerr = pose_error(pose_f, pose_c)
        trans_hist.append(terr)
        rot_hist.append(rerr)
        log_row(frame_idx, qf, qc, terr, rerr, lost_f, lost_c)
    else:
        pose_f = model_full.pose_cv.astype(np.float32)
        pose_c = model_cls.pose_cv.astype(np.float32)
        qf = qc = terr = rerr = 0.0

    # 左パネル: フル歯列
    rendered_f = viewer_full.view_color(False)
    wire_f     = draw_wireframe(frame_img, edges_full, pose_f, K, (0, 200, 0))
    left       = cv2.addWeighted(rendered_f, 0.45, wire_f, 0.55, 0)
    cv2.putText(left, f"FULL  Q:{qf:.2f}  LOST:{lost_f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 右パネル: クラスタ
    rendered_c = viewer_cls.view_color(False)
    wire_c     = draw_wireframe(frame_img, edges_cls, pose_c, K, (255, 100, 0))
    right      = cv2.addWeighted(rendered_c, 0.45, wire_c, 0.55, 0)
    cv2.putText(right, f"CLUSTER  Q:{qc:.2f}  LOST:{lost_c}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

    top = np.hstack([left, right])

    # 誤差グラフ
    panel = draw_error_plot(trans_hist, rot_hist, top.shape[1], PANEL_H)
    if tracking_active:
        cv2.putText(panel,
                    f"Trans err: {terr:.2f} mm   Rot err: {rerr:.2f} deg",
                    (10, PANEL_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    show = np.vstack([top, panel])
    cv2.imshow("evaluate_dual", show)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        tracking_active = not tracking_active
        print("追跡" + ("開始" if tracking_active else "停止"))
    elif key == ord("s"):
        save_slot(slot_f, full_poses_dir, pose_f)
        save_slot(slot_c, cluster_poses_dir, pose_c)
        print(f"フル スロット{slot_f}, クラスタ スロット{slot_c} を保存")
    else:
        # 1〜9 でフル, Shift+1〜9 でクラスタ
        slot_map_lo = {ord(str(i)): i for i in range(1, 10)}
        slot_map_hi = {ord(c): i for i, c in enumerate("!@#$%^&*(", 1)}
        if key in slot_map_lo:
            n = slot_map_lo[key]
            p = load_slot(n, full_poses_dir)
            if p is not None:
                pose_f = p; model_full.reset_pose_cv(pose_f); slot_f = n
                print(f"フル スロット{n} 読み込み")
        elif key in slot_map_hi:
            n = slot_map_hi[key]
            p = load_slot(n, cluster_poses_dir)
            if p is not None:
                pose_c = p; model_cls.reset_pose_cv(pose_c); slot_c = n
                print(f"クラスタ スロット{n} 読み込み")

cap.release()
log_f.close()
cv2.destroyAllWindows()
print(f"終了。ログ: {log_path}")
