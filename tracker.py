"""
few_teeth_tracking/tracker.py

少数歯牙クラスタ（対象歯＋隣接1〜2本）用トラッカー。
pym3t (親ディレクトリの .pyd) を使い、クラスタ用OBJモデルを追跡する。
Unity との連携は test_tracker.py と同じ UDP/TCP ポートを使う。
追跡品質・姿勢をフレームごとに CSV ログとして記録する。

pym3t_contra/test_tracker.py（実運用で鍛えられた版）をベースに、
単一オブジェクト用に簡略化して移植:
  - 自動姿勢選択（保存済み12スロット + 回転摂動を粗く採点→上位だけ精密化）
  - 手動ナッジ中は reset_pose_cv を呼ばない（毎回呼ぶとレンダラと衝突しクラッシュする）
  - One-Euro フィルタで Unity 送信用の姿勢だけ平滑化（FIT QUALITY 判定は生の姿勢のまま）
  - 品質が悪いフレームは last_good_pose を送り続け、Unity 側が飛ばないようにする

モデルの準備:
    models/ に対象クラスタの OBJ を置き、CLUSTER_MODEL_NAME を設定する。
    初回起動時に M3T が .meta ファイルを自動生成する（数分かかる）。
"""

import os
import sys

# few_teeth_tracking は pym3t とは別ディレクトリに配置されているため、
# pym3t.pyd 等の所在地を明示的に指定する。
PYM3T_DIR = r"C:\Users\rytkw\Projects\pym3t"
sys.path.insert(0, PYM3T_DIR)

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import csv
import json
import socket
import struct
import threading
import traceback
import time

import cv2
import numpy as np
import pym3t

# ─── クラッシュ捕捉 ─────────────────────────────────────────────────────────
# 原因不明の「勝手に終了する」現象を調査するため、メインスレッド・別スレッド
# 双方の未処理例外を必ずファイルに記録する（標準出力のバッファリングで
# 見えなくなることがあるため、ファイルへの記録は open() 直後に flush する）。

_CRASH_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "crash.log")


def _write_crash_log(where, exc_type, exc_value, exc_tb):
    os.makedirs(os.path.dirname(_CRASH_LOG_PATH), exist_ok=True)
    with open(_CRASH_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n--- {where} crash at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        f.flush()
    print(f"[CRASH] {where}: {exc_type.__name__}: {exc_value}  (詳細: {_CRASH_LOG_PATH})", flush=True)


def _excepthook(exc_type, exc_value, exc_tb):
    _write_crash_log("main thread", exc_type, exc_value, exc_tb)
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def _threading_excepthook(args):
    _write_crash_log(f"thread '{args.thread.name}'", args.exc_type, args.exc_value, args.exc_traceback)


sys.excepthook = _excepthook
threading.excepthook = _threading_excepthook

# ─── 設定 ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 追跡するクラスタモデルのファイル名（models/ 以下）
CLUSTER_MODEL_NAME = "cluster_molar_premolar_smooth_pinned.obj"

model_path        = os.path.join(SCRIPT_DIR, "models", CLUSTER_MODEL_NAME)
pose_path         = os.path.join(SCRIPT_DIR, "initial_pose.npy")
# クラスタごとに typodont 座標系内での位置が大きく異なる（README のクラスタ一覧参照）ため、
# 姿勢スロットはモデルファイル名ごとに分けて保存する。分けないと、別クラスタで保存した
# スロットを読み込んだ際にモデルが実物から大きくズレた位置に出てしまう。
_model_key        = os.path.splitext(CLUSTER_MODEL_NAME)[0]
pose_slots_dir    = os.path.join(SCRIPT_DIR, "initial_poses", _model_key)
calibration_path  = os.path.join(PYM3T_DIR, "camera_calibration.npz")
log_dir           = os.path.join(SCRIPT_DIR, "logs")

# OBJ の 1 モデル単位 = 何メートルか（typodont と同じスケールなら 0.04）
MODEL_UNIT_IN_METER = 0.04
# M3T の内部マルチスケール探索範囲を決めるバウンディング球の半径（メートル）。
# 0.20 はフル歯列アーチ（typodont 全体）用の値がそのまま残っていたもので、
# cluster_molar_premolar_crownonly.obj の実寸（0.04倍後で外接球半径 約0.032m）
# に対して6倍以上大きい。対象に対してスケールが大きすぎる探索範囲は
# region modality の直線探索の粒度を対象サイズに対して粗くしてしまうため、
# 実寸に合わせて縮小する。
MODEL_SPHERE_RADIUS = 0.035

# シャープエッジ（品質スコア用）の判定角度。歯冠だけの滑らかなスキャンだと
# 90度でも数本しか残らないが、それでも歯間の谷は実ジオメトリの折れ目として
# 拾えるので、これはこのままでよい。
# ただし cluster_molar_premolar.obj は最大二面角が89.59度で90度に届かず
# シャープエッジ0本でRuntimeErrorになるため60度に下げた（27本確保できる）。
SHARP_EDGE_ANGLE_DEG = 60.0

# Unity 連携
UNITY_UDP_ENABLED   = True
UNITY_UDP_HOST      = "127.0.0.1"
UNITY_UDP_PORT      = 5005
UNITY_COMMAND_PORT  = 5006
UNITY_VIDEO_ENABLED = True
UNITY_VIDEO_PORT    = 5010
UNITY_VIDEO_FPS     = 10
UNITY_VIDEO_WIDTH   = 640
UNITY_VIDEO_JPEG_QUALITY = 60

POSE_SLOT_COUNT = 12
POSE_SLOT_KEYS = {
    ord("1"): 1, ord("2"): 2, ord("3"): 3, ord("4"): 4,
    ord("5"): 5, ord("6"): 6, ord("7"): 7, ord("8"): 8,
    ord("9"): 9, ord("0"): 10, ord("-"): 11, ord("="): 12,
}

SNAP_START_STEPS = 30
SNAP_FRAME_STEPS = 3
SNAP_FRAME_COUNT = 15
AUTO_SELECT_STEPS = 8
# pym3t 内部の n_corr_iterations/n_update_iterations (5/2) は固定で Python
# から変更できないため、1フレームにつき tracker.step() を複数回呼ぶことで
# 探索範囲を広げる（特に速い回転で有効）。コストは概ね倍率分かかる。
# ただし molar_premolar クラスタのように歯冠が小さく輪郭が弱い対象では、
# 1フレームで3回も補正をかけると弱いエッジのノイズを増幅して暴走しやすい
# ことがログ解析（tz が1秒未満で100mm超ジャンプ）で見えたため、まずは
# 1（pym3tネイティブの1回分）に戻して暴走が減るか検証する。
TRACKING_STEPS_PER_FRAME = 1
ROTATION_REJECT_DEG = 120.0
ROTATION_PENALTY_WEIGHT = 4.0
QUALITY_SELECT_WEIGHT = 300.0  # 候補順位を支配する: (1 - edge_match) * これ

# 自動姿勢選択の候補生成: 保存済み12スロットだけでは隙間（10〜30度以上）
# ができることがあり、AUTO_SELECT_STEPS の範囲では埋まらない。各スロットに
# 安価な回転摂動を加えて候補を密にし、Cannyエッジとの粗い一致度で上位だけ
# 実際にトラッカーで精密化する。
COARSE_YAW_STEPS_DEG = (-20.0, -10.0, 0.0, 10.0, 20.0)
COARSE_PITCH_STEPS_DEG = (-10.0, 0.0, 10.0)
COARSE_TOP_K = 5
VIEW_MODES = ("M3T", "WIREFRAME", "BLEND")
FRAME_READ_RETRY_LIMIT = 120

# 継続的な自動再ロック（品質が落ちたら毎回全探索）は暴れやすい（pose が
# ふらつく）ことが実運用で分かったため、デフォルトは OFF。品質計算と
# last_good_pose によるフリーズ送信自体は常に有効（Unity 側が飛ばないための
# 安全策）。't' キーでライブに切り替え可能。
QUALITY_CHECK_ENABLED = False
QUALITY_EDGE_MAX_DIST_PX = 6.0   # 投影したエッジが実エッジにこの距離以内なら「ヒット」
QUALITY_LOSS_THRESHOLD  = 0.40   # これ未満のヒット率は「悪いフレーム」
QUALITY_GOOD_THRESHOLD  = 0.55   # これ以上で lost_streak をリセット/「良好」とみなす
QUALITY_LOSS_CONSECUTIVE = 10    # 何フレーム連続で悪いと再ロックを試みるか
RECOVERY_COOLDOWN_SEC    = 1.5   # 自動再ロックの最短間隔

# ─── カメラ初期化 ─────────────────────────────────────────────────────────

# 歯・顎模型が白く光沢があるため、自動露出だとハイライトが白飛びして
# 輪郭（明暗の境界）が消えてしまうことがある。露出を手動で少し下げて
# クリッピングを抑えられるか試すためのスイッチ。
# None = 自動露出のまま（元の挙動）。数値を入れると自動露出をOFFにして
# その値を設定する（DirectShowは露出を2の指数スケールで扱うことが多く、
# -4 前後が明るめ、-8 前後で大分暗くなる。カメラ機種依存なので実際の
# 映像を見ながら調整すること）。
CAMERA_MANUAL_EXPOSURE = -6

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if CAMERA_MANUAL_EXPOSURE is not None:
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_MANUAL_EXPOSURE)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("カメラの読み取りに失敗しました")

current_exposure = CAMERA_MANUAL_EXPOSURE if CAMERA_MANUAL_EXPOSURE is not None else cap.get(cv2.CAP_PROP_EXPOSURE)

if CAMERA_MANUAL_EXPOSURE is not None:
    actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    actual_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    print(f"手動露出を設定: 要求={CAMERA_MANUAL_EXPOSURE} 実際の値(exposure={actual_exposure}, auto_exposure={actual_auto})")
    print("  ドライバによっては反映されないことがある。'[' で露出を下げる/']' で上げる（実行中に調整可能）。")

h, w = frame.shape[:2]

K = np.array([
    [800,   0, w / 2],
    [  0, 800, h / 2],
    [  0,   0,     1],
], dtype=np.float32)
dist = None
undistort_map = None

if os.path.exists(calibration_path):
    calib = np.load(calibration_path)
    calib_w = int(calib["image_width"][0])
    calib_h = int(calib["image_height"][0])
    if calib_w == w and calib_h == h:
        K    = calib["K"].astype(np.float32)
        dist = calib["dist"].astype(np.float32)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
        undistort_map = cv2.initUndistortRectifyMap(
            K, dist, None, new_K.astype(np.float32), (w, h), cv2.CV_16SC2)
        K = new_K.astype(np.float32)
        print("カメラキャリブレーション読み込み完了")
    else:
        print(f"キャリブレーション解像度不一致: {calib_w}x{calib_h} vs {w}x{h}")
else:
    print("camera_calibration.npz が見つからないため仮の K を使用")


def prepare_frame(raw):
    if undistort_map is not None:
        return cv2.remap(raw, undistort_map[0], undistort_map[1], cv2.INTER_LINEAR)
    return raw


# ─── OBJ エッジ読み込み ───────────────────────────────────────────────────

def load_obj_edges(path, max_edges=500, sharp_angle_deg=90.0):
    """OBJ を読み込み、幾何学的に鋭いエッジ（歯間の谷・稜線など実際の折れ目）
    だけを残す。歯肉/歯牙の色境界のような、3Dの折れ目が無い平坦面の境界を
    誤って拾わないよう、二面角（隣接する2面の法線のなす角）でフィルタする。
    """
    vertices, faces = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                p = line.split()
                if len(p) >= 4:
                    vertices.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                ids = []
                for tok in line.split()[1:]:
                    try:
                        ids.append(int(tok.split("/")[0]) - 1)
                    except Exception:
                        pass
                if len(ids) >= 3:
                    faces.append(ids)

    vertices = np.array(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError(f"OBJ の頂点を正しく読み込めませんでした: {vertices.shape}")

    n_verts = vertices.shape[0]
    tri_faces = [f for f in faces if len(f) == 3]
    other_faces = [f for f in faces if len(f) != 3]

    tri_arr = np.array(tri_faces, dtype=np.int64)
    p0, p1, p2 = vertices[tri_arr[:, 0]], vertices[tri_arr[:, 1]], vertices[tri_arr[:, 2]]
    tri_normals = np.cross(p1 - p0, p2 - p0)
    tri_norm_len = np.linalg.norm(tri_normals, axis=1, keepdims=True)
    tri_norm_len[tri_norm_len < 1e-12] = 1.0
    tri_normals = (tri_normals / tri_norm_len).astype(np.float32)

    tri_edge_pairs = np.stack([
        tri_arr[:, [0, 1]], tri_arr[:, [1, 2]], tri_arr[:, [2, 0]],
    ], axis=1).reshape(-1, 2)
    tri_face_ids = np.repeat(np.arange(len(tri_faces), dtype=np.int64), 3)

    other_normals, other_edge_a, other_edge_b, other_face_ids = [], [], [], []
    face_offset = len(tri_faces)
    for face_i, face in enumerate(other_faces):
        pts = vertices[face]
        n = len(pts)
        normal = np.zeros(3, dtype=np.float64)
        for i in range(n):
            cur, nxt = pts[i], pts[(i + 1) % n]
            normal[0] += (cur[1] - nxt[1]) * (cur[2] + nxt[2])
            normal[1] += (cur[2] - nxt[2]) * (cur[0] + nxt[0])
            normal[2] += (cur[0] - nxt[0]) * (cur[1] + nxt[1])
        norm_len = np.linalg.norm(normal)
        other_normals.append(normal / norm_len if norm_len > 1e-12 else np.array([0.0, 0.0, 1.0]))
        for i in range(n):
            other_edge_a.append(face[i])
            other_edge_b.append(face[(i + 1) % n])
            other_face_ids.append(face_offset + face_i)

    if other_faces:
        all_normals = np.concatenate(
            [tri_normals, np.array(other_normals, dtype=np.float32).reshape(-1, 3)], axis=0)
        edge_a_arr = np.concatenate([tri_edge_pairs[:, 0], np.array(other_edge_a, dtype=np.int64)])
        edge_b_arr = np.concatenate([tri_edge_pairs[:, 1], np.array(other_edge_b, dtype=np.int64)])
        face_idx_arr = np.concatenate([tri_face_ids, np.array(other_face_ids, dtype=np.int64)])
    else:
        all_normals = tri_normals
        edge_a_arr = tri_edge_pairs[:, 0]
        edge_b_arr = tri_edge_pairs[:, 1]
        face_idx_arr = tri_face_ids

    a_can = np.minimum(edge_a_arr, edge_b_arr).astype(np.int64)
    b_can = np.maximum(edge_a_arr, edge_b_arr).astype(np.int64)
    keys = a_can * n_verts + b_can

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_faces = face_idx_arr[order]

    change = np.concatenate([[True], sorted_keys[1:] != sorted_keys[:-1]])
    group_starts = np.nonzero(change)[0]
    group_ends = np.concatenate([group_starts[1:], [len(sorted_keys)]])
    two_face_mask = (group_ends - group_starts) == 2
    two_face_starts = group_starts[two_face_mask]
    face_a_idx = sorted_faces[two_face_starts]
    face_b_idx = sorted_faces[two_face_starts + 1]

    na = all_normals[face_a_idx]
    nb = all_normals[face_b_idx]
    cos_angle = np.clip(np.sum(na * nb, axis=1), -1.0, 1.0)
    dihedral_deg = np.degrees(np.arccos(cos_angle))

    sharp_mask = dihedral_deg >= sharp_angle_deg
    sharp_keys = sorted_keys[two_face_starts[sharp_mask]]

    sharp_a = (sharp_keys // n_verts).astype(int)
    sharp_b = (sharp_keys % n_verts).astype(int)
    edges = list(zip(sharp_a.tolist(), sharp_b.tolist()))

    print(f"  シャープエッジ (dihedral >= {sharp_angle_deg} deg): {len(edges)} / {len(group_starts)} 総エッジ")

    if len(edges) == 0:
        raise RuntimeError("シャープエッジが見つかりませんでした。sharp_angle_deg を下げてください。")

    if len(edges) > max_edges:
        idx = np.linspace(0, len(edges) - 1, max_edges).astype(int)
        edges = [edges[i] for i in idx]

    return vertices, edges


def rot_x(deg):
    a = np.deg2rad(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], dtype=np.float32)


def rot_y(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], dtype=np.float32)


def rot_z(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=np.float32)


# ─── One-Euro フィルタ（Unity 送信用の姿勢だけ平滑化） ───────────────────

def rotmat_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw, qx, qy, qz = 0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw, qx, qy, qz = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw, qx, qy, qz = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw, qx, qy, qz = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_to_rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float32)


def slerp(q0, q1, t):
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot) * t
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def one_euro_smoothing_factor(t_e, cutoff):
    r = 2.0 * np.pi * cutoff * t_e
    return r / (r + 1.0)


class OneEuroFilter:
    """標準的な One-Euro フィルタ（位置用）。静止時は強く平滑化し、動きが
    速くなるほど生の値に近づく。"""

    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.x_prev = self.dx_prev = self.t_prev = None

    def filter(self, x, t):
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()
        t_e = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / t_e
        a_d = one_euro_smoothing_factor(t_e, self.dcutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        speed = float(np.linalg.norm(dx_hat))
        cutoff = self.mincutoff + self.beta * speed
        a = one_euro_smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class OneEuroRotationFilter:
    """回転版 One-Euro フィルタ。角速度でカットオフを駆動し、slerp で補間する。"""

    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.q_prev, self.speed_prev, self.t_prev = None, 0.0, None

    def filter(self, R, t):
        q = rotmat_to_quat(R)
        if self.q_prev is None:
            self.q_prev, self.t_prev = q, t
            return R.copy()
        t_e = max(t - self.t_prev, 1e-3)
        dot = np.clip(abs(float(np.dot(q, self.q_prev))), -1.0, 1.0)
        speed = (2.0 * np.arccos(dot)) / t_e
        a_d = one_euro_smoothing_factor(t_e, self.dcutoff)
        speed_hat = a_d * speed + (1.0 - a_d) * self.speed_prev
        cutoff = self.mincutoff + self.beta * speed_hat
        a = one_euro_smoothing_factor(t_e, cutoff)
        q_hat = slerp(self.q_prev, q, a)
        self.q_prev, self.speed_prev, self.t_prev = q_hat, speed_hat, t
        return quat_to_rotmat(q_hat)


pos_filter = OneEuroFilter(mincutoff=1.0, beta=0.5)
rot_filter = OneEuroRotationFilter(mincutoff=1.0, beta=0.05)


def smooth_pose(raw_pose):
    """Unity へ送る姿勢だけ平滑化する。トラッカー自身が使う生の pose 変数は
    そのままにして、FIT QUALITY 判定が平滑化のラグに引きずられないようにする。"""
    now = time.time()
    smoothed = np.eye(4, dtype=np.float32)
    smoothed[:3, 3] = pos_filter.filter(raw_pose[:3, 3], now)
    smoothed[:3, :3] = rot_filter.filter(raw_pose[:3, :3], now)
    return smoothed


# ─── M3T セットアップ ─────────────────────────────────────────────────────

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"クラスタモデルが見つかりません: {model_path}\n"
        "extract_cluster_mesh.py / segment_teeth.py でモデルを生成してから起動してください。"
    )

# contra 版は use_texture=True だが、クラスタOBJにはテクスチャ/色情報が
# 無いため、ここでは region（輪郭）のみを使う。
tracker = pym3t.Tracker(w, h, K, use_region=True, use_depth=False, use_texture=False)
model = pym3t.Model(CLUSTER_MODEL_NAME, model_path, "", "", MODEL_UNIT_IN_METER, MODEL_SPHERE_RADIUS)

print("OBJ エッジ読み込み中...")
vertices, edges = load_obj_edges(model_path, max_edges=500, sharp_angle_deg=SHARP_EDGE_ANGLE_DEG)
vertices *= MODEL_UNIT_IN_METER
used_ids = sorted(set(i for e in edges for i in e))
vertex_map = {old: new for new, old in enumerate(used_ids)}
draw_vertices = vertices[used_ids]
draw_edges = [(vertex_map[a], vertex_map[b]) for a, b in edges]
print(f"  品質判定用 頂点:{draw_vertices.shape} エッジ:{len(draw_edges)}")

# 表示専用の密なエッジセット（人間が目視で位置合わせできるよう、角度に
# 関係なく全エッジを対象にする。品質判定用のシャープエッジだけだと疎すぎる）。
viz_vertices, viz_edges = load_obj_edges(model_path, max_edges=2500, sharp_angle_deg=0.0)
viz_vertices *= MODEL_UNIT_IN_METER
viz_used_ids = sorted(set(i for e in viz_edges for i in e))
viz_vertex_map = {old: new for new, old in enumerate(viz_used_ids)}
viz_draw_vertices = viz_vertices[viz_used_ids]
viz_draw_edges = [(viz_vertex_map[a], viz_vertex_map[b]) for a, b in viz_edges]
print(f"  表示用 頂点:{viz_draw_vertices.shape} エッジ:{len(viz_draw_edges)}")

EDGE_A_IDX = np.array([a for a, b in draw_edges], dtype=np.int32)
EDGE_B_IDX = np.array([b for a, b in draw_edges], dtype=np.int32)
EDGE_SAMPLE_T = np.linspace(0.0, 1.0, 4, dtype=np.float32)

model.reset_pose_cv(np.eye(4, dtype=np.float32))  # 後で load_pose_slot の結果に差し替える
tracker.add_model(model)
viewer = pym3t.Viewer(tracker)


# ─── 姿勢スロット ─────────────────────────────────────────────────────────

os.makedirs(pose_slots_dir, exist_ok=True)


def slot_pose_path(slot):
    return os.path.join(pose_slots_dir, f"initial_pose_{slot}.npy")


def load_pose_slot(slot):
    path = slot_pose_path(slot)
    if os.path.exists(path):
        return np.load(path).astype(np.float32), path
    # 旧形式（モデル別に分ける前）の単一ファイルは cluster.obj の分のみ引き続き読める。
    if slot == 1 and _model_key == "cluster" and os.path.exists(pose_path):
        return np.load(pose_path).astype(np.float32), pose_path
    return None, path


def save_pose_slot(slot, pose_mat):
    path = slot_pose_path(slot)
    np.save(path, pose_mat.astype(np.float32))
    if slot == 1 and _model_key == "cluster":
        np.save(pose_path, pose_mat.astype(np.float32))
    return path


selected_pose_slot = 1
pose, loaded_pose_path = load_pose_slot(selected_pose_slot)
if pose is not None:
    print(f"初期姿勢スロット {selected_pose_slot} を読み込み: {loaded_pose_path}")
else:
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = 0.4
model.reset_pose_cv(pose)


# ─── 追跡品質評価・自動姿勢選択 ──────────────────────────────────────────

CANNY_AUTO_SIGMA = 0.33


def compute_edge_distance_map(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 固定しきい値(50,150)だと、消灯時など画像全体が暗く勾配が弱いシーンで
    # エッジがほとんど検出されず、姿勢が合っていても品質スコアが不当に低く
    # 出てしまう。画像の中央値ベースでしきい値を自動決定し、明るさに
    # 依存しないエッジ密度にする（いわゆる auto-canny）。
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - CANNY_AUTO_SIGMA) * median))
    upper = int(min(255, (1.0 + CANNY_AUTO_SIGMA) * median))
    canny = cv2.Canny(gray, lower, upper)
    return cv2.distanceTransform(255 - canny, cv2.DIST_L2, 3)


def project_vertices(verts, candidate_pose):
    R = candidate_pose[:3, :3]
    t = candidate_pose[:3, 3]
    pts_cam = (R @ verts.T).T + t
    valid = pts_cam[:, 2] > 0.05
    x = K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]
    y = K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]
    pts2d = np.stack([x, y], axis=1)
    finite = np.isfinite(pts2d).all(axis=1)
    return pts2d, valid & finite


def estimate_tracking_quality(dist_map, candidate_pose, max_dist=QUALITY_EDGE_MAX_DIST_PX):
    """投影したワイヤーフレームのうち、実画像のエッジ付近に乗っている割合。
    pym3t は残差/信頼度を返さないため、これが唯一のフィット判定手段になる。
    エッジごとサンプル数を固定してベクトル化し、候補が多くても軽い。"""
    pts2d, valid = project_vertices(draw_vertices, candidate_pose)
    a2d, b2d = pts2d[EDGE_A_IDX], pts2d[EDGE_B_IDX]
    edge_valid = valid[EDGE_A_IDX] & valid[EDGE_B_IDX]

    t = EDGE_SAMPLE_T[None, :, None]
    samples = a2d[:, None, :] * (1.0 - t) + b2d[:, None, :] * t

    hgt, wid = dist_map.shape[:2]
    xi = np.round(samples[..., 0]).astype(np.int32)
    yi = np.round(samples[..., 1]).astype(np.int32)
    in_bounds = (xi >= 0) & (xi < wid) & (yi >= 0) & (yi < hgt) & edge_valid[:, None]
    xi_c, yi_c = np.clip(xi, 0, wid - 1), np.clip(yi, 0, hgt - 1)
    hits = in_bounds & (dist_map[yi_c, xi_c] <= max_dist)

    total = int(np.count_nonzero(in_bounds))
    return float(np.count_nonzero(hits)) / total if total > 0 else 0.0


def projected_center(candidate_pose):
    center_model = np.mean(draw_vertices, axis=0)
    center_cam = candidate_pose[:3, :3] @ center_model + candidate_pose[:3, 3]
    if center_cam[2] <= 0.05:
        return None
    x = K[0, 0] * center_cam[0] / center_cam[2] + K[0, 2]
    y = K[1, 1] * center_cam[1] / center_cam[2] + K[1, 2]
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return np.array([x, y], dtype=np.float32)


def rotation_angle_deg(pose_a, pose_b):
    r_delta = pose_a[:3, :3] @ pose_b[:3, :3].T
    cos_angle = np.clip((float(np.trace(r_delta)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))


def generate_candidate_poses():
    """保存済み12スロットそれぞれに、安価な回転摂動を加えて候補を密にする。"""
    candidates = []
    for slot in range(1, POSE_SLOT_COUNT + 1):
        base_pose, _ = load_pose_slot(slot)
        if base_pose is None:
            continue
        for yaw in COARSE_YAW_STEPS_DEG:
            for pitch in COARSE_PITCH_STEPS_DEG:
                candidate = base_pose.copy()
                if yaw != 0.0 or pitch != 0.0:
                    delta = rot_y(yaw) @ rot_x(pitch)
                    candidate[:3, :3] = delta @ base_pose[:3, :3]
                candidates.append((slot, candidate))
    return candidates


def pose_auto_select(frame_bgr, dist_map, use_rotation_guard=True, reference_pose=None):
    """2段階の自動姿勢選択。
    Stage1（安価）: 候補全部を生の投影でCannyエッジと粗く採点（tracker.step()なし）。
    Stage2（高価）: 上位 COARSE_TOP_K だけ実際にトラッカーで精密化して最終採点。
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    original_pose = pose.copy().astype(np.float32)
    if reference_pose is None:
        reference_pose, _ = load_pose_slot(selected_pose_slot)
        if reference_pose is None:
            reference_pose = original_pose

    coarse_results = []
    for slot, candidate_pose in generate_candidate_poses():
        rotation_delta = rotation_angle_deg(candidate_pose, reference_pose)
        if use_rotation_guard and rotation_delta > ROTATION_REJECT_DEG:
            continue
        depth = float(candidate_pose[2, 3])
        if not (0.03 < depth < 2.0):
            continue
        if projected_center(candidate_pose) is None:
            continue
        coarse_quality = estimate_tracking_quality(dist_map, candidate_pose)
        coarse_results.append((coarse_quality, rotation_delta, slot, candidate_pose))

    if not coarse_results:
        model.reset_pose_cv(original_pose)
        print("自動姿勢選択失敗: 粗選抜を通った候補がありません")
        return original_pose, None

    coarse_results.sort(key=lambda item: item[0], reverse=True)
    shortlist = coarse_results[:COARSE_TOP_K]
    print(f"自動姿勢選択: 候補 {len(coarse_results)} 個中、上位 {len(shortlist)} 個を精密化")

    best_slot, best_pose, best_score, best_quality = None, None, np.inf, 0.0
    for coarse_quality, rotation_delta, slot, candidate_pose in shortlist:
        before_center = projected_center(candidate_pose)
        model.reset_pose_cv(candidate_pose)
        for _ in range(AUTO_SELECT_STEPS):
            tracker.step(frame_rgb)
        refined_pose = model.pose_cv.copy().astype(np.float32)
        after_center = projected_center(refined_pose)
        if after_center is None:
            continue

        quality = estimate_tracking_quality(dist_map, refined_pose)
        move_px = float(np.linalg.norm(after_center - before_center))
        depth = float(refined_pose[2, 3])
        center_penalty = float(np.linalg.norm(after_center - np.array([w * 0.5, h * 0.5], dtype=np.float32))) * 0.02
        depth_penalty = 0.0 if 0.03 < depth < 2.0 else 10000.0
        rotation_penalty = rotation_delta * ROTATION_PENALTY_WEIGHT if use_rotation_guard else 0.0
        quality_penalty = (1.0 - quality) * QUALITY_SELECT_WEIGHT
        score = quality_penalty + move_px + center_penalty + depth_penalty + rotation_penalty
        print(f"  候補(元スロット{slot}): score={score:.2f} coarse_q={coarse_quality:.2f} "
              f"refined_q={quality:.2f} move_px={move_px:.2f} rot={rotation_delta:.1f}deg depth={depth:.3f}")

        if score < best_score:
            best_score, best_slot, best_pose, best_quality = score, slot, refined_pose, quality

    if best_pose is None:
        model.reset_pose_cv(original_pose)
        print("自動姿勢選択失敗: 精密化後に有効な候補がありません")
        return original_pose, None

    model.reset_pose_cv(best_pose)
    print(f"自動選択完了（元スロット{best_slot}近傍）: score={best_score:.2f} quality={best_quality:.2f}")
    return best_pose, best_slot


def snap_to_current_frame(frame_bgr, steps=SNAP_START_STEPS):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    for _ in range(steps):
        tracker.step(frame_rgb)
    return model.pose_cv.copy().astype(np.float32)


# ─── ワイヤーフレーム描画 ─────────────────────────────────────────────────

def draw_obj_wireframe(bgr, pose_mat):
    R = pose_mat[:3, :3]
    t = pose_mat[:3, 3]
    pts_cam = (R @ viz_draw_vertices.T).T + t
    valid = pts_cam[:, 2] > 0.05

    out = bgr.copy()
    if np.count_nonzero(valid) < 10:
        cv2.putText(out, "Model is behind camera or too close", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out

    x = K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]
    y = K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]
    pts2d = np.stack([x, y], axis=1)
    finite = np.isfinite(pts2d).all(axis=1)

    for a, b in viz_draw_edges:
        if not (valid[a] and valid[b] and finite[a] and finite[b]):
            continue
        p1, p2 = pts2d[a].astype(int), pts2d[b].astype(int)
        if (0 <= p1[0] < w and 0 <= p1[1] < h) or (0 <= p2[0] < w and 0 <= p2[1] < h):
            cv2.line(out, tuple(p1), tuple(p2), (255, 0, 255), 1)
    return out


# ─── 追跡開始・再ロック ───────────────────────────────────────────────────

lost_streak = 0
last_recovery_time = 0.0
last_good_pose = None
last_quality = 0.0
recovery_escalated = False


def start_tracking_from_frame(frame_bgr, use_rotation_guard=True):
    global pose, selected_pose_slot, snap_frames, tracking_active
    global lost_streak, last_recovery_time, last_good_pose, last_quality, recovery_escalated

    tracking_active = True
    dist_map = compute_edge_distance_map(frame_bgr)
    auto_pose, auto_slot = pose_auto_select(frame_bgr, dist_map, use_rotation_guard=use_rotation_guard)
    pose = auto_pose
    if auto_slot is not None:
        selected_pose_slot = auto_slot

    pose = snap_to_current_frame(frame_bgr)
    snap_frames = SNAP_FRAME_COUNT

    lost_streak = 0
    last_recovery_time = time.time()
    last_good_pose = pose.copy()
    # SNAP のウォームアップ中は品質判定をスキップするため、ここで暫定的に
    # 「良好」扱いにしておく（そうしないと古い last_quality が低いままだと
    # Unity 送信のフリーズゲートがこの新しい姿勢を止めてしまう）。
    last_quality = 1.0
    recovery_escalated = False
    pos_filter.x_prev = None
    rot_filter.q_prev = None

    mode = "SAFE" if use_rotation_guard else "GLOBAL"
    print(f"追跡開始 / {mode} 自動姿勢選択 / SNAPブースト")


def relock_from_frame(frame_bgr):
    """追跡中にスロットキーや 'r' で手動再ロックする場合の GLOBAL 全探索。
    以前は保存済みスロットの生の姿勢へ飛ばして snap に任せるだけだったが、
    実物が大きくずれていると収束せず、押しても押しても復帰しないことが
    あった。Space/g のコールドスタートと同じ無制限探索を使うことで解決する。
    """
    global pose, selected_pose_slot, snap_frames
    global lost_streak, last_recovery_time, last_good_pose, last_quality, recovery_escalated

    dist_map = compute_edge_distance_map(frame_bgr)
    auto_pose, auto_slot = pose_auto_select(frame_bgr, dist_map, use_rotation_guard=False)
    pose = auto_pose
    if auto_slot is not None:
        selected_pose_slot = auto_slot

    pose = snap_to_current_frame(frame_bgr)
    snap_frames = SNAP_FRAME_COUNT

    lost_streak = 0
    last_recovery_time = time.time()
    last_good_pose = pose.copy()
    last_quality = 1.0
    recovery_escalated = False
    pos_filter.x_prev = None
    rot_filter.q_prev = None
    print("手動再ロック / GLOBAL 自動姿勢選択 / SNAPブースト")


# ─── Unity 通信 ──────────────────────────────────────────────────────────

unity_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if UNITY_UDP_ENABLED else None


def send_pose_to_unity(pose_mat, mode_text):
    if unity_udp_sock is None:
        return
    payload = json.dumps({
        "name": CLUSTER_MODEL_NAME,
        "timestamp": time.time(),
        "mode": mode_text,
        "pose_cv": pose_mat.astype(float).reshape(16).tolist(),
    }, separators=(",", ":")).encode("utf-8")
    unity_udp_sock.sendto(payload, (UNITY_UDP_HOST, UNITY_UDP_PORT))


command_socket = None
if UNITY_UDP_ENABLED:
    try:
        command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        command_socket.bind((UNITY_UDP_HOST, UNITY_COMMAND_PORT))
        command_socket.setblocking(False)
        print(f"Unity コマンド UDP 待受中: {UNITY_UDP_HOST}:{UNITY_COMMAND_PORT}")
    except OSError as e:
        command_socket = None
        print(f"Unity コマンド UDP 無効: {e}")


def stop_tracking_from_unity():
    global pose, tracking_active
    if tracking_active:
        pose = model.pose_cv.copy().astype(np.float32)
    tracking_active = False
    print("ADJUST モード / Unity から停止")


def poll_unity_commands(frame_bgr):
    if command_socket is None:
        return
    while True:
        try:
            data, _ = command_socket.recvfrom(1024)
        except BlockingIOError:
            break
        except OSError as e:
            print(f"Unity コマンド UDP エラー: {e}")
            break

        command = data.decode("utf-8", errors="ignore").strip().lower()
        if command == "start":
            if not tracking_active:
                start_tracking_from_frame(frame_bgr, use_rotation_guard=True)
        elif command == "global_start":
            if not tracking_active:
                start_tracking_from_frame(frame_bgr, use_rotation_guard=False)
        elif command == "stop":
            stop_tracking_from_unity()
        elif command == "quit":
            raise KeyboardInterrupt
        elif command:
            print(f"未知の Unity コマンド: {command}")


video_clients_lock = threading.Lock()
video_clients: list[socket.socket] = []
last_video_send_time = 0.0


def send_video_frame_to_unity(frame_bgr):
    global last_video_send_time
    if not UNITY_VIDEO_ENABLED:
        return
    now = time.time()
    if now - last_video_send_time < 1.0 / UNITY_VIDEO_FPS:
        return
    last_video_send_time = now

    with video_clients_lock:
        if not video_clients:
            return

    video_frame = frame_bgr
    if UNITY_VIDEO_WIDTH > 0 and frame_bgr.shape[1] > UNITY_VIDEO_WIDTH:
        scale = UNITY_VIDEO_WIDTH / frame_bgr.shape[1]
        video_frame = cv2.resize(frame_bgr, (UNITY_VIDEO_WIDTH, int(frame_bgr.shape[0] * scale)),
                                  interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", video_frame, [int(cv2.IMWRITE_JPEG_QUALITY), UNITY_VIDEO_JPEG_QUALITY])
    if not ok:
        return
    payload = encoded.tobytes()
    header = struct.pack(">I", len(payload))
    dead = []
    with video_clients_lock:
        for client in video_clients:
            try:
                client.sendall(header)
                client.sendall(payload)
            except OSError:
                dead.append(client)
        for client in dead:
            try:
                client.close()
            except OSError:
                pass
            video_clients.remove(client)


def unity_video_server_loop():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((UNITY_UDP_HOST, UNITY_VIDEO_PORT))
        server.listen(2)
        server.settimeout(0.5)
        print(f"Unity 動画 TCP 待受中: {UNITY_UDP_HOST}:{UNITY_VIDEO_PORT}")
        while True:
            try:
                client, _ = server.accept()
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with video_clients_lock:
                    video_clients.append(client)
                print("Unity 動画クライアント接続")
            except socket.timeout:
                continue
            except OSError as e:
                print(f"Unity 動画 TCP 停止: {e}")
                break
    except OSError as e:
        print(f"Unity 動画 TCP 無効: {e}")
    finally:
        try:
            server.close()
        except OSError:
            pass


if UNITY_VIDEO_ENABLED:
    threading.Thread(target=unity_video_server_loop, daemon=True).start()


# ─── CSV ロガー ──────────────────────────────────────────────────────────

os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"tracking_{time.strftime('%Y%m%d_%H%M%S')}.csv")
log_file = open(log_path, "w", newline="", encoding="utf-8")
log_writer = csv.writer(log_file)
log_writer.writerow([
    "frame", "timestamp", "mode", "quality",
    "tx", "ty", "tz",
    "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22",
])


def log_frame(frame_idx, mode_text, quality, pose_mat):
    R, t = pose_mat[:3, :3], pose_mat[:3, 3]
    log_writer.writerow([
        frame_idx, f"{time.time():.6f}", mode_text, f"{quality:.4f}",
        f"{t[0]:.6f}", f"{t[1]:.6f}", f"{t[2]:.6f}",
        f"{R[0,0]:.6f}", f"{R[0,1]:.6f}", f"{R[0,2]:.6f}",
        f"{R[1,0]:.6f}", f"{R[1,1]:.6f}", f"{R[1,2]:.6f}",
        f"{R[2,0]:.6f}", f"{R[2,1]:.6f}", f"{R[2,2]:.6f}",
    ])


# ─── メインループ ─────────────────────────────────────────────────────────

tracking_active = False
view_mode = 0
snap_frames = 0
frame_idx = 0
frame_read_failures = 0

print(f"ログ: {log_path}")
print("キー: Space=通常追跡開始(保護あり)/停止  g=全探索で追跡開始  ESC=終了")
print("      1〜9,0,-,= : 姿勢スロット読込  p=保存  r=選択スロットで再ロック  v=表示切替  t=自動再ロックON/OFF")
print("      a/d=X移動  q/e=Y移動  w/s=Z(奥行き)移動  i/k,j/l,u/o=回転（追跡停止中のみ）")

cv2.namedWindow("few_teeth_tracking", cv2.WINDOW_NORMAL)

while True:
    ret, raw = cap.read()
    if not ret:
        frame_read_failures += 1
        if frame_read_failures == 1:
            print("カメラ読み取り失敗。再試行中...")
        if frame_read_failures >= FRAME_READ_RETRY_LIMIT:
            print("カメラ読み取りが繰り返し失敗したため終了します")
            break
        time.sleep(0.05)
        continue
    if frame_read_failures > 0:
        print("カメラ読み取り復帰")
        frame_read_failures = 0

    frame_img = prepare_frame(raw)
    frame_idx += 1

    send_video_frame_to_unity(frame_img)
    poll_unity_commands(frame_img)

    key = cv2.waitKey(1) & 0xFF
    step_t = 0.005
    step_r = 2.0

    if key == 27:  # ESC
        break
    elif key == ord(" "):
        if tracking_active:
            stop_tracking_from_unity()
        else:
            start_tracking_from_frame(frame_img, use_rotation_guard=True)
    elif key == ord("g"):
        start_tracking_from_frame(frame_img, use_rotation_guard=False)
    elif key == ord("v"):
        view_mode = (view_mode + 1) % len(VIEW_MODES)
    elif key == ord("p"):
        saved_path = save_pose_slot(selected_pose_slot, pose)
        print(f"姿勢スロット {selected_pose_slot} を保存: {saved_path}")
    elif key == ord("t"):
        QUALITY_CHECK_ENABLED = not QUALITY_CHECK_ENABLED
        lost_streak = 0
        print(f"自動再ロック: {'ON' if QUALITY_CHECK_ENABLED else 'OFF'}")
    elif key == ord("["):
        current_exposure -= 1
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
        print(f"露出を下げました: 要求={current_exposure} 実際={cap.get(cv2.CAP_PROP_EXPOSURE)}")
    elif key == ord("]"):
        current_exposure += 1
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
        print(f"露出を上げました: 要求={current_exposure} 実際={cap.get(cv2.CAP_PROP_EXPOSURE)}")
    # 手動ナッジ中は reset_pose_cv を呼ばない。毎回モデル側に反映しようと
    # するとレンダラの内部状態と衝突してクラッシュすることがある。ローカル
    # の pose 変数だけ更新し、表示用ワイヤーフレームはそれを直接使う。
    elif not tracking_active and key == ord("a"):
        pose[0, 3] -= step_t
    elif not tracking_active and key == ord("d"):
        pose[0, 3] += step_t
    elif not tracking_active and key == ord("q"):
        pose[1, 3] -= step_t
    elif not tracking_active and key == ord("e"):
        pose[1, 3] += step_t
    elif not tracking_active and key == ord("w"):
        pose[2, 3] -= step_t
    elif not tracking_active and key == ord("s"):
        pose[2, 3] += step_t
    elif not tracking_active and key == ord("i"):
        pose[:3, :3] = rot_x(step_r) @ pose[:3, :3]
    elif not tracking_active and key == ord("k"):
        pose[:3, :3] = rot_x(-step_r) @ pose[:3, :3]
    elif not tracking_active and key == ord("j"):
        pose[:3, :3] = rot_y(step_r) @ pose[:3, :3]
    elif not tracking_active and key == ord("l"):
        pose[:3, :3] = rot_y(-step_r) @ pose[:3, :3]
    elif not tracking_active and key == ord("u"):
        pose[:3, :3] = rot_z(step_r) @ pose[:3, :3]
    elif not tracking_active and key == ord("o"):
        pose[:3, :3] = rot_z(-step_r) @ pose[:3, :3]
    elif key == ord("r"):
        if tracking_active:
            relock_from_frame(frame_img)
        else:
            loaded_pose, loaded_path = load_pose_slot(selected_pose_slot)
            if loaded_pose is not None:
                pose = loaded_pose
                model.reset_pose_cv(pose)
                print(f"姿勢スロット {selected_pose_slot} を読み込み: {loaded_path}")
            else:
                print(f"姿勢スロット {selected_pose_slot} は未保存")
    elif key in POSE_SLOT_KEYS:
        selected_pose_slot = POSE_SLOT_KEYS[key]
        loaded_pose, loaded_path = load_pose_slot(selected_pose_slot)
        if loaded_pose is not None:
            pose = loaded_pose
            model.reset_pose_cv(pose)
            if tracking_active:
                relock_from_frame(frame_img)
            print(f"姿勢スロット {selected_pose_slot} を読み込み: {loaded_path}")
        else:
            print(f"姿勢スロット {selected_pose_slot} は未保存")

    if tracking_active:
        frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

        if snap_frames > 0:
            for _ in range(SNAP_FRAME_STEPS):
                tracker.step(frame_rgb)
            snap_frames -= 1
            mode_text = f"SNAP {snap_frames}"
        else:
            for _ in range(TRACKING_STEPS_PER_FRAME):
                tracker.step(frame_rgb)
            mode_text = "TRACKING"

        pose = model.pose_cv.copy().astype(np.float32)

        if snap_frames == 0:
            dist_map = compute_edge_distance_map(frame_img)
            last_quality = estimate_tracking_quality(dist_map, pose)

            if last_quality >= QUALITY_GOOD_THRESHOLD:
                lost_streak = 0
                last_good_pose = pose.copy()
                recovery_escalated = False
            elif last_quality < QUALITY_LOSS_THRESHOLD:
                lost_streak += 1
            # デッドゾーン内はストリークを維持する

            if QUALITY_CHECK_ENABLED:
                now = time.time()
                if (lost_streak >= QUALITY_LOSS_CONSECUTIVE
                        and now - last_recovery_time >= RECOVERY_COOLDOWN_SEC):
                    use_guard = last_good_pose is not None and not recovery_escalated
                    print(f"追跡品質低下 ({last_quality:.2f} < {QUALITY_LOSS_THRESHOLD}) → "
                          f"自動再ロック (guard={'on' if use_guard else 'off'})")
                    auto_pose, auto_slot = pose_auto_select(
                        frame_img, dist_map, use_rotation_guard=use_guard,
                        reference_pose=last_good_pose if use_guard else None,
                    )
                    pose = auto_pose
                    if auto_slot is not None:
                        selected_pose_slot = auto_slot
                    last_recovery_time = now
                    lost_streak = 0
                    recovery_escalated = use_guard
                    snap_frames = SNAP_FRAME_COUNT

        log_frame(frame_idx, mode_text, last_quality, pose)

        if view_mode == 0:
            show = viewer.view_color(False)
        elif view_mode == 1:
            show = draw_obj_wireframe(frame_img, pose)
        else:
            rendered = viewer.view_color(False)
            wire = draw_obj_wireframe(frame_img, pose)
            show = cv2.addWeighted(rendered, 0.45, wire, 0.55, 0)

        quality_color = (0, 255, 0) if last_quality >= QUALITY_GOOD_THRESHOLD else (0, 165, 255)
        cv2.putText(show, f"FIT QUALITY: {last_quality:.2f}  lost_streak: {lost_streak}",
                    (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)

        # 品質が悪いフレームは last_good_pose を送り続け、Unity 側のモデルが
        # 飛ばないようにする（この判定はプレビュー表示や FIT QUALITY 自体には
        # 影響しない。送信内容だけをフリーズさせる）。
        pose_to_smooth = pose
        if last_good_pose is not None and last_quality < QUALITY_LOSS_THRESHOLD:
            pose_to_smooth = last_good_pose
        display_pose = smooth_pose(pose_to_smooth)
    else:
        show = draw_obj_wireframe(frame_img, pose)
        mode_text = "ADJUST"
        display_pose = pose

    send_pose_to_unity(display_pose, mode_text)

    cv2.putText(show, f"MODE: {mode_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(show, f"VIEW: {VIEW_MODES[view_mode]}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(show, f"SLOT: {selected_pose_slot}  MODEL: {CLUSTER_MODEL_NAME}", (20, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
    if not tracking_active:
        cv2.putText(show, "a/d q/e w/s=move  i/k j/l u/o=rotate  p=save  Space=track(safe)  g=track(global)  ESC=quit",
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    cv2.imshow("few_teeth_tracking", show)

cap.release()
log_file.close()
if unity_udp_sock is not None:
    unity_udp_sock.close()
if command_socket is not None:
    command_socket.close()
with video_clients_lock:
    for client in video_clients:
        try:
            client.close()
        except OSError:
            pass
    video_clients.clear()
cv2.destroyAllWindows()
print(f"終了。ログ: {log_path}")
