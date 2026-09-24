"""marker_reference.py — ArUco マーカーを「定規」にしてマーカーレス追跡の精度を測る。

使い方（tracker_four_group.py から）:
    ref = MarkerReference(K, dist=None, registration_path=...)
    det = ref.detect(frame_bgr)          # 毎フレーム。検出できなければ None
    ref.register(T_cam_model)            # 追跡が合っていると判断した瞬間に 'm' で一度だけ
    err = ref.error(T_cam_model)         # 以降毎フレーム: {"mm":..., "deg":..., "predicted": 4x4}

考え方:
    マーカー（台座に貼付、歯列と剛体）の姿勢 T_cam_ref はフレームごとに直接推定できる。
    一度だけ「合っている」追跡姿勢 T_cam_model を使って T_ref_model = inv(T_cam_ref) @ T_cam_model
    を求めて保存すれば、以後は T_cam_ref @ T_ref_model が「マーカーから予測した歯牙姿勢」になり、
    マーカーレス追跡の姿勢との差が mm / 度で出る。
    追跡側はマーカーを一切使わないので、評価はマーカーレスのまま成立する。

マーカーは markers/aruco_sheet_A4.pdf（DICT_4X4_50）を 100% で印刷したものを前提とする。
    id 0-3 : 30 mm 単体      id 4-9 : 20 mm 単体      id 10-17 : 15 mm 単体
    id 20-23: 2x2 GridBoard（20 mm, 間隔 5 mm）     id 24-29: 3x2 GridBoard（15 mm, 間隔 4 mm）
ボードは複数マーカーをまとめて 1 つの PnP で解くので単体より安定する。印刷倍率が 100% で
なかった場合は scale に実測/設計の比を渡す（例: 49 mm/50 mm → 0.98）。
"""
import json
import os

import cv2
import numpy as np

SINGLE_SIZES_M = {}
SINGLE_SIZES_M.update({i: 0.030 for i in range(0, 4)})
SINGLE_SIZES_M.update({i: 0.020 for i in range(4, 10)})
SINGLE_SIZES_M.update({i: 0.015 for i in range(10, 18)})

BOARD_SPECS = {
    "board2x2": dict(size=(2, 2), marker=0.020, gap=0.005, ids=[20, 21, 22, 23]),
    "board3x2": dict(size=(3, 2), marker=0.015, gap=0.004, ids=[24, 25, 26, 27, 28, 29]),
}


def _to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, np.float64).ravel()
    return T


def rotation_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def average_transforms(Ts):
    """4x4 剛体変換の平均。並進は算術平均、回転は R_i の総和を SVD で直交化（chordal 平均）。"""
    Ts = [np.asarray(T, np.float64) for T in Ts]
    M = sum(T[:3, :3] for T in Ts)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return out


def average_transforms_robust(Ts, min_tol_mm=3.0, k=2.0, min_keep=2):
    """外れ値に強い平均。全サンプルの平均からの距離の中央値 d_med を取り、
    max(min_tol_mm, k*d_med) を超えるサンプルを除外して平均し直す（1 回反復）。
    ずれた追跡や粗い PnP のまま 'm' を押した登録が混ざっても、平均が引きずられないようにする。
    返り値: (平均 4x4, 採用したサンプルの index リスト)"""
    Ts = [np.asarray(T, np.float64) for T in Ts]
    if len(Ts) < 3:
        return average_transforms(Ts), list(range(len(Ts)))
    M0 = average_transforms(Ts)
    d = np.array([np.linalg.norm(T[:3, 3] - M0[:3, 3]) * 1000.0 for T in Ts])
    tol = max(min_tol_mm, k * float(np.median(d)))
    keep = [i for i in range(len(Ts)) if d[i] <= tol]
    if len(keep) < min_keep:
        keep = list(np.argsort(d)[:min_keep])
    M1 = average_transforms([Ts[i] for i in keep])
    # 2 回目: 改善した平均に対して再判定（外れ値が平均を引っ張っていた場合に効く）
    d1 = np.array([np.linalg.norm(T[:3, 3] - M1[:3, 3]) * 1000.0 for T in Ts])
    tol1 = max(min_tol_mm, k * float(np.median(d1[keep])))
    keep2 = [i for i in range(len(Ts)) if d1[i] <= tol1]
    if len(keep2) < min_keep:
        keep2 = keep
    return average_transforms([Ts[i] for i in keep2]), keep2


def transform_spread(Ts, T_mean):
    """各サンプルの平均からのずれ（mm, deg）。"""
    d_mm = [float(np.linalg.norm(np.asarray(T)[:3, 3] - T_mean[:3, 3]) * 1000.0) for T in Ts]
    d_deg = [rotation_deg(np.asarray(T)[:3, :3], T_mean[:3, :3]) for T in Ts]
    return d_mm, d_deg


class MarkerReference:
    def __init__(self, K, dist=None, registration_path=None, scale=1.0,
                 dictionary=cv2.aruco.DICT_4X4_50):
        self.K = np.asarray(K, np.float64)
        self.dist = np.zeros((5, 1)) if dist is None else np.asarray(dist, np.float64).reshape(-1, 1)
        self.adict = cv2.aruco.getPredefinedDictionary(dictionary)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.adict, params)
        self.boards = {
            name: cv2.aruco.GridBoard(spec["size"], spec["marker"] * scale, spec["gap"] * scale,
                                      self.adict, np.array(spec["ids"], dtype=np.int32))
            for name, spec in BOARD_SPECS.items()
        }
        self.board_ids = {name: set(spec["ids"]) for name, spec in BOARD_SPECS.items()}
        self.single_sizes = {i: s * scale for i, s in SINGLE_SIZES_M.items()}
        self.registration_path = registration_path
        self.T_ref_model = {}   # key -> 4x4（外れ値を除いたサンプルの平均）
        self.samples = {}       # key -> [4x4, ...]（'m' を押すたびに追加。視点を変えて複数回登録して平均する）
        self.kept = {}          # key -> 平均に採用したサンプルの index（外れ値は除外される）
        self.last = None        # 直近フレームの検出結果
        self.load()

    # ── 検出 ────────────────────────────────────────────────────────────────
    def detect(self, frame_bgr, detect_max_width=800):
        """マーカー検出。幅が detect_max_width を超える画像は縮小して検出し、コーナーだけ
        フル解像度でサブピクセル精緻化する（720p でフル解像度検出すると 20〜40 ms かかる）。"""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        scale = 1.0
        if gray.shape[1] > detect_max_width:
            scale = detect_max_width / float(gray.shape[1])
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = gray
        corners, ids, _ = self.detector.detectMarkers(small)
        self.last = None
        if ids is None or len(ids) == 0:
            return None
        ids = ids.flatten()
        if scale != 1.0:
            # 縮小画像のコーナーをフル解像度座標へ戻し、元画像でサブピクセル精緻化
            corners = [c / scale for c in corners]
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
            corners = [cv2.cornerSubPix(gray, c.reshape(-1, 1, 2).astype(np.float32), (5, 5), (-1, -1), crit).reshape(1, -1, 2)
                       for c in corners]

        best = None
        # 1) ボード（2 枚以上見えているものを優先。マーカー数が多いほど良い）
        for name, board in self.boards.items():
            sel = [i for i, mid in enumerate(ids) if mid in self.board_ids[name]]
            if len(sel) < 2:
                continue
            obj_pts, img_pts = board.matchImagePoints([corners[i] for i in sel],
                                                      ids[sel].reshape(-1, 1))
            if obj_pts is None or len(obj_pts) < 8:
                continue
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                continue
            rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, self.K, self.dist, rvec, tvec)
            cand = self._make_result(name, rvec, tvec, obj_pts, img_pts,
                                     [corners[i] for i in sel], ids[sel])
            if best is None or cand["n_markers"] > best["n_markers"]:
                best = cand

        # 2) ボードが無ければ、既知サイズの単体マーカーのうち画像上で最大のもの
        if best is None:
            for i, mid in enumerate(ids):
                size = self.single_sizes.get(int(mid))
                if size is None:
                    continue
                h = size / 2.0
                obj_pts = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
                img_pts = corners[i].reshape(-1, 2).astype(np.float32)
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.dist,
                                              flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    continue
                cand = self._make_result(f"id{int(mid)}", rvec, tvec, obj_pts, img_pts,
                                         [corners[i]], ids[i:i + 1])
                if best is None or cand["perimeter_px"] > best["perimeter_px"]:
                    best = cand

        self.last = best
        return best

    def _make_result(self, key, rvec, tvec, obj_pts, img_pts, corners, ids):
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.dist)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - np.asarray(img_pts).reshape(-1, 2)) ** 2, axis=1))))
        perim = float(sum(cv2.arcLength(c.reshape(-1, 1, 2).astype(np.float32), True) for c in corners))
        return dict(key=key, T=_to_T(rvec, tvec), rvec=rvec, tvec=tvec, n_markers=len(corners),
                    rms_px=rms, perimeter_px=perim, corners=corners, ids=np.asarray(ids).reshape(-1, 1))

    # ── 登録・予測・誤差 ─────────────────────────────────────────────────────
    def register(self, T_cam_model):
        """今見えている基準（ボード/マーカー）に対する歯牙モデルの固定変換をサンプルとして追加し、
        そのキーの全サンプルの平均を基準として採用する。
        PnP クリック初期化の偶発誤差を減らすため、視点・距離を変えて複数回登録するのが前提。
        （モデル形状やカメラ内部パラメータの系統誤差は平均しても残る）"""
        if self.last is None:
            return None
        key = self.last["key"]
        T = np.linalg.inv(self.last["T"]) @ np.asarray(T_cam_model, np.float64)
        self.samples.setdefault(key, []).append(T)
        self._recompute(key)
        self.save()
        return key

    def _recompute(self, key):
        Ts = self.samples.get(key, [])
        if not Ts:
            self.T_ref_model.pop(key, None)
            self.kept.pop(key, None)
            return
        self.T_ref_model[key], self.kept[key] = average_transforms_robust(Ts)

    def undo_last_sample(self):
        """直近に追加したサンプルを 1 つ取り消す（今見えているキーについて）。"""
        if self.last is None:
            return None
        key = self.last["key"]
        if not self.samples.get(key):
            return None
        self.samples[key].pop()
        if not self.samples[key]:
            self.samples.pop(key, None)
        self._recompute(key)
        self.save()
        return key

    def summary(self, key=None):
        """登録サンプル数（全体 n / 採用 n_used）と、採用サンプルの平均からのばらつき（最大・RMS）。"""
        key = key or (self.last["key"] if self.last is not None else None)
        if key is None or key not in self.T_ref_model:
            return None
        Ts = self.samples.get(key, [self.T_ref_model[key]])
        kept = self.kept.get(key, list(range(len(Ts))))
        used = [Ts[i] for i in kept] if kept else Ts
        d_mm, d_deg = transform_spread(used, self.T_ref_model[key])
        return dict(key=key, n=len(Ts), n_used=len(used), max_mm=max(d_mm), max_deg=max(d_deg),
                    rms_mm=float(np.sqrt(np.mean(np.square(d_mm)))), rms_deg=float(np.sqrt(np.mean(np.square(d_deg)))))

    def clear(self):
        """登録を全消去する。誤操作に備えて、消す前の JSON をタイムスタンプ付きで残す。戻り値: バックアップパス or None"""
        bak = None
        if self.registration_path and os.path.exists(self.registration_path) and self.samples:
            import shutil, time
            bak = self.registration_path[:-5] + time.strftime("_cleared_%Y%m%d_%H%M%S.json.bak")
            shutil.copy(self.registration_path, bak)
        self.T_ref_model = {}
        self.samples = {}
        self.kept = {}
        self.save()
        return bak

    def is_registered(self):
        return self.last is not None and self.last["key"] in self.T_ref_model

    def predict(self):
        if not self.is_registered():
            return None
        return self.last["T"] @ self.T_ref_model[self.last["key"]]

    def error(self, T_cam_model):
        P = self.predict()
        if P is None:
            return None
        T = np.asarray(T_cam_model, np.float64)
        d = T[:3, 3] - P[:3, 3]
        return dict(mm=float(np.linalg.norm(d) * 1000.0), dx_mm=d * 1000.0,
                    deg=rotation_deg(P[:3, :3], T[:3, :3]), predicted=P, key=self.last["key"],
                    rms_px=self.last["rms_px"], n_markers=self.last["n_markers"],
                    n_samples=len(self.samples.get(self.last["key"], [])))

    # ── 追跡器向けにマーカー領域を消す ───────────────────────────────────────
    def mask_out(self, img, pad_px=12, method="inpaint"):
        """検出したマーカー群の凸包（pad_px 拡張）を塗り消した画像を返す。
        基準用ボードの強いエッジが region/edge 追跡や品質評価を引き込まないようにするため。
        マーカー検出自体は元画像で済ませてから呼ぶこと。"""
        if self.last is None:
            return img
        H, W = img.shape[:2]
        pts = np.vstack([c.reshape(-1, 2) for c in self.last["corners"]]).astype(np.int32)
        hull = cv2.convexHull(pts)
        # 処理はボード周辺の ROI に限定する（フル画面の np.where / dilate は 720p で 10 ms 超）
        pad = int(pad_px)
        x0 = max(0, int(pts[:, 0].min()) - pad - 4); x1 = min(W, int(pts[:, 0].max()) + pad + 5)
        y0 = max(0, int(pts[:, 1].min()) - pad - 4); y1 = min(H, int(pts[:, 1].max()) + pad + 5)
        if x1 <= x0 or y1 <= y0:
            return img
        roi = img[y0:y1, x0:x1]
        mask = np.zeros(roi.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, hull - np.array([[x0, y0]], np.int32), 255)
        if pad > 0:
            # 楕円カーネル 101x101 は 20 ms かかる（非分離）。矩形は 0.4 ms。角の丸みは不要。
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * pad + 1, 2 * pad + 1)))
        out = img.copy()
        if method == "inpaint":
            out[y0:y1, x0:x1] = cv2.inpaint(roi, mask, 3, cv2.INPAINT_TELEA)
            return out
        if method == "blur":
            # 縮小してぼかしてから戻す（縮小率 1/4 でぼかし半径は実質 4 倍）
            small = cv2.resize(roi, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            blurred = cv2.resize(cv2.GaussianBlur(small, (15, 15), 0), (roi.shape[1], roi.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
            roi_out = roi.copy()
            cv2.copyTo(blurred, mask, roi_out)
            out[y0:y1, x0:x1] = roi_out
            return out
        # "fill": 周囲の中央値色で塗る
        fill = np.median(roi[mask == 0], axis=0).astype(np.uint8) if (mask == 0).any() else np.array([128, 128, 128], np.uint8)
        roi_out = roi.copy(); roi_out[mask > 0] = fill
        out[y0:y1, x0:x1] = roi_out
        return out

    # ── 描画 ────────────────────────────────────────────────────────────────
    def draw(self, img, axis_len=0.02):
        if self.last is None:
            return
        cv2.aruco.drawDetectedMarkers(img, self.last["corners"], self.last["ids"])
        cv2.drawFrameAxes(img, self.K, self.dist, self.last["rvec"], self.last["tvec"], axis_len, 2)

    # ── 保存・読込 ───────────────────────────────────────────────────────────
    def save(self):
        if not self.registration_path:
            return
        os.makedirs(os.path.dirname(self.registration_path), exist_ok=True)
        with open(self.registration_path, "w", encoding="utf-8") as f:
            json.dump({"version": 2,
                       "mean": {k: v.tolist() for k, v in self.T_ref_model.items()},
                       "samples": {k: [T.tolist() for T in Ts] for k, Ts in self.samples.items()}},
                      f, indent=1)

    def load(self):
        if not self.registration_path or not os.path.exists(self.registration_path):
            return
        with open(self.registration_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("version") == 2:
            self.samples = {k: [np.asarray(T, np.float64) for T in Ts] for k, Ts in data.get("samples", {}).items()}
        else:
            # 旧形式（key -> 4x4）: 1 サンプル扱いで読み込む
            self.samples = {k: [np.asarray(v, np.float64)] for k, v in data.items()}
        for k in list(self.samples):
            self._recompute(k)
