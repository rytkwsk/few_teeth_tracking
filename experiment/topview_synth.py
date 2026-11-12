# -*- coding: utf-8 -*-
"""斜めのカメラ映像から、目標の歯を真上（歯の長軸方向）から見た映像を幾何的に作る試作（2026-11-12）。

先生の指摘（上からの視点は AI で仮想的に作れる）に対して、まず AI を使わない方法で、何が見えて何が見えないかを確かめる。
  1. 追跡姿勢（F = 歯列全体）で歯のメッシュをカメラ座標に置く
  2. 真上の正射影カメラから歯の表面点を求める（z バッファ）
  3. その点が実カメラから見えるか（歯列全体の z バッファで隠れていないか・面が横を向きすぎていないか）を判定し、
     見える点だけ映像の色を取る。見えない点は灰色（推測で埋めない）
出力は 1 フレームにつき [入力（歪み補正後）に輪郭を重ねたもの | 映像から作った上面図 | CG の上面図] の 3 枚並び。

    python experiment/topview_synth.py experiment/sessions/topview_20261112/clip01            # 録画クリップ
    python experiment/topview_synth.py --log logs/tracking_20261112_123331.csv --cg-only        # 映像なし（姿勢と CG だけ）

姿勢は録画に入っていないので、トラッカーの CSV（logs/tracking_*.csv）から、フレームの撮影時刻の直後に書かれた行を使う
（行の時刻は処理後の time.time()。静止〜ゆっくりの動きなら数十 ms の差は無視できる）。
メッシュの座標は OBJ のまま（1 = 25 mm）。トラッカーと同じく MODEL_UNIT_IN_METER を掛けて m にする。Unity の x 反転は関係ない。
"""
import argparse
import bisect
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
UNITY_TARGETS = os.path.join(os.path.dirname(REPO), "unity-few-teeth-tracking", "Assets", "Resources", "Targets")
MODEL_UNIT_IN_METER = 0.025   # tracker_four_group.py と同じ

TOP_MM_PER_PX = 0.04          # 上面図の解像度
TOP_HALF_MM = 7.5             # 上面図の半幅（歯の中心から）
VIS_DEPTH_TOL_MM = 0.8        # 実カメラの z バッファとの差がこれ以下なら「見えている」（斜め 60° だと 1 画素で奥行きが ~0.3 mm 変わる）
MIN_COS = 0.20                # 面の法線とカメラ方向の cos がこれ未満は横を向きすぎ（引き伸ばされる）ので使わない
UNSEEN_GRAY = (110, 110, 110)


# ─── メッシュ ────────────────────────────────────────────────────────────────

def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, np.float64) * MODEL_UNIT_IN_METER, np.asarray(faces, np.int64)


def vertex_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], n)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


@njit(cache=True)
def raster(xy, z, F, W, H):
    """xy: 画素座標, z: 小さいほど手前。z バッファ・面番号・重心座標 (b1, b2) を返す（z は画面上で線形補間）。"""
    zb = np.full((H, W), np.inf)
    fid = np.full((H, W), -1, np.int64)
    b1 = np.zeros((H, W))
    b2 = np.zeros((H, W))
    for f in range(F.shape[0]):
        i0, i1, i2 = F[f, 0], F[f, 1], F[f, 2]
        x0, y0, x1, y1, x2, y2 = xy[i0, 0], xy[i0, 1], xy[i1, 0], xy[i1, 1], xy[i2, 0], xy[i2, 1]
        if not (np.isfinite(z[i0]) and np.isfinite(z[i1]) and np.isfinite(z[i2])):
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        xmin = max(int(np.floor(min(x0, x1, x2))), 0)
        xmax = min(int(np.ceil(max(x0, x1, x2))), W - 1)
        ymin = max(int(np.floor(min(y0, y1, y2))), 0)
        ymax = min(int(np.ceil(max(y0, y1, y2))), H - 1)
        for py in range(ymin, ymax + 1):
            for px in range(xmin, xmax + 1):
                cx, cy = px + 0.5, py + 0.5
                l0 = ((y1 - y2) * (cx - x2) + (x2 - x1) * (cy - y2)) / den
                l1 = ((y2 - y0) * (cx - x2) + (x0 - x2) * (cy - y2)) / den
                l2 = 1.0 - l0 - l1
                if l0 < 0 or l1 < 0 or l2 < 0:
                    continue
                d = l0 * z[i0] + l1 * z[i1] + l2 * z[i2]
                if d < zb[py, px]:
                    zb[py, px] = d
                    fid[py, px] = f
                    b1[py, px] = l1
                    b2[py, px] = l2
    return zb, fid, b1, b2


def interp(A, F, fid, b1, b2):
    """面番号と重心座標から頂点属性 A を画素ごとに補間する（当たらない画素は nan）。"""
    out = np.full(fid.shape + (A.shape[1],), np.nan)
    m = fid >= 0
    f = F[fid[m]]
    l1, l2 = b1[m][:, None], b2[m][:, None]
    out[m] = (1 - l1 - l2) * A[f[:, 0]] + l1 * A[f[:, 1]] + l2 * A[f[:, 2]]
    return out


# ─── 入力 ───────────────────────────────────────────────────────────────────

def load_calibration(path, w, h):
    c = np.load(path)
    K, dist = c["K"].astype(np.float64), c["dist"].astype(np.float64)
    assert int(c["image_width"][0]) == w and int(c["image_height"][0]) == h, "キャリブの解像度が映像と違う"
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))   # トラッカーと同じ（alpha=0）
    maps = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)
    return newK, maps


def load_log(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for d in csv.DictReader(f):
            if not d["mode"].startswith("TRACKING") or not d["tx"]:
                continue
            R = np.array([float(d[f"r{i}{j}"]) for i in range(3) for j in range(3)]).reshape(3, 3)
            t = np.array([float(d[k]) for k in ("tx", "ty", "tz")])
            rows.append((float(d["timestamp"]), R, t, d["mode"], float(d["quality"]),
                         float(d["err_mm"]) if d.get("err_mm") else float("nan")))
    return rows


def find_log_for(ts0, ts1):
    for p in sorted(glob.glob(os.path.join(REPO, "logs", "tracking_*.csv")), reverse=True):
        with open(p, encoding="utf-8") as f:
            r = csv.DictReader(f)
            first = next(r, None)
        if first and float(first["timestamp"]) <= ts0 + 1.0:
            return p
    return None


def load_target(tooth):
    with open(os.path.join(UNITY_TARGETS, f"{tooth}.json"), encoding="utf-8") as f:
        d = json.load(f)
    s = MODEL_UNIT_IN_METER
    tgt = dict(access=np.array(d["access"]) * s, mesial=np.array(d["mesial"]), buccal=np.array(d["buccal"]),
               outline=np.array(d["outline"]).reshape(-1, 3) * s,
               canals=[(c["name"], np.array(c["orifice"]) * s, np.array(c["entry"]) * s) for c in d["canals"]],
               depth_mm=d["depth_mm"])
    e = tgt["canals"][0]
    up = e[2] - e[1]                     # 根管口 → 切削開始点 = 歯の長軸（咬合面側が +）
    tgt["up"] = up / np.linalg.norm(up)
    return tgt


# ─── 上面図 ─────────────────────────────────────────────────────────────────

class TopView:
    """歯の長軸に沿って真上から見る正射影。画面の上 = 実カメラの視線を咬合面に投影した向き（Unity の小窓と同じ）。"""

    def __init__(self, tgt, Vt, Ft, R_cam):
        self.c = tgt["access"]
        self.up = tgt["up"]
        fwd = R_cam.T @ np.array([0.0, 0.0, 1.0])            # カメラの視線（モデル座標）
        fwd -= self.up * np.dot(fwd, self.up)
        if np.linalg.norm(fwd) < 1e-6:                          # 真上から見ているときは近心を上に
            fwd = tgt["mesial"].copy()
        self.ey = fwd / np.linalg.norm(fwd)                    # 画面の上
        self.ex = np.cross(self.ey, self.up)                   # 画面の右（上から見て右手系）
        self.ex /= np.linalg.norm(self.ex)
        self.n = int(round(2 * TOP_HALF_MM / TOP_MM_PER_PX))
        xy, h = self.to_px(Vt)
        self.zb, self.fid, self.b1, self.b2 = raster(xy, -h, Ft, self.n, self.n)
        self.P = interp(Vt, Ft, self.fid, self.b1, self.b2)     # 画素ごとの表面点（モデル座標, m）
        self.N = interp(vertex_normals(Vt, Ft), Ft, self.fid, self.b1, self.b2)

    def to_px(self, P):
        d = np.atleast_2d(P) - self.c
        s = 1000.0 / TOP_MM_PER_PX
        x = d @ self.ex * s + self.n / 2
        y = -(d @ self.ey) * s + self.n / 2
        return np.stack([x, y], 1), d @ self.up

    def shaded(self):
        """CG の上面図（法線の陰影）。"""
        img = np.full((self.n, self.n, 3), 40, np.uint8)
        m = self.fid >= 0
        light = self.up * 0.8 + self.ey * 0.3 - self.ex * 0.3
        light /= np.linalg.norm(light)
        s = np.clip(self.N[m] @ light, 0, 1) * 0.75 + 0.25
        img[m] = (np.stack([s * 235, s * 225, s * 215], 1)).astype(np.uint8)
        return img


def synth(top, R, t, K, frame, cam_zb):
    """映像から上面図を作る。戻り値: 画像, 使えた画素の割合（歯の表面のうち）, 見えている/見えていないのマスク。"""
    h_img, w_img = frame.shape[:2]
    m = top.fid >= 0
    P = top.P[m]
    Pc = P @ R.T + t
    uv = Pc @ K.T
    u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
    inside = (u >= 0) & (u < w_img - 1) & (v >= 0) & (v < h_img - 1) & (Pc[:, 2] > 0)
    ui = np.clip(u.astype(int), 0, w_img - 1)
    vi = np.clip(v.astype(int), 0, h_img - 1)
    not_hidden = np.abs(cam_zb[vi, ui] - Pc[:, 2]) * 1000.0 < VIS_DEPTH_TOL_MM
    view = -Pc / np.linalg.norm(Pc, axis=1, keepdims=True)
    cosang = np.sum((top.N[m] @ R.T) * view, 1)
    ok = inside & not_hidden & (cosang > MIN_COS)

    img = np.zeros((top.n, top.n, 3), np.uint8)
    img[:] = 40
    mapx = np.full((top.n, top.n), -1, np.float32)
    mapy = np.full((top.n, top.n), -1, np.float32)
    ys, xs = np.nonzero(m)
    mapx[ys[ok], xs[ok]] = u[ok]
    mapy[ys[ok], xs[ok]] = v[ok]
    sampled = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    seen = np.zeros((top.n, top.n), bool)
    seen[ys[ok], xs[ok]] = True
    unseen = m & ~seen
    img[seen] = sampled[seen]
    img[unseen] = UNSEEN_GRAY
    hatch = ((np.add.outer(np.arange(top.n), np.arange(top.n)) // 6) % 2 == 0)
    img[unseen & hatch] = (90, 90, 90)
    cosmap = np.zeros((top.n, top.n))
    cosmap[ys, xs] = cosang
    return img, seen, unseen, cosmap


# ─── 描画 ───────────────────────────────────────────────────────────────────

def draw_target_top(img, top, tgt, scale):
    """上面図に、髄腔開拡の輪郭・根管口・M/D/B/L・5 mm の目盛りを重ねる。"""
    def P(p):
        xy, _ = top.to_px(p)
        return (xy[0] * scale).astype(int)
    pts = np.array([P(p) for p in tgt["outline"]], np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (60, 220, 60), 2, cv2.LINE_AA)
    for name, o, _ in tgt["canals"]:
        q = tuple(P(o))
        cv2.circle(img, q, 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, q, 4, (60, 220, 60), -1, cv2.LINE_AA)
        cv2.putText(img, name, (q[0] + 8, q[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, name, (q[0] + 8, q[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    c = np.array(P(tgt["access"]), float)
    r = top.n * scale * 0.44
    for lab, d in (("M", tgt["mesial"]), ("D", -tgt["mesial"]), ("B", tgt["buccal"]), ("L", -tgt["buccal"])):
        dx, dy = np.dot(d, top.ex), -np.dot(d, top.ey)
        q = (int(c[0] + dx * r) - 6, int(c[1] + dy * r) + 6)
        cv2.putText(img, lab, q, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, lab, q, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    L = int(5.0 / TOP_MM_PER_PX * scale)
    y0 = img.shape[0] - 14
    cv2.line(img, (12, y0), (12 + L, y0), (255, 255, 255), 3)
    cv2.putText(img, "5 mm", (12, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_target_cam(img, R, t, K, tgt, Vt, Ft):
    def proj(P):
        Pc = np.atleast_2d(P) @ R.T + t
        uv = Pc @ K.T
        return (uv[:, :2] / uv[:, 2:]).astype(int)
    pts = proj(tgt["outline"]).astype(np.int32)
    cv2.polylines(img, [pts], True, (60, 220, 60), 2, cv2.LINE_AA)
    for name, o, e in tgt["canals"]:
        a, b = proj(np.stack([e, o]))
        cv2.line(img, tuple(a), tuple(b), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(b), 3, (0, 200, 255), -1, cv2.LINE_AA)


def label(img, text, y=22):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def render_frame(frame, R, t, K, tgt, Vt, Ft, Va, Fa, info="", out_size=420, crop=True):
    h_img, w_img = frame.shape[:2]
    # 実カメラの z バッファ（歯列全体。隣の歯による隠れも判定する）
    Pc = Va @ R.T + t
    uv = Pc @ K.T
    cam_zb, _, _, _ = raster(uv[:, :2] / uv[:, 2:], Pc[:, 2], Fa, w_img, h_img)

    top = TopView(tgt, Vt, Ft, R)
    syn, seen, unseen, cosmap = synth(top, R, t, K, frame, cam_zb)
    cg = top.shaded()

    # 目標の歯の表面のうち、髄腔開拡の輪郭の内側でどれだけ見えているか
    poly, _ = top.to_px(tgt["outline"])
    inner = np.zeros((top.n, top.n), np.uint8)
    cv2.fillPoly(inner, [poly.astype(np.int32)], 1)
    inner = inner.astype(bool) & (top.fid >= 0)
    cover_access = float(seen[inner].mean()) if inner.any() else 0.0
    cover_tooth = float(seen[top.fid >= 0].mean())

    scale = out_size / top.n
    syn = cv2.resize(syn, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    cg = cv2.resize(cg, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    draw_target_top(syn, top, tgt, scale)
    draw_target_top(cg, top, tgt, scale)

    view = frame.copy()
    draw_target_cam(view, R, t, K, tgt, Vt, Ft)
    if crop:   # 歯のまわりを切り出して並べる
        c = (np.atleast_2d(tgt["access"]) @ R.T + t) @ K.T
        cx, cy = int(c[0, 0] / c[0, 2]), int(c[0, 1] / c[0, 2])
        half = 170
        x0, y0 = np.clip(cx - half, 0, w_img - 2 * half), np.clip(cy - half, 0, h_img - 2 * half)
        view = view[y0:y0 + 2 * half, x0:x0 + 2 * half]
    view = cv2.resize(view, (out_size, out_size))

    # カメラの傾き（歯の長軸とカメラ視線のなす角）
    ray = (R @ tgt["access"] + t)
    ray /= np.linalg.norm(ray)
    tilt = float(np.degrees(np.arccos(np.clip(-np.dot(R @ tgt["up"], ray), -1, 1))))
    label(view, f"input  (axis tilt {tilt:.0f} deg) {info}")
    label(syn, f"from video: seen {cover_tooth*100:.0f}% / access {cover_access*100:.0f}%")
    label(syn, "gray = not visible (not filled)", y=out_size - 34)
    label(cg, "CG model (current inset)")
    return np.hstack([view, syn, cg]), dict(tilt=tilt, cover_tooth=cover_tooth, cover_access=cover_access)


# ─── カメラ直結（トラッカーなし）───────────────────────────────────────────────

def run_live(a, tgt, Vt, Ft, Va, Fa):
    """姿勢はトラッカーの代わりに ArUco ボード + マーカー基準の登録から求める（誤差は登録の精度 ~1〜2 mm）。
    トラッカーと同じく、歪み補正した画像と補正後の K でボードを検出する。"""
    import time
    sys.path.insert(0, REPO)
    from camera_io import open_camera
    from marker_reference import MarkerReference
    reg = os.path.join(REPO, "initial_poses", "four_group_molar_premolar_canine_incisor", "marker_registration.json")
    cap, (w, h) = open_camera()
    K, maps = load_calibration(a.calib, w, h)
    marker = MarkerReference(K, None, reg)
    out_dir = a.out or os.path.join(HERE, "topview_live")
    os.makedirs(out_dir, exist_ok=True)
    for _ in range(20):            # 露出が落ち着くまで捨てる
        cap.read()
    saved, n = 0, 0
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                print("カメラから読めない")
                break
            n += 1
            frame = cv2.remap(raw, maps[0], maps[1], cv2.INTER_LINEAR)
            marker.detect(frame)
            T = marker.predict()
            if T is None:
                msg = "ボードが見えない" if marker.last is None else "マーカー基準が未登録"
                if a.frames:
                    print(f"  frame {n}: {msg}")
                    if n > 300:
                        break
                    continue
                img = cv2.resize(frame, (1260, 709))
                label(img, msg)
            else:
                t0 = time.time()
                img, st = render_frame(frame, T[:3, :3], T[:3, 3], K, tgt, Vt, Ft, Va, Fa,
                                       info=f"pose: ArUco {marker.last['key']}")
                if a.frames and n % 5 == 0:
                    p = os.path.join(out_dir, f"live_{time.strftime('%H%M%S')}_{saved:02d}.jpg")
                    cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    saved += 1
                    print(f"  {p}  tilt {st['tilt']:.0f} deg  seen {st['cover_tooth']*100:.0f}%  "
                          f"access {st['cover_access']*100:.0f}%  ({(time.time()-t0)*1000:.0f} ms)")
                    if saved >= a.frames:
                        break
            if not a.frames:
                cv2.imshow("top view from video (q: quit, s: save)", img)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("s"):
                    p = os.path.join(out_dir, f"live_{time.strftime('%H%M%S')}.jpg")
                    cv2.imwrite(p, img)
                    print(p)
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", help="録画クリップのフォルダ（video.avi + frames.csv）")
    ap.add_argument("--log", help="トラッカーの CSV（省略時はクリップの時刻を含むものを logs/ から探す）")
    ap.add_argument("--tooth", default="37")
    ap.add_argument("--arch", default=os.path.join(REPO, "models", "arch_4groups_smooth.obj"))
    ap.add_argument("--calib", default=os.path.join(os.environ.get("PYM3T_DIR", ""), "camera_calibration.npz"))
    ap.add_argument("--every", type=int, default=10, help="何フレームごとに作るか")
    ap.add_argument("--cg-only", action="store_true", help="映像なしで、ログの最後の姿勢の CG だけ確認する")
    ap.add_argument("--live", action="store_true",
                    help="トラッカーなしでカメラから直接。姿勢は ArUco ボード + マーカー基準の登録（トラッカーを止めてから使う）")
    ap.add_argument("--frames", type=int, default=0, help="--live: この枚数を保存したら終わる（0 = ウィンドウで見続ける）")
    ap.add_argument("--out", help="出力フォルダ（既定: <クリップ>/topview、--live は experiment/topview_live）")
    a = ap.parse_args()

    tgt = load_target(a.tooth)
    Vt, Ft = load_obj(os.path.join(REPO, "models", f"tooth_{a.tooth}_smooth.obj"))
    Va, Fa = load_obj(a.arch)

    if a.live:
        run_live(a, tgt, Vt, Ft, Va, Fa)
        return

    if a.cg_only:
        rows = load_log(a.log)
        _, R, t, mode, q, err = rows[-1]
        K = np.array([[900, 0, 640], [0, 900, 360], [0, 0, 1]], float)
        if os.path.exists(a.calib):
            K, _ = load_calibration(a.calib, 1280, 720)
        frame = np.full((720, 1280, 3), 60, np.uint8)
        Pc = Va @ R.T + t
        uv = Pc @ K.T
        zb, fid, _, _ = raster(uv[:, :2] / uv[:, 2:], Pc[:, 2], Fa, 1280, 720)
        frame[fid >= 0] = (200, 200, 200)
        img, st = render_frame(frame, R, t, K, tgt, Vt, Ft, Va, Fa, info=f"{mode} q={q:.2f}")
        out = a.out or os.path.join(HERE, "topview_cg_check.png")
        cv2.imwrite(out, img)
        print(out, st)
        return

    clip = a.clip
    with open(os.path.join(clip, "frames.csv"), encoding="utf-8") as f:
        fts = [float(r["timestamp"]) for r in csv.DictReader(f)]
    log = a.log or find_log_for(fts[0], fts[-1])
    if not log:
        sys.exit("クリップの時刻を含むトラッカーの CSV が見つからない（--log で指定）")
    rows = load_log(log)
    rts = [r[0] for r in rows]
    print(f"clip {clip}: {len(fts)} frames, log {os.path.basename(log)} ({len(rows)} tracking rows)")

    cap = cv2.VideoCapture(os.path.join(clip, "video.avi"))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K, maps = load_calibration(a.calib, w, h)
    out_dir = a.out or os.path.join(clip, "topview")
    os.makedirs(out_dir, exist_ok=True)
    stats = []
    for i, ts in enumerate(fts):
        ok, raw = cap.read()
        if not ok:
            break
        if i % a.every:
            continue
        j = bisect.bisect_left(rts, ts)
        if j >= len(rows) or rows[j][0] - ts > 0.2:
            continue
        _, R, t, mode, q, err = rows[j]
        frame = cv2.remap(raw, maps[0], maps[1], cv2.INTER_LINEAR)
        img, st = render_frame(frame, R, t, K, tgt, Vt, Ft, Va, Fa, info=f"q={q:.2f}")
        cv2.imwrite(os.path.join(out_dir, f"top_{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        stats.append(dict(frame=i, mode=mode, quality=q, marker_err_mm=err, **st))
        print(f"  frame {i:5d}  {mode:14s} q={q:.2f}  tilt {st['tilt']:5.1f} deg  "
              f"seen {st['cover_tooth']*100:4.0f}%  access {st['cover_access']*100:4.0f}%")
    with open(os.path.join(out_dir, "stats.csv"), "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(stats[0]))
        wtr.writeheader()
        wtr.writerows(stats)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
