# -*- coding: utf-8 -*-
"""器具の先端を映像から見つけ、根管口の照準との横方向の距離を出す（根管治療ガイドの案 B の試作、オフライン）。

器具は追跡しない（器具の 3D モデルもマーカーも使わない）。追跡している「治療する歯」の投影の中で、歯の色ではない画素
（ペン・バー）を見つけ、その塊の一番根元側（歯の長軸を下った側）の画素を先端とする。先端は歯の表面に当たっている
はずなので、カメラからその画素への視線と歯のメッシュの交点を先端の 3D 位置とし、OBJ 座標の xz（上から見た面）で
各根管口の照準（Unity の Targets/*.json の entry）との距離を mm で出す。
  - 歯の姿勢は再生 CSV の ArUco 基準（ref_*、--pose ref）かトラッカーの出力（--pose track）
  - 歯の色は遮蔽前（events の occlude より前）のフレームの歯の画素から Lab の平均・共分散で作る
  - 削り込んで先端が表面より深くなると視線の交点がずれる（斜めから見ている分）。ここで見るのは削り始めの当て位置

    python experiment/tip_detect.py <セッション>/<クリップ> [--tag base] [--pose ref] [--tooth 36]
出力: <クリップ>/tip_<tag>.csv（フレームごと）、tip_<tag>.jpg（遮蔽中のフレームを並べた確認用の絵）、tip_<tag>_top.png
（上から見た先端の分布と 1 mm の照準）。--pose track のときは tip_<tag>_track.*。mb / ml / d 区間ごとのずれも表示する。
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITY = os.path.join(os.path.dirname(REPO), "unity-few-teeth-tracking")
UNIT_M = 0.025          # OBJ 1 単位 = 25 mm（MODEL_UNIT_IN_METER）
OCCLUDED = ("occlude", "move", "hold", "mb", "ml", "d", "trace")   # mb/ml/d/trace = record.py の 9（reamer）


def load_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(V), np.array(F, dtype=np.int64)


def camera(meta_path, w, h):
    """トラッカーと同じ歪み補正（getOptimalNewCameraMatrix の alpha 0）。補正後の K と remap の表"""
    calib = np.load(json.load(open(meta_path, encoding="utf-8"))["calibration"])
    K, dist = calib["K"].astype(np.float32), calib["dist"].astype(np.float32)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    maps = cv2.initUndistortRectifyMap(K, dist, None, new_K.astype(np.float32), (w, h), cv2.CV_16SC2)
    return new_K.astype(np.float64), maps


def pose(row, use_ref):
    if use_ref:
        if not row["ref_rx"]:
            return None
        R = cv2.Rodrigues(np.array([float(row[k]) for k in ("ref_rx", "ref_ry", "ref_rz")]))[0]
        t = np.array([float(row[k]) for k in ("ref_tx", "ref_ty", "ref_tz")])
    else:
        R = np.array([float(row[f"r{i}{j}"]) for i in range(3) for j in range(3)]).reshape(3, 3)
        t = np.array([float(row[k]) for k in ("tx", "ty", "tz")])
    return R, t


def project(P, K):
    return np.stack([K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2], K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]], 1)


def ray_hit(Vc, F, ray):
    """原点からの視線 ray と三角形の最も近い交点（Möller–Trumbore をまとめて）。無ければ None"""
    A, B, C = Vc[F[:, 0]], Vc[F[:, 1]], Vc[F[:, 2]]
    e1, e2 = B - A, C - A
    p = np.cross(ray, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = -A
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ ray) * inv
    d = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (d > 0)
    if not hit.any():
        return None
    return ray * d[hit].min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--pose", choices=("ref", "track"), default="ref")
    ap.add_argument("--tooth", default="36")
    ap.add_argument("--maha", type=float, default=25.0, help="歯の色からのマハラノビス距離の 2 乗がこれを超えたら歯ではない")
    ap.add_argument("--min-area", type=int, default=40, help="器具とみなす塊の最小の画素数")
    ap.add_argument("--detector", choices=("color", "bg"), default="color",
                    help="color = 歯の色でない画素の、歯の長軸を下った端（59° 用。30° で横から入る器具ではシャフトが歯の輪郭を"
                         "横切る所を先端と取り違えた）/ bg = 遮蔽前の映像との差（模型とカメラが動かない録画用）の塊で、"
                         "器具が範囲に入ってくる所からいちばん遠い点")
    ap.add_argument("--bg-lum", type=float, default=0.22,
                    help="bg: 遮蔽前の映像との明るさの比の対数の絶対値がこれを超えたら器具。器具は 0.4〜0.75 暗く（光った所は明るく）、"
                         "器具の影は ~0.1 なので、影を先端と取り違えないよう間で切る（Lab の差 20 だと影の端を先端にした、reamer_04）。"
                         "色みの差は圧縮のブロックの揺れに埋もれて使えなかった")
    ap.add_argument("--bg-margin", type=int, default=90, help="bg: 歯の投影をこれだけ広げた範囲で器具を探す（画素）")
    args = ap.parse_args()

    out = args.tag if args.pose == "ref" else f"{args.tag}_{args.pose}"   # 出力 tip_<out>.*（姿勢・検出法ごとに分ける）
    if args.detector != "color":
        out += f"_{args.detector}"
    clip = args.clip if os.path.isabs(args.clip) else os.path.join(REPO, "experiment", "sessions", args.clip)
    V, F = load_obj(os.path.join(REPO, "models", f"tooth_{args.tooth}_smooth.obj"))
    target = json.load(open(os.path.join(UNITY, "Assets", "Resources", "Targets", f"{args.tooth}.json")))
    entries = {c["name"]: np.array(c["entry"]) for c in target["canals"] if "entry" in c}
    access = np.array(target["access"])
    rows = {int(r["frame"]): r for r in csv.DictReader(open(os.path.join(clip, f"replay_{args.tag}.csv"), encoding="utf-8"))}
    events = [(int(r["frame_index"]), r["event"]) for r in csv.DictReader(open(os.path.join(clip, "events.csv"), encoding="utf-8"))]
    occlude_at = next((f for f, e in events if e == "occlude"), 10 ** 9)

    def phase(i):
        ph = "pre"
        for f, e in events:
            if i >= f:
                ph = e
        return ph

    cap = cv2.VideoCapture(os.path.join(clip, "video.avi"))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K, maps = camera(os.path.join(clip, "meta.json"), w, h)
    Vm = V * UNIT_M
    color_px, mu, icov = [], None, None
    bg_frames, bg = [], None
    out_rows, sheet, tilts = [], [], []
    i = -1
    while True:
        ok, raw = cap.read()
        i += 1
        if not ok:
            break
        r = rows.get(i)
        pz = pose(r, args.pose == "ref") if r else None
        if pz is None:
            continue
        R, t = pz
        # カメラの傾き（歯の長軸 = OBJ の +y と、切削開始点 → カメラの角度）。Unity の M3TTargetOverlay の表示と同じ
        a_cam = (access * UNIT_M) @ R.T + t
        tilts.append(float(np.degrees(np.arccos(np.clip(R[:, 1] @ (-a_cam / np.linalg.norm(a_cam)), -1.0, 1.0)))))
        img = cv2.remap(raw, maps[0], maps[1], cv2.INTER_LINEAR)
        Vc = Vm @ R.T + t
        uv = project(Vc, K)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.round(uv[f]).astype(np.int32).reshape(-1, 1, 2) for f in F], 255)
        inner = cv2.erode(mask, np.ones((5, 5), np.uint8))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
        ph = phase(i)

        # 歯の色: 遮蔽前のフレームで集める（隠れる前の歯の画素）
        if i < occlude_at:
            color_px.append(lab[inner > 0][::7])
            if i % 3 == 0:
                bg_frames.append(img.sum(axis=2, dtype=np.float32))
            continue
        if mu is None:
            P = np.concatenate(color_px)
            mu, icov = P.mean(0), np.linalg.inv(np.cov(P.T) + np.eye(3) * 1e-3)
            bg = np.median(np.stack(bg_frames[-30:]), axis=0)

        if args.detector == "bg":
            roi = cv2.dilate(mask, np.ones((2 * args.bg_margin + 1,) * 2, np.uint8))
            d = np.abs(np.log((img.sum(axis=2, dtype=np.float32) + 3.0) / (bg + 3.0)))
            foreign = ((d > args.bg_lum) & (roi > 0)).astype(np.uint8)
            foreign = cv2.morphologyEx(foreign, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        else:
            diff = lab - mu
            d2 = np.einsum("hwi,ij,hwj->hw", diff, icov, diff)
            foreign = ((d2 > args.maha) & (inner > 0)).astype(np.uint8)
            foreign = cv2.morphologyEx(foreign, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(foreign)

        # 画面上の「根元側」の向き = 切削開始点から歯の長軸（OBJ の +y）を下った向き
        a_c = (access * UNIT_M) @ R.T + t
        b_c = ((access - [0, 0.2, 0]) * UNIT_M) @ R.T + t
        down = project(np.array([b_c]), K)[0] - project(np.array([a_c]), K)[0]
        down /= np.linalg.norm(down) + 1e-9

        tip = None
        if n > 1 and args.detector == "bg":
            # 歯の投影にかかる塊のうち最大のもの。器具は範囲の縁から入ってくるので、縁に接する画素（入口）から
            # いちばん遠い、歯の上の画素を先端とする（入口が無ければ塊の中で歯の中心からいちばん遠い点の反対側）
            on_tooth = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= args.min_area and (inner[lab_cc == k] > 0).any()]
            if on_tooth:
                k = max(on_tooth, key=lambda k: stats[k, cv2.CC_STAT_AREA])
                comp = lab_cc == k
                edge = comp & (cv2.erode(roi, np.ones((5, 5), np.uint8)) == 0)
                ys, xs = np.nonzero(comp & (inner > 0))
                if edge.any():
                    ey, ex = np.nonzero(edge)
                    entry = np.array([ex.mean(), ey.mean()])
                else:
                    cy, cx = np.nonzero(comp)
                    my, mx = np.nonzero(mask)
                    c = np.array([mx.mean(), my.mean()])
                    far = np.argmax((cx - c[0]) ** 2 + (cy - c[1]) ** 2)
                    entry = np.array([cx[far], cy[far]], float)
                j = int(np.argmax((xs - entry[0]) ** 2 + (ys - entry[1]) ** 2))
                tip = (float(xs[j]), float(ys[j]))
        elif n > 1:
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[k, cv2.CC_STAT_AREA] >= args.min_area:
                ys, xs = np.nonzero(lab_cc == k)
                j = int(np.argmax(xs * down[0] + ys * down[1]))
                tip = (float(xs[j]), float(ys[j]))
        row = {"frame": i, "phase": ph, "area": int(foreign.sum())}
        if tip is not None:
            ray = np.linalg.inv(K) @ np.array([tip[0], tip[1], 1.0])
            ray /= np.linalg.norm(ray)
            hit = ray_hit(Vc, F, ray)
            row.update({"tip_u": round(tip[0], 1), "tip_v": round(tip[1], 1)})
            if hit is not None:
                pm = (R.T @ (hit - t)) / UNIT_M                    # OBJ 座標
                row.update({"x": round(pm[0], 5), "y": round(pm[1], 5), "z": round(pm[2], 5)})
                for name, e in entries.items():
                    row[f"d_{name}_mm"] = round(float(np.linalg.norm((pm - e)[[0, 2]]) * 25.0), 2)
                row["d_access_mm"] = round(float(np.linalg.norm((pm - access)[[0, 2]]) * 25.0), 2)
        out_rows.append(row)

        # 確認用の絵（遮蔽中を 8 枚くらい）
        if ph in OCCLUDED and len(sheet) < 8 and i % 15 == 0:
            vis = img.copy()
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (255, 255, 255), 1)
            vis[foreign > 0] = (0.5 * vis[foreign > 0] + [0, 0, 127]).astype(np.uint8)
            for name, e in entries.items():
                ec = (e * UNIT_M) @ R.T + t
                p = project(np.array([ec]), K)[0]
                cv2.circle(vis, (int(p[0]), int(p[1])), 4, (255, 255, 255), 1)
            if tip is not None:
                cv2.drawMarker(vis, (int(tip[0]), int(tip[1])), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
            cx, cy = np.mean(np.nonzero(mask)[1]), np.mean(np.nonzero(mask)[0])
            x0, y0 = int(max(0, cx - 160)), int(max(0, cy - 120))
            crop = vis[y0:y0 + 240, x0:x0 + 320].copy()
            cv2.putText(crop, f"#{i} {ph}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            sheet.append(cv2.resize(crop, (320, 240)))

    keys = sorted({k for r in out_rows for k in r}, key=lambda k: (k not in ("frame", "phase", "area"), k))
    with open(os.path.join(clip, f"tip_{out}.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(out_rows)
    if sheet:
        while len(sheet) % 4:
            sheet.append(np.zeros_like(sheet[0]))
        cv2.imwrite(os.path.join(clip, f"tip_{out}.jpg"),
                    np.vstack([np.hstack(sheet[k:k + 4]) for k in range(0, len(sheet), 4)]))

    occ = [r for r in out_rows if r["phase"] in OCCLUDED]
    found = [r for r in occ if "x" in r]
    print(f"{os.path.basename(clip)}: occluded frames {len(occ)}, tip on the tooth {len(found)}, "
          f"camera tilt from the tooth axis {np.median(tilts) if tilts else float('nan'):.0f} deg")
    if found:
        P = np.array([[r["x"], r["z"]] for r in found]) * 25.0
        print(f"  tip xz spread (std) {P.std(0).round(2)} mm; nearest target median: "
              + ", ".join(f"{n} {np.median([r[f'd_{n}_mm'] for r in found]):.1f} mm" for n in entries))
    summarize_phases(out_rows, entries, target, clip, out)


def summarize_phases(out_rows, entries, target, clip, tag):
    """record.py の 9（reamer）の mb / ml / d 区間ごとに、狙った照準からのずれを近遠心（+M）・頬舌（+B）に分けて出す。
    正解は「術者が当てた黄色い円」= AR 画面に描いた照準なので、--pose track（表示と同じ姿勢）で比べるのが目的に合う。
    上から見た図（tip_<tag>_top.png）も描く。"""
    mesial = np.array(target["mesial"])[[0, 2]]
    buccal = np.array(target["buccal"])[[0, 2]]
    aimed = {"mb": "MB", "ml": "ML", "d": "D"}
    lines = []
    for ph, name in aimed.items():
        rows = [r for r in out_rows if r["phase"] == ph]
        if name not in entries or not rows:
            continue
        hits = [r for r in rows if "x" in r]
        if not hits:
            lines.append(f"  {ph:3s} -> {name:2s}: frames {len(rows)}, tip found 0")
            continue
        off = (np.array([[r["x"], r["z"]] for r in hits]) - entries[name][[0, 2]]) * 25.0
        m, b = off @ mesial, off @ buccal
        d = np.hypot(m, b)
        nearest = [min(entries, key=lambda n: r[f"d_{n}_mm"]) for r in hits]
        right = sum(n == name for n in nearest) / len(hits)
        lines.append(f"  {ph:3s} -> {name:2s}: frames {len(rows)}, tip found {len(hits)} ({len(hits) / len(rows):.0%}), "
                     f"error median {np.median(d):.1f} / p90 {np.percentile(d, 90):.1f} mm, "
                     f"offset median M {np.median(m):+.1f} B {np.median(b):+.1f} mm, "
                     f"spread (std) M {m.std():.1f} B {b.std():.1f} mm, nearest = {name} {right:.0%}")
    if lines:
        print("  per aimed orifice (+M = mesial, +B = buccal):")
        print("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # 上から見た図: 横 = 近遠心（右が M）、縦 = 頬舌（上が B）。原点 = 切削開始点
    origin = np.array(target["access"])[[0, 2]]

    def to_mb(p):
        q = (np.atleast_2d(p) - origin) * 25.0
        return q @ mesial, q @ buccal

    fig, ax = plt.subplots(figsize=(6, 6))
    if target.get("outline"):
        o = np.array(target["outline"]).reshape(-1, 3)[:, [0, 2]]
        x, y = to_mb(np.vstack([o, o[:1]]))
        ax.plot(x, y, color="0.6", lw=1)
    colors = {"trace": "0.75", "mb": "tab:red", "ml": "tab:blue", "d": "tab:green"}
    for ph, c in colors.items():
        pts = np.array([[r["x"], r["z"]] for r in out_rows if r["phase"] == ph and "x" in r])
        if len(pts):
            x, y = to_mb(pts)
            ax.scatter(x, y, s=6, color=c, alpha=0.5, label=f"{ph} ({len(pts)})")
    for name, e in entries.items():
        x, y = to_mb(e[[0, 2]])
        ax.add_patch(plt.Circle((x[0], y[0]), 1.0, fill=False, color="k", lw=1.5, zorder=3))
        ax.text(x[0] + 1.1, y[0] + 1.1, name, fontsize=12, weight="bold", zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("mesial (+) / distal (-)  [mm]")
    ax.set_ylabel("buccal (+) / lingual (-)  [mm]")
    ax.set_title(f"{os.path.basename(clip)}: detected tip, circles = 1 mm targets", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(clip, f"tip_{tag}_top.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
