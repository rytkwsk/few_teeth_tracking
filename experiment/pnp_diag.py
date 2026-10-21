# -*- coding: utf-8 -*-
"""遮蔽中の特徴点 PnP の切り分け（2026-10-21）。replay.py --dump の pnp_<tag>.pkl と replay_<tag>.csv を使う。

    python experiment/pnp_diag.py 20261021_home --tag diag

ArUco の登録誤差（約 0.8 mm = 数 px の一定のずれ）を消すため、真値は「キーフレームの姿勢 × ArUco で測ったキーフレーム
からの相対運動」とする（T_true(t) = T_gt(t) T_gt(t_kf)^-1 T_kf）。各フレームで:
  res_est  : PnP の推定姿勢で点を投影したときの残差中央値 [px]（PnP 自身が見ている当てはまり）
  res_true : 真値の姿勢で投影したときの残差中央値 [px]（点が正しく追えていれば小さい）
  lk_bad   : 点ごとの「結び付けた時刻からの真の動き」と LK の動きの差が 3 px を超える点の割合（追跡が壊れた点）
  err      : 推定姿勢と真値の差（回転 deg / F 重心 mm）
を出し、点の由来（キーフレーム / 補充）ごとにも分ける。
判定: res_true が res_est と同程度に小さいのに err が大きい → 解き方の問題（曖昧さ・事前分布）。
      res_true が大きい・lk_bad が多い → 点が壊れている（LK の乗り移り・補充の焼き付き）。
"""
import argparse
import csv
import glob
import os
import pickle

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(HERE, "sessions")
LK_BAD_PX = 3.0


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def proj(Ks, R, t, X):
    Xc = X @ R.T + t
    return (Xc[:, :2] / Xc[:, 2:3]) * np.array([Ks[0, 0], Ks[1, 1]]) + np.array([Ks[0, 2], Ks[1, 2]])


def gt_table(clip, tag):
    ts, poses = [], []
    for r in csv.DictReader(open(os.path.join(clip, f"replay_{tag}.csv"), encoding="utf-8")):
        rv = np.array([fnum(r[k]) for k in ("ref_rx", "ref_ry", "ref_rz")])
        tv = np.array([fnum(r[k]) for k in ("ref_tx", "ref_ty", "ref_tz")])
        if not (np.isfinite(rv).all() and np.isfinite(tv).all()) or fnum(r["err_mm"]) > 100:
            continue
        ts.append(fnum(r["timestamp"]))
        poses.append((cv2.Rodrigues(rv)[0], tv))
    return np.array(ts), poses


def gt_at(ts, poses, t, tol=0.03):
    if len(ts) == 0:
        return None
    i = int(np.argmin(np.abs(ts - t)))
    return poses[i] if abs(ts[i] - t) <= tol else None


def rel(Ra, ta, Rb, tb):
    """T_a T_b^-1"""
    R = Ra @ Rb.T
    return R, ta - R @ tb


def phase_of(t, events):
    cur = "pre"
    for te, name in events:
        if t >= te:
            cur = name
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--tag", default="diag")
    ap.add_argument("--every", type=float, default=0.5, help="フレーム表示の間隔 [s]")
    args = ap.parse_args()
    summary = []
    for clip in sorted(glob.glob(os.path.join(SESSIONS_DIR, args.session, "occ_*"))):
        pk = os.path.join(clip, f"pnp_{args.tag}.pkl")
        if not os.path.exists(pk):
            continue
        D = pickle.load(open(pk, "rb"))
        if not D["frames"]:
            print(f"\n## {os.path.basename(clip)}: 統合中の PnP なし")
            continue
        cF = np.asarray(D["F_centroid"], np.float64)
        ts, poses = gt_table(clip, args.tag)
        events = [(float(r["timestamp"]), r["event"]) for r in csv.DictReader(open(os.path.join(clip, "events.csv"), encoding="utf-8"))]
        t0 = events[0][0]
        print(f"\n## {os.path.basename(clip)}   列: t 区間 | 点数(補充) | res_est res_true px | lk_bad% (初期/補充) | "
              f"誤差 deg mm | 真の回転 deg 移動 mm")
        last = -9.0
        abs_floor = {}
        for fr in D["frames"]:
            seg = D["segments"][fr["seg"]]
            Ks, Xall, born, bposes = seg["Ks"], seg["X"], seg["born"], seg["born_poses"]
            t_kf, R_kf, t_kfv = bposes[0]
            g_kf = gt_at(ts, poses, t_kf) if t_kf is not None else None
            g_t = gt_at(ts, poses, fr["t"])
            if g_kf is None or g_t is None:
                continue
            # 真値（登録の偏りを消したもの）: キーフレーム姿勢 × 真の相対運動
            Rr, tr = rel(g_t[0], g_t[1], g_kf[0], g_kf[1])
            R_true, t_true = Rr @ R_kf, Rr @ t_kfv + tr
            idx = fr["idx"]
            X, u = Xall[idx], fr["cur"]
            res_est = np.linalg.norm(proj(Ks, fr["R"], fr["t_vec"], X) - u, axis=1)
            res_true = np.linalg.norm(proj(Ks, R_true, t_true, X) - u, axis=1)
            res_abs = np.linalg.norm(proj(Ks, g_t[0], g_t[1], X) - u, axis=1)   # ArUco の絶対姿勢（登録の偏りぶん底上げ）
            if fr["seg"] not in abs_floor:
                abs_floor[fr["seg"]] = float(np.median(res_abs))
            # LK の健全性: 結び付けた時刻 tb からの真の動きで、結び付けた位置 born_uv を動かした予測と比べる
            lk_err = np.full(len(idx), np.nan)
            for bi in np.unique(born[idx]):
                m = born[idx] == bi
                tb, Rb, tbv = bposes[bi]
                g_b = gt_at(ts, poses, tb)
                if g_b is None:
                    continue
                Rm, tm = rel(g_t[0], g_t[1], g_b[0], g_b[1])
                Xs = X[m]
                disp = proj(Ks, Rm @ Rb, Rm @ tbv + tm, Xs) - proj(Ks, Rb, tbv, Xs)
                lk_err[m] = np.linalg.norm(u[m] - (seg["born_uv"][idx[m]] + disp), axis=1)
            init = born[idx] == 0
            err_deg = float(np.degrees(np.linalg.norm(cv2.Rodrigues(fr["R"] @ R_true.T)[0])))
            err_mm = float(np.linalg.norm((fr["R"] @ cF + fr["t_vec"]) - (R_true @ cF + t_true)) * 1000)
            true_deg = float(np.degrees(np.linalg.norm(cv2.Rodrigues(Rr)[0])))
            true_mm = float(np.linalg.norm((R_true @ cF + t_true) - (R_kf @ cF + t_kfv)) * 1000)
            ph = phase_of(fr["t"], events)
            ok_lk = np.isfinite(lk_err)
            bad_i = float(np.mean(lk_err[init & ok_lk] > LK_BAD_PX) * 100) if (init & ok_lk).any() else float("nan")
            bad_r = float(np.mean(lk_err[~init & ok_lk] > LK_BAD_PX) * 100) if (~init & ok_lk).any() else float("nan")
            summary.append((os.path.basename(clip), ph, fr["still"], np.median(res_est), np.median(res_true), bad_i, bad_r,
                            err_deg, err_mm, true_deg, true_mm, (~init).sum(), len(idx)))
            if fr["t"] - last >= args.every:
                last = fr["t"]
                print(f"  {fr['t']-t0:5.1f} {ph:8s}{'(still)' if fr['still'] else '       '} | {len(idx):3d}({(~init).sum():3d}) | "
                      f"{np.median(res_est):5.2f} {np.median(res_true):6.2f} abs {np.median(res_abs):5.2f}(始 {abs_floor[fr['seg']]:4.2f}) | "
                      f"{bad_i:5.1f} / {bad_r:5.1f} | "
                      f"{err_deg:5.1f} {err_mm:5.1f} | {true_deg:5.1f} {true_mm:5.1f}")
    # まとめ: 動かしている区間（move/hold、静止判定でないフレーム）
    S = [s for s in summary if s[1] in ("move", "hold") and not s[2]]
    if S:
        A = np.array([s[3:11] for s in S], dtype=float)
        big = A[:, 4] > 3.0
        print("\n== まとめ（move/hold、PnP が解いたフレーム）")
        print(f"  フレーム {len(S)}  うち誤差 >3°: {big.sum()}")
        for name, m in (("誤差 ≤3°", ~big), ("誤差 >3°", big)):
            if m.any():
                print(f"  {name}: res_est {np.nanmedian(A[m,0]):.2f} px  res_true {np.nanmedian(A[m,1]):.2f} px  "
                      f"lk_bad 初期 {np.nanmedian(A[m,2]):.1f}% 補充 {np.nanmedian(A[m,3]):.1f}%")


if __name__ == "__main__":
    main()
