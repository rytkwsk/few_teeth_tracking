# -*- coding: utf-8 -*-
"""統合中の回転追従（FUSION_ROT_*、2026-10-14）の効き方を CSV から集計する。

使い方: python analysis/fusion_rot_stats.py [logs/tracking_*.csv ...]   （省略時は最新 1 本）

出力:
  1. F 追跡中（統合でない）の歯群ボディ回転と出力回転の差 gA/gB/gD_vs_out_deg の分布
     → 歯群の回転が F とどれだけ系統的にずれるか（追従先の偏り）。中央値が大きいほど偏り補正が重要。
  2. 統合（F 離脱）区間ごとの: 継続秒数、回転源の内訳（A/B/固定）、累計追従量 fus_rot_deg の最大、
     離脱直前と復帰直後（+1 s）のマーカー回転誤差（登録済みのとき）、復帰時再探索の結果。
  3. 回転源別のマーカー回転誤差（統合中、登録済みのとき）。
ArUco の 1 フレーム反転（err 100 mm 超）は除外する。
"""
import csv
import glob
import os
import sys

import numpy as np

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def pct(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def summarize(name, v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if len(v) == 0:
        print(f"  {name:38s} n=0")
        return
    print(f"  {name:38s} n={len(v):5d}  median={np.median(v):6.2f}  p90={pct(v, 90):6.2f}  max={v.max():6.2f}")


def analyze(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        print(f"{path}: 空")
        return
    cols = rows[0].keys()
    if "fus_rot_src" not in cols:
        print(f"{path}: fus_rot_src 列なし（2026-10-14 より前のログ）")
        return
    t = np.array([fnum(r["timestamp"]) for r in rows])
    t0 = t[0]
    mode = [r["mode"] for r in rows]
    err_deg = np.array([fnum(r["err_deg"]) for r in rows])
    err_mm = np.array([fnum(r["err_mm"]) for r in rows])
    flip = err_mm > 100.0          # ArUco の 1 フレーム反転
    err_deg = np.where(flip, np.nan, err_deg)
    registered = np.isfinite(err_deg).any()
    in_fusion = np.array([r["fus_rot_deg"] != "" for r in rows])   # fusion_off_f のフレームだけ値が入る
    src = [r["fus_rot_src"] for r in rows]
    rot_total = np.array([fnum(r["fus_rot_deg"]) for r in rows])

    print(f"=== {os.path.basename(path)}  rows={len(rows)}  dur={t[-1]-t0:.0f}s  marker={'yes' if registered else 'no'}")
    print("-- 1. F 追跡中: 歯群ボディの回転 vs 出力回転（偏り）")
    f_track = np.array([(m.startswith("TRACKING[F]")) and not f for m, f in zip(mode, in_fusion)])
    for k in ("A", "B", "D"):
        v = np.array([fnum(r[f"g{k}_vs_out_deg"]) for r in rows])
        summarize(f"g{k}_vs_out_deg (F tracking)", v[f_track])
        if registered:
            summarize(f"  g{k}_rot_deg vs marker (F tracking)", np.array([fnum(r[f"g{k}_rot_deg"]) for r in rows])[f_track])
    if registered:
        summarize("F output rot err vs marker (F tracking)", err_deg[f_track])

    print("-- 2. 統合（F 離脱）区間")
    segs = []
    i = 0
    n = len(rows)
    while i < n:
        if not in_fusion[i]:
            i += 1
            continue
        j = i
        while j < n and in_fusion[j]:
            j += 1
        segs.append((i, j))
        i = j
    if not segs:
        print("  統合区間なし")
    for (i, j) in segs:
        dur = t[j - 1] - t[i]
        srcs = {}
        for s in src[i:j]:
            key = s if s else "fixed"
            srcs[key] = srcs.get(key, 0) + 1
        src_text = ", ".join(f"{k}:{v}" for k, v in sorted(srcs.items(), key=lambda kv: -kv[1]))
        rmax = np.nanmax(rot_total[i:j]) if np.isfinite(rot_total[i:j]).any() else float("nan")
        line = f"  t={t[i]-t0:6.1f}s  {dur:5.1f}s  rot_followed_max={rmax:5.1f}deg  src[{src_text}]"
        if registered:
            before = np.nanmedian(err_deg[max(0, i - 10):i]) if i > 0 else float("nan")
            during = np.nanmedian(err_deg[i:j])
            after_mask = (t >= t[j - 1]) & (t <= t[j - 1] + 1.0)
            after = np.nanmedian(err_deg[after_mask]) if after_mask.any() else float("nan")
            line += f"  rot_err before={before:5.1f} during={during:5.1f} after(+1s)={after:5.1f} deg"
        ev = [r["event"] for r in rows[j:min(n, j + 40)] if r["event"].startswith("return_search:")]
        if ev:
            line += f"  {ev[0]}"
        print(line)

    if registered:
        print("-- 3. 統合中の回転源別マーカー回転誤差")
        for key in ("A", "B", "A(cap)", "B(cap)", ""):
            m = np.array([f and (s == key) for f, s in zip(in_fusion, src)])
            summarize(f"src={key or 'fixed'}", err_deg[m])


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(LOG_DIR, "tracking_*.csv")))[-1:]
    for p in paths:
        analyze(p)
