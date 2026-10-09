# -*- coding: utf-8 -*-
"""ArUco 基準誤差 (err_mm / err_deg) の集計と、fit quality との関係。
usage: python marker_stats.py [tracking_*.csv ...]   (省略時は最新)"""
import csv, glob, math, os, sys

logs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")


def fl(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None


def pct(s, q):
    return s[min(len(s) - 1, int(len(s) * q))]


def describe(name, xs, unit):
    if not xs:
        print(f"  {name:28s} n=0", flush=True)
        return
    s = sorted(xs)
    print(f"  {name:28s} n={len(s):5d}  median={pct(s,.5):6.2f}{unit}  p90={pct(s,.9):6.2f}{unit}  "
          f"p99={pct(s,.99):6.2f}{unit}  max={s[-1]:6.2f}{unit}", flush=True)


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


def spearman(a, b):
    if len(a) < 10:
        return float("nan")
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra)); vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else float("nan")


def run(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    data = []
    for r in rows:
        mm, deg = fl(r.get("err_mm")), fl(r.get("err_deg"))
        if mm is None or deg is None:
            continue
        mode = r["mode"]
        data.append(dict(mm=mm, deg=deg, q=fl(r["quality"]), qF=fl(r.get("qF")), mode=mode,
                         snap=mode.startswith("SNAP"), hold="hold" in mode,
                         src=mode[mode.find("[") + 1:mode.find("]")] if "[" in mode else "?",
                         nsamp=r.get("ref_n_samples", "")))
    print(f"=== {os.path.basename(path)}  rows={len(rows)}  with marker error={len(data)}", flush=True)
    if not data:
        print("  (err_mm empty: marker not registered or not visible)", flush=True)
        return
    print(f"  registration samples: {sorted(set(d['nsamp'] for d in data))}", flush=True)
    trk = [d for d in data if not d["snap"]]
    describe("translation, all tracking", [d["mm"] for d in trk], " mm")
    describe("rotation,    all tracking", [d["deg"] for d in trk], " deg")
    by_src = {}
    for d in trk:
        by_src.setdefault(d["src"], []).append(d)
    for src, ds in sorted(by_src.items(), key=lambda kv: -len(kv[1])):
        describe(f"translation, pose src {src}", [d["mm"] for d in ds], " mm")
    describe("translation, HOLD frames", [d["mm"] for d in trk if d["hold"]], " mm")
    describe("translation, SNAP frames", [d["mm"] for d in data if d["snap"]], " mm")

    qd = [d for d in trk if d["q"] is not None]
    if qd:
        rho = spearman([d["q"] for d in qd], [d["mm"] for d in qd])
        print(f"  Spearman(quality, err_mm) = {rho:+.2f}   (strongly negative = q is a useful accuracy proxy; ~0 = q says little)", flush=True)
        for lo, hi in ((0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)):
            sel = sorted(d["mm"] for d in qd if lo <= d["q"] < hi)
            if sel:
                print(f"    q in [{lo:.1f},{hi:.1f}): n={len(sel):5d}  err median={pct(sel,.5):5.2f} mm  p90={pct(sel,.9):5.2f} mm", flush=True)
        for thr in (2.0, 3.0, 5.0):
            bad = sum(1 for d in qd if d["q"] >= 0.6 and d["mm"] > thr)
            good_q = sum(1 for d in qd if d["q"] >= 0.6)
            if good_q:
                print(f"    q>=0.6 but err>{thr:.0f} mm: {bad}/{good_q} frames ({bad/good_q:.1%})", flush=True)


paths = sys.argv[1:] or sorted(glob.glob(os.path.join(logs, "tracking_*.csv")), key=os.path.getmtime)[-1:]
for p in paths:
    run(p)
