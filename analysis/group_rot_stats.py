# -*- coding: utf-8 -*-
"""⑤ 歯群ごとの回転精度 (gA/gB/gD_rot_deg, *_cen_mm, *_cen_depth_mm)、① 復帰時再探索 (event 列)、
② 警告 (rot_uncertain 列) の集計。ArUco の誤検出フレーム (err_mm > 60) は除外する。
usage: python group_rot_stats.py [tracking_*.csv]   (省略時は最新)"""
import csv, glob, math, os, sys

logs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
FLIP_MM = 60.0


def fl(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None


def pct(s, q):
    return s[min(len(s) - 1, int(len(s) * q))]


def describe(label, xs, unit):
    if not xs:
        print(f"  {label:34s} n=0", flush=True); return
    s = sorted(xs)
    print(f"  {label:34s} n={len(s):5d}  median={pct(s,.5):6.2f}{unit}  p90={pct(s,.9):6.2f}{unit}  max={s[-1]:6.2f}{unit}", flush=True)


def valid(r):
    mm = fl(r.get("err_mm"))
    return mm is not None and mm <= FLIP_MM


def run(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    t0 = float(rows[0]["timestamp"]) if rows else 0.0
    has = rows and "gA_rot_deg" in rows[0]
    has_warn = rows and "rot_uncertain" in rows[0]
    print(f"=== {os.path.basename(path)}  rows={len(rows)}  group cols={bool(has)}  warn col={bool(has_warn)}", flush=True)
    if not has:
        return
    flips = sum(1 for r in rows if fl(r.get("err_mm")) is not None and fl(r.get("err_mm")) > FLIP_MM)
    print(f"  ArUco flip frames excluded: {flips}", flush=True)

    evs = [(float(r["timestamp"]) - t0, r["event"]) for r in rows if r.get("event")]
    n_search = sum(1 for _, e in evs if e.startswith("return_search"))
    n_skip = sum(1 for _, e in evs if e.startswith("return_no_search"))
    print(f"  F returns: {len(evs)}  (searched {n_search}, skipped {n_skip})", flush=True)
    for t, e in evs:
        if not e.startswith("return_search"):
            continue
        before = [fl(r["err_deg"]) for r in rows if valid(r) and t - 1.5 <= float(r["timestamp"]) - t0 < t]
        after = [fl(r["err_deg"]) for r in rows if valid(r) and t + 1.0 <= float(r["timestamp"]) - t0 < t + 3.0]
        b = f"{sorted(before)[len(before)//2]:.1f}" if before else "-"
        a = f"{sorted(after)[len(after)//2]:.1f}" if after else "-"
        print(f"    t={t:6.1f}s  {e}   rot err before {b} -> after {a} deg", flush=True)

    if has_warn:
        w = [r for r in rows if valid(r) and r["mode"].startswith("TRACKING")]
        on = [fl(r["err_deg"]) for r in w if r["rot_uncertain"] == "1"]
        off = [fl(r["err_deg"]) for r in w if r["rot_uncertain"] != "1"]
        print(f"  -- ② warning coverage: ON {len(on)} frames ({len(on)/max(len(w),1):.1%}), OFF {len(off)}", flush=True)
        describe("rot err while warning ON", on, " deg")
        describe("rot err while warning OFF", off, " deg")
        for thr in (5.0, 10.0):
            big_on = sum(1 for x in on if x > thr); big_off = sum(1 for x in off if x > thr)
            print(f"    frames with rot err > {thr:.0f} deg: warning ON {big_on}, OFF {big_off}  "
                  f"(OFF = silent error)", flush=True)

    for state, pred in (("F tracking (unoccluded)", lambda m: m.startswith("TRACKING[F]")),
                        ("group/fusion mode", lambda m: m.startswith("TRACKING[") and not m.startswith("TRACKING[F]"))):
        sel = [r for r in rows if pred(r["mode"]) and valid(r) and fl(r.get("gA_rot_deg")) is not None]
        print(f"  -- {state}: n={len(sel)}", flush=True)
        for k in ("A", "B", "D"):
            describe(f"{k}: own rotation error", [fl(r[f"g{k}_rot_deg"]) for r in sel if fl(r[f"g{k}_rot_deg"]) is not None], " deg")
            describe(f"{k}: centroid error (3D)", [fl(r[f"g{k}_cen_mm"]) for r in sel if fl(r[f"g{k}_cen_mm"]) is not None], " mm")
            describe(f"{k}: centroid depth component", [abs(fl(r[f"g{k}_cen_depth_mm"])) for r in sel if fl(r[f"g{k}_cen_depth_mm"]) is not None], " mm")
        describe("F: output rotation error", [fl(r["gF_rot_deg"]) for r in sel if fl(r["gF_rot_deg"]) is not None], " deg")


paths = sys.argv[1:] or sorted(glob.glob(os.path.join(logs, "tracking_*.csv")), key=os.path.getmtime)[-1:]
for p in paths:
    run(p)
