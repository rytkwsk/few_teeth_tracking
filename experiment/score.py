# -*- coding: utf-8 -*-
"""再生結果の採点（ArUco 基準，2026-10-21）。

    python experiment/score.py 20261021_home --tags base depth5

各クリップの replay_<tag>.csv を events.csv の区間（lead / occlude / move / hold / release ...）で切り、
試行の種類 × 区間ごとに、ArUco との誤差が 2 mm かつ 2° 未満のフレームの割合、p90、最大を出す。
release の最初の 1 s（F への復帰中）は "release" から外して "return" として別に数える。
ArUco が見えないフレーム・1 フレームの反転（>100 mm）は除く。--per-clip でクリップごとの全体値も出す。
"""
import argparse
import csv
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(HERE, "sessions")
GOAL_MM, GOAL_DEG = 2.0, 2.0
RETURN_SEC = 1.0


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def load_events(clip):
    ev = []
    with open(os.path.join(clip, "events.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ev.append((float(r["timestamp"]), r["event"]))
    return ev


def phase_of(t, events):
    cur, t_cur = "pre", None
    for te, name in events:
        if t >= te:
            cur, t_cur = name, te
    if cur == "release" and t_cur is not None and t - t_cur < RETURN_SEC:
        return "return"
    return cur


def clip_rows(clip, tag):
    path = os.path.join(clip, f"replay_{tag}.csv")
    if not os.path.exists(path):
        return None
    events = load_events(clip)
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            mm, deg = fnum(r["err_mm"]), fnum(r["err_deg"])
            if not np.isfinite(mm) or mm > 100:
                continue
            out.append((phase_of(fnum(r["timestamp"]), events), mm, deg))
    return out


def summarize(vals):
    if not vals:
        return None
    mm = np.array([v[0] for v in vals]); deg = np.array([v[1] for v in vals])
    ok = (mm < GOAL_MM) & (deg < GOAL_DEG)
    return dict(n=len(mm), ok=100 * ok.mean(), p90mm=np.percentile(mm, 90), p90deg=np.percentile(deg, 90),
                maxmm=mm.max(), maxdeg=deg.max())


def compare(clips, tags):
    """種類ごと（遮蔽中 = occlude/move/hold、復帰後 = return/release、遮蔽なし = lead/still/move(遮蔽なし試行)）に tag を並べる。"""
    occl = {"occlude", "move", "hold"}
    post = {"return", "release"}
    print(f"{'種類':12s} {'区間':6s} " + " ".join(f"{t:>30s}" for t in tags))
    print(" " * 20 + " ".join(f"{'ok%  p90mm p90°  max mm/°':>30s}" for _ in tags))
    groups = {}
    for tag in tags:
        for clip in clips:
            rows = clip_rows(clip, tag)
            if rows is None:
                continue
            kind = os.path.basename(clip).rsplit("_", 1)[0]
            for ph, mm, deg in rows:
                if kind in ("static", "slow"):
                    g = "visible" if ph in ("lead", "still", "move") else ("post" if ph in post else None)
                else:
                    g = "occl" if ph in occl else ("post" if ph in post else ("pre" if ph == "lead" else None))
                if g:
                    groups.setdefault((kind, g, tag), []).append((mm, deg))
                    groups.setdefault(("ALL", g, tag), []).append((mm, deg))
    keys = sorted({(k, g) for k, g, _ in groups}, key=lambda x: (x[0] == "ALL", x[0], x[1]))
    for kind, g in keys:
        cells = []
        for tag in tags:
            s = summarize(groups.get((kind, g, tag), []))
            cells.append(f"{s['ok']:5.1f} {s['p90mm']:5.1f} {s['p90deg']:5.1f} {s['maxmm']:5.1f}/{s['maxdeg']:4.1f}" if s else " " * 30)
        print(f"{kind:12s} {g:6s} " + " ".join(f"{c:>30s}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--per-clip", action="store_true")
    ap.add_argument("--compare", action="store_true", help="tag を横に並べた要約だけ出す")
    args = ap.parse_args()
    clips = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(SESSIONS_DIR, args.session, "*", "events.csv")))
    if args.compare:
        compare(clips, args.tags)
        return
    order = ["lead", "still", "occlude", "move", "hold", "return", "release"]
    for tag in args.tags:
        print(f"\n=== tag={tag}   （{GOAL_MM:.0f} mm・{GOAL_DEG:.0f}° 未満の割合 / p90 mm・deg / 最大 mm・deg）")
        agg = {}
        for clip in clips:
            rows = clip_rows(clip, tag)
            if rows is None:
                continue
            kind = os.path.basename(clip).rsplit("_", 1)[0]
            for ph, mm, deg in rows:
                agg.setdefault((kind, ph), []).append((mm, deg))
                agg.setdefault(("ALL", "occluded" if ph in ("occlude", "move", "hold") else "visible"), []).append((mm, deg))
            if args.per_clip:
                s = summarize([(mm, deg) for _, mm, deg in rows])
                print(f"  {os.path.basename(clip):16s} ok {s['ok']:5.1f}%  p90 {s['p90mm']:5.1f} mm {s['p90deg']:5.1f}°"
                      f"  max {s['maxmm']:5.1f} mm {s['maxdeg']:5.1f}°")
        kinds = sorted({k for k, _ in agg if k != "ALL"}) + ["ALL"]
        for kind in kinds:
            phs = sorted({p for k, p in agg if k == kind}, key=lambda p: order.index(p) if p in order else 99)
            for ph in phs:
                s = summarize(agg[(kind, ph)])
                print(f"  {kind:10s} {ph:9s} n={s['n']:5d}  ok {s['ok']:5.1f}%  p90 {s['p90mm']:5.1f} mm {s['p90deg']:5.1f}°"
                      f"  max {s['maxmm']:5.1f} mm {s['maxdeg']:5.1f}°")


if __name__ == "__main__":
    main()
