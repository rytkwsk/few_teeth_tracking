"""追跡ログ（logs/tracking_*.csv）を比較集計する。

使い方:
    python analyze_runs.py logs/tracking_A.csv=switching logs/tracking_B.csv=baseline
    python analyze_runs.py logs/tracking_*.csv          # ラベル省略可（anchor_switching 列があれば自動判定）
    python analyze_runs.py ... --start 60 --end 300      # 区間指定（秒、CSV 先頭からの相対）
    python analyze_runs.py ... --md out.md               # Markdown 表を書き出す

出力する項目（run ごと）:
    duration / fps
    usable      : 姿勢を出せたフレームの割合（TRACKING で hold でも SNAP でもない）
    hold / snap : 保持・再探索に落ちていた割合
    hold_eps    : 保持エピソード数と中央値・最長の継続時間（ガイドが止まって見えた時間）
    relocks     : 自動再位置合わせ（SNAP 開始）の回数
    jumps       : 1 フレームで 15 mm 超 / 30 mm 超 跳んだ回数
    switches    : 姿勢源の切替回数と、歯群が姿勢を出した割合
    group_shift : 歯群エピソード中の歯列中心のずれ（遮蔽直前の F 姿勢基準、模型静止前提）中央値 / 最悪 [mm]
    aruco_F     : 歯列全体出力時の ArUco 基準との差 中央値 / p90 [mm]（整合性の確認。真値ではない）
"""
import argparse
import csv
import glob
import os
import sys
import numpy as np

CENTROID = np.array([-0.0096, 0.0145, -0.0157])  # arch_4groups_smooth.obj の重心（モデル座標, m）


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def rot_of(r):
    return np.array([[fl(r[f"r{i}{j}"]) for j in range(3)] for i in range(3)])


def rot_angle(Ra, Rb):
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def source_of(mode):
    if mode.startswith("SNAP"):
        return "snap"
    if "hold" in mode:
        return "hold"
    if mode.startswith("TRACKING"):
        return mode.split("[")[-1].rstrip("]")
    return "other"


def episodes(flags):
    """True が連続する区間 [(start, end_exclusive), ...]"""
    out, i, n = [], 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def analyze(path, label=None, t_start=None, t_end=None):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return None
    ts = np.array([fl(r["timestamp"]) for r in rows])
    t0 = ts[0]
    if t_start is not None or t_end is not None:
        lo = t0 + (t_start or 0.0)
        hi = t0 + (t_end if t_end is not None else 1e18)
        keep = (ts >= lo) & (ts <= hi)
        rows = [r for r, k in zip(rows, keep) if k]
        ts = ts[keep]
    n = len(rows)
    if n < 10:
        return None
    dur = float(ts[-1] - ts[0])
    src = [source_of(r["mode"]) for r in rows]
    if label is None:
        col = rows[0].get("anchor_switching")
        if col is not None and col != "":
            label = {"0": "baseline", "1": "best-one", "2": "fusion"}.get(col, col)
        else:
            label = os.path.basename(path)
    T = np.array([[fl(r["tx"]), fl(r["ty"]), fl(r["tz"])] for r in rows])
    R = [rot_of(r) for r in rows]
    cen = np.array([R[i] @ CENTROID + T[i] for i in range(n)])

    def is_group(s):   # "A", "D", "A+D" など（統合モードは "+" 区切り）
        return s not in ("F", "hold", "snap", "other") and all(ch in "ABCD+" for ch in s)
    usable = np.array([s == "F" or is_group(s) for s in src])
    hold = np.array([s == "hold" for s in src])
    snap = np.array([s == "snap" for s in src])
    hold_eps = episodes(hold | snap)
    hold_durs = [float(ts[e - 1] - ts[s]) + (dur / n) for s, e in hold_eps]
    relocks = sum(1 for a, b in zip(src, src[1:]) if a != "snap" and b == "snap")
    step = np.linalg.norm(np.diff(cen, axis=0), axis=1) * 1000.0
    jumps15 = int(np.nansum(step > 15))
    jumps30 = int(np.nansum(step > 30))
    tracked = [s for s in src if s == "F" or is_group(s)]
    switches = sum(1 for a, b in zip(tracked, tracked[1:]) if a != b)
    group_share = sum(1 for s in src if is_group(s)) / n

    # 歯群エピソード中の歯列中心のずれ（遮蔽直前 10 フレームの F 姿勢を基準）
    shifts, shift_worst, group_frames = [], [], {}
    for s0, e0 in episodes(np.array([is_group(s) for s in src])):
        base = [m for m in range(max(0, s0 - 30), s0) if src[m] == "F"][-10:]
        if len(base) < 5:
            continue
        cb = cen[base].mean(axis=0)
        dev = np.linalg.norm(cen[s0:e0] - cb, axis=1) * 1000.0
        shifts.append(float(np.median(dev)))
        shift_worst.append(float(dev.max()))
        key = src[s0]
        group_frames[key] = group_frames.get(key, 0) + (e0 - s0)

    eF = np.array([fl(r["err_mm"]) for r, s in zip(rows, src) if s == "F"])
    eF = eF[~np.isnan(eF)]

    return dict(
        label=label, path=path, frames=n, duration=dur, fps=n / dur if dur > 0 else float("nan"),
        usable=usable.mean(), hold=hold.mean(), snap=snap.mean(),
        hold_eps=len(hold_eps),
        hold_med=float(np.median(hold_durs)) if hold_durs else 0.0,
        hold_max=float(max(hold_durs)) if hold_durs else 0.0,
        relocks=relocks, jumps15=jumps15, jumps30=jumps30,
        switches=switches, group_share=group_share, group_frames=group_frames,
        shift_med=float(np.median(shifts)) if shifts else float("nan"),
        shift_worst=float(max(shift_worst)) if shift_worst else float("nan"),
        shift_eps=len(shifts),
        arucoF_med=float(np.median(eF)) if len(eF) else float("nan"),
        arucoF_p90=float(np.percentile(eF, 90)) if len(eF) else float("nan"),
        arucoF_n=int(len(eF)),
    )


def fmt(v, spec):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def table(results):
    rows = [
        ("項目", [r["label"] for r in results]),
        ("継続時間 [s] / fps", [f"{r['duration']:.0f} / {r['fps']:.1f}" for r in results]),
        ("姿勢を出せたフレーム", [f"{r['usable']*100:.1f}%" for r in results]),
        ("保持（hold）の割合", [f"{r['hold']*100:.1f}%" for r in results]),
        ("再探索（SNAP）の割合", [f"{r['snap']*100:.1f}%" for r in results]),
        ("保持エピソード 数 / 中央値 / 最長 [s]", [f"{r['hold_eps']} / {r['hold_med']:.1f} / {r['hold_max']:.1f}" for r in results]),
        ("自動再位置合わせ回数", [str(r["relocks"]) for r in results]),
        ("1 フレームの跳び >15 mm / >30 mm", [f"{r['jumps15']} / {r['jumps30']}" for r in results]),
        ("姿勢源の切替回数", [str(r["switches"]) for r in results]),
        ("歯群が姿勢を出した割合", [f"{r['group_share']*100:.1f}%" for r in results]),
        ("遮蔽中のずれ 中央値 / 最悪 [mm]（歯群エピソード数）", [f"{fmt(r['shift_med'], '.1f')} / {fmt(r['shift_worst'], '.1f')} ({r['shift_eps']})" for r in results]),
        ("ArUco 整合 F 中央値 / p90 [mm]（n）", [f"{fmt(r['arucoF_med'], '.1f')} / {fmt(r['arucoF_p90'], '.1f')} ({r['arucoF_n']})" for r in results]),
    ]
    head = "| " + " | ".join(rows[0][1]) + " |"
    head = "| " + rows[0][0] + " | " + " | ".join(rows[0][1]) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in results) + " |"
    lines = [head, sep]
    for name, vals in rows[1:]:
        lines.append("| " + name + " | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="CSV パス、または CSV=ラベル")
    ap.add_argument("--start", type=float, default=None, help="集計開始 [s]（CSV 先頭からの相対）")
    ap.add_argument("--end", type=float, default=None, help="集計終了 [s]")
    ap.add_argument("--md", default=None, help="Markdown 表の出力先")
    args = ap.parse_args()

    results = []
    for item in args.runs:
        path, label = (item.split("=", 1) + [None])[:2] if "=" in item else (item, None)
        for p in sorted(glob.glob(path)) or [path]:
            if not os.path.exists(p):
                print(f"見つかりません: {p}", file=sys.stderr)
                continue
            r = analyze(p, label, args.start, args.end)
            if r is None:
                print(f"フレームが足りません: {p}", file=sys.stderr)
                continue
            results.append(r)
    if not results:
        sys.exit(1)

    md = table(results)
    print(md)
    for r in results:
        print(f"\n[{r['label']}] {r['path']}  frames={r['frames']}  歯群フレーム={r['group_frames']}")
    print("\n注: 「遮蔽中のずれ」は模型が静止している前提で、各歯群エピソード直前 10 フレームの歯列全体の姿勢を基準にした"
          "歯列中心の移動量。ArUco 整合は追跡器自身の姿勢から登録した基準との差で、絶対精度ではない。")
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"\nMarkdown を書き出しました: {args.md}")


if __name__ == "__main__":
    main()
