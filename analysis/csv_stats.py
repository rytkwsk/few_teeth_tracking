import csv, math, glob, os, sys

logs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")


def fl(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None


def stats(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    modes = {}
    per = {k: [] for k in ("qA", "qB", "qC", "qD", "qF")}
    have = groups_ok = F_ok = groups_ok_F_bad = 0
    abc = []
    for r in rows:
        m = r["mode"].split()[0]
        modes[m] = modes.get(m, 0) + 1
        q = {k: fl(r[k]) for k in per}
        if all(v is None for v in q.values()):
            continue
        have += 1
        for k, v in q.items():
            if v is not None:
                per[k].append(v)
        g = [q[k] for k in ("qA", "qB", "qC", "qD") if q[k] is not None]
        if g and q["qF"] is not None:
            bg = max(g)
            groups_ok += bg >= 0.46
            F_ok += q["qF"] >= 0.46
            groups_ok_F_bad += (bg >= 0.46 and q["qF"] < 0.40)
        if all(q[k] is not None for k in ("qA", "qB", "qC")):
            abc.append((q["qA"] + q["qB"] + q["qC"]) / 3.0)
    print(f"=== {os.path.basename(path)}  rows={len(rows)}  with-q={have}", flush=True)
    top = dict(sorted(modes.items(), key=lambda kv: -kv[1])[:6])
    snap = modes.get("SNAP", 0)
    print(f"modes: {top}   SNAP share={snap/max(len(rows),1):.1%}", flush=True)
    for k, v in per.items():
        if v:
            s = sorted(v)
            print(f"  {k}: n={len(v):5d} mean={sum(v)/len(v):.3f} median={s[len(s)//2]:.3f} "
                  f"p90={s[int(len(s)*0.9)]:.3f} frac>=0.46={sum(1 for x in v if x >= 0.46)/len(v):.1%}", flush=True)
    if abc:
        s = sorted(abc)
        print(f"  rigid(A,B,C mean): median={s[len(s)//2]:.3f} p90={s[int(len(s)*0.9)]:.3f} "
              f"frac>=0.40={sum(1 for x in abc if x >= 0.40)/len(abc):.1%}", flush=True)
    print(f"  best-group>=0.46: {groups_ok} | F>=0.46: {F_ok} | group ok & F<0.40: {groups_ok_F_bad}", flush=True)


paths = sys.argv[1:] or sorted(glob.glob(os.path.join(logs, "tracking_*.csv")), key=os.path.getmtime)[-1:]
for p in paths:
    stats(p)
