import csv, math, glob, os
logs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
p = sorted(glob.glob(os.path.join(logs, "tracking_*.csv")), key=os.path.getmtime)[-1]
rows = list(csv.DictReader(open(p, encoding="utf-8", newline="")))
def fl(x):
    try:
        v=float(x); return None if math.isnan(v) else v
    except: return None
t0 = float(rows[0]["timestamp"])
by = {}
seg = []; cur=None
for r in rows:
    mm, dg = fl(r["err_mm"]), fl(r["err_deg"])
    if mm is None: continue
    m = r["mode"]; src = m[m.find("[")+1:m.find("]")] if "[" in m else "?"
    by.setdefault(src, []).append(dg)
    t = float(r["timestamp"]) - t0
    if cur is None or cur[0] != src:
        if cur: seg.append(cur)
        cur = [src, t, t, [], []]
    cur[2] = t; cur[3].append(mm); cur[4].append(dg)
if cur: seg.append(cur)
print("rotation error by pose source:")
for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
    s = sorted(v); print(f"  {k:8s} n={len(s):5d} median={s[len(s)//2]:5.1f} deg  p90={s[int(len(s)*.9)]:5.1f}  max={s[-1]:5.1f}")
print("\nsegments (pose source runs >= 1 s):")
for src, a, b, mms, dgs in seg:
    if b - a < 1.0: continue
    sm, sd = sorted(mms), sorted(dgs)
    print(f"  {a:6.1f}-{b:6.1f}s  {src:7s} err {sm[0]:4.1f}..{sm[-1]:4.1f} mm (med {sm[len(sm)//2]:4.1f})   rot med {sd[len(sd)//2]:4.1f} deg")
