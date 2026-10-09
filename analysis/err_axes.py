import csv, math, glob, os
logs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
p = sorted(glob.glob(os.path.join(logs, "tracking_*.csv")), key=os.path.getmtime)[-1]
rows = list(csv.DictReader(open(p, encoding="utf-8", newline="")))
def fl(x):
    try:
        v=float(x); return None if math.isnan(v) else v
    except: return None
d = []
for r in rows:
    t = [fl(r[k]) for k in ("tx","ty","tz")]; ref = [fl(r[k]) for k in ("ref_tx","ref_ty","ref_tz")]
    if None in t or None in ref: continue
    d.append([(a-b)*1000 for a,b in zip(t,ref)] + [float(r["timestamp"])])
n = len(d); print(f"{os.path.basename(p)}  n={n}")
import statistics as st
for i, ax in enumerate(("x (left/right)","y (up/down)","z (depth, along camera ray)")):
    v=[row[i] for row in d]
    print(f"  {ax:30s} mean(bias)={st.mean(v):+6.2f} mm   std(jitter)={st.pstdev(v):5.2f} mm")
# frame-to-frame jitter (high-frequency) vs slow drift
dd=[math.dist(d[k][:3], d[k-1][:3]) for k in range(1,n)]
s=sorted(dd); print(f"  frame-to-frame change: median={s[len(s)//2]:.2f} mm  p90={s[int(len(s)*.9)]:.2f} mm")
# distance to camera
z=[fl(r["tz"]) for r in rows if fl(r["tz"]) is not None]; print(f"  camera distance: median {sorted(z)[len(z)//2]*100:.1f} cm")
