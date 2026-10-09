import csv, glob, os
logs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
for p in sorted(glob.glob(os.path.join(logs, "tracking_*.csv"))):
    rows = list(csv.DictReader(open(p, encoding="utf-8", newline="")))
    gaps = []
    for a, b in zip(rows, rows[1:]):
        if a["mode"].startswith("TRACKING") and b["mode"].startswith("TRACKING"):
            dt = (float(b["timestamp"]) - float(a["timestamp"])) * 1000.0
            if 0 < dt < 5000:
                gaps.append(dt)
    if not gaps:
        print(os.path.basename(p), "no tracking gaps"); continue
    g = sorted(gaps); n = len(g)
    pct = lambda q: g[min(n-1, int(n*q))]
    big = sum(1 for x in gaps if x > 100); huge = sum(1 for x in gaps if x > 200)
    print(f"{os.path.basename(p)}: tracking intervals n={n}  median={pct(.5):.0f}ms  p90={pct(.9):.0f}ms  "
          f"p99={pct(.99):.0f}ms  max={g[-1]:.0f}ms  >100ms: {big} ({big/n:.1%})  >200ms: {huge}", flush=True)
