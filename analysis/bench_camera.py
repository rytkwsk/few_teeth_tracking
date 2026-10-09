import cv2, time


def bench(label, backend, w, h, mjpg=False, buf=None, n=60):
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        print(f"{label:30s} OPEN FAILED", flush=True)
        return
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if buf is not None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buf)
    for _ in range(10):
        cap.read()
    t = time.perf_counter()
    ok = 0
    for _ in range(n):
        r, _f = cap.read()
        ok += int(r)
    dt = (time.perf_counter() - t) / n * 1000.0
    aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fs = "".join(chr((fcc >> (8 * i)) & 0xFF) for i in range(4))
    print(f"{label:30s} {aw}x{ah} fourcc={fs!r:8s} {dt:5.1f} ms/frame = {1000.0/dt:5.1f} fps  (ok {ok}/{n})", flush=True)
    cap.release()


bench("DSHOW 720p default",      cv2.CAP_DSHOW, 1280, 720)
bench("DSHOW 720p MJPG",         cv2.CAP_DSHOW, 1280, 720, mjpg=True)
bench("DSHOW 720p MJPG buf=1",   cv2.CAP_DSHOW, 1280, 720, mjpg=True, buf=1)
bench("DSHOW 640x480 default",   cv2.CAP_DSHOW, 640, 480)
bench("MSMF  720p default",      cv2.CAP_MSMF, 1280, 720)
print("done", flush=True)
