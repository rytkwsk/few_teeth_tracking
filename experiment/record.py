# -*- coding: utf-8 -*-
"""オフライン調整用の録画ガイド（2026-10-21）。

Unity でトラッカーを起動し、位置合わせして追跡を始めてから（F で合っている状態）、別の PowerShell で:

    python experiment/record.py --session 20261021_home

番号を選ぶと試行を 1 本録画する。トラッカーに UDP（127.0.0.1:5006）で record_start / event / record_stop を送り、
画面の指示とビープで動きのタイミングを合わせる。録画は experiment/sessions/<session>/<種類>_<番号>/ に入る。
再生は experiment/replay.py、採点は experiment/score.py。

注意: ArUco ボードは隠さない（誤差の基準）。手はボードと反対側から入れる。
"""
import argparse
import os
import socket
import sys
import time

try:
    import winsound
except ImportError:            # Mac
    winsound = None

HOST, PORT = "127.0.0.1", 5006
SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")

# 種類: (説明, [(区間名, 秒, 指示), ...])
# 最初の lead は F で追える状態（PnP のキーフレームと F の見張りがここで作られる）。
TRIALS = {
    "1": ("static", "遮蔽なし・静止（基準の確認）", [
        ("lead", 3, "そのまま（触らない）"),
        ("still", 9, "そのまま（触らない）"),
    ]),
    "2": ("slow", "遮蔽なし・ゆっくり移動", [
        ("lead", 3, "そのまま"),
        ("move", 8, "台座を持って 1〜2 cm ゆっくり動かす・少し回す（歯は隠さない）"),
        ("release", 4, "手を離して静止"),
    ]),
    "3": ("occ_static", "遮蔽中・静止", [
        ("lead", 3, "そのまま"),
        ("occlude", 2, "手（またはハンドピース）で歯列の一部を隠す"),
        ("hold", 6, "隠したまま動かさない"),
        ("release", 4, "手を離して静止"),
    ]),
    "4": ("occ_trans", "遮蔽中・平行移動", [
        ("lead", 3, "そのまま"),
        ("occlude", 2, "片手で歯列の一部を隠す"),
        ("move", 5, "隠したまま、もう片方の手で台座を 1〜2 cm 横・前後にずらす（回さない）"),
        ("hold", 2, "止めて隠したまま"),
        ("release", 4, "手を離して静止"),
    ]),
    "5": ("occ_rot", "遮蔽中・回転", [
        ("lead", 3, "そのまま"),
        ("occlude", 2, "片手で歯列の一部を隠す"),
        ("move", 5, "隠したまま、台座を机の上で 10〜25° 回す"),
        ("hold", 2, "止めて隠したまま"),
        ("release", 4, "手を離して静止"),
    ]),
    # 大きく動かす版（2026-10-21 追加。5 s では 10° 前後しか回せなかった）
    "6": ("occ_transL", "遮蔽中・大きく平行移動（12 s）", [
        ("lead", 3, "そのまま"),
        ("occlude", 2, "片手で歯列の一部を隠す"),
        ("move", 12, "隠したまま、台座を 2〜3 cm ずらす（左右・前後に往復してもよい。回さない）"),
        ("hold", 3, "止めて隠したまま"),
        ("release", 4, "手を離して静止"),
    ]),
    "7": ("occ_rotL", "遮蔽中・大きく回転（12 s）", [
        ("lead", 3, "そのまま"),
        ("occlude", 2, "片手で歯列の一部を隠す"),
        ("move", 12, "隠したまま、台座を机の上で 20〜30° 回す（往復してもよい）"),
        ("hold", 3, "止めて隠したまま"),
        ("release", 4, "手を離して静止"),
    ]),
    "8": ("occ_mixL", "遮蔽中・回転＋移動（12 s）", [
        ("lead", 3, "そのまま"),
        ("occlude", 2, "片手で歯列の一部を隠す"),
        ("move", 12, "隠したまま、台座を回しながらずらす（20° ・2 cm 程度）"),
        ("hold", 3, "止めて隠したまま"),
        ("release", 4, "手を離して静止"),
    ]),
}


def beep(freq=880, ms=150):
    if winsound is not None:
        winsound.Beep(freq, ms)
    else:
        sys.stdout.write("\a")
        sys.stdout.flush()


class Tracker:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))

    def _drain(self):
        """前のコマンドへの遅れた返事を捨てる（トラッカーが再ロック等で数秒止まると返事が遅れて届く）。"""
        self.sock.setblocking(False)
        try:
            while True:
                self.sock.recvfrom(1024)
        except (BlockingIOError, ConnectionResetError, OSError):
            pass
        self.sock.setblocking(True)

    def send(self, text, timeout=8.0):
        """コマンドを送り、そのコマンドへの返事（status → "status:"、ほか → "ok:<コマンド名>" か "error:"）を待つ。"""
        self._drain()
        self.sock.sendto(text.encode("utf-8"), (HOST, PORT))
        name = text.split(":", 1)[0]
        want = "status:" if name == "status" else f"ok:{name}"
        t_end = time.time() + timeout
        while True:
            left = t_end - time.time()
            if left <= 0:
                return "error:no_reply"
            self.sock.settimeout(left)
            try:
                data, _ = self.sock.recvfrom(1024)
            except socket.timeout:
                return "error:no_reply"
            except ConnectionResetError:
                continue
            r = data.decode("utf-8")
            if r.startswith(want) or r.startswith("error:"):
                return r


def next_name(session, kind):
    d = os.path.join(SESSIONS_DIR, session)
    n = 1
    while os.path.exists(os.path.join(d, f"{kind}_{n:02d}")):
        n += 1
    return f"{kind}_{n:02d}"


def run_trial(tr, session, kind, phases):
    st = tr.send("status")
    if not st.startswith("status:") or "tracking=1" not in st:
        print(f"  トラッカーが追跡中ではない（{st}）。Unity で位置合わせして追跡を開始してから。")
        return
    if "recording=1" in st:
        tr.send("record_stop")   # 前の試行の録画が残っていたら止める
    name = next_name(session, kind)
    r = tr.send(f"record_start:{session}/{name}")
    if not r.startswith("ok"):
        if r == "error:no_reply":
            tr.send("record_stop")   # 返事が届かなかっただけで始まっていることがある
        print(f"  録画を開始できない: {r}（F で合って安定しているときに。統合中・再ロック中は不可）")
        return
    print(f"  ● 録画中: {session}/{name}")
    try:
        for i, (phase, sec, text) in enumerate(phases):
            tr.send(f"event:{phase}")
            beep(1200 if i > 0 else 660, 200)
            t_end = time.time() + sec
            while True:
                left = t_end - time.time()
                if left <= 0:
                    break
                sys.stdout.write(f"\r    [{phase:8s}] {text}  残り {left:4.1f} s   ")
                sys.stdout.flush()
                time.sleep(0.1)
            print()
    finally:
        tr.send("event:end")
        r = tr.send("record_stop")
        beep(440, 400)
        print(f"  ■ 停止（{r}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=time.strftime("%Y%m%d") + "_home")
    args = ap.parse_args()
    tr = Tracker()
    print(f"セッション: {args.session}  （保存先 {os.path.join(SESSIONS_DIR, args.session)}）")
    print("ArUco ボードは隠さないこと。各試行の最初の 3 s は触らない（F で追える状態から始める）。")
    while True:
        print()
        for k, (kind, desc, _) in TRIALS.items():
            print(f"  {k}: {desc} [{kind}]")
        print("  s: トラッカーの状態   q: 終了")
        c = input("> ").strip().lower()
        if c == "q":
            break
        if c == "s":
            print("  " + tr.send("status"))
            continue
        if c in TRIALS:
            kind, desc, phases = TRIALS[c]
            run_trial(tr, args.session, kind, phases)


if __name__ == "__main__":
    main()
