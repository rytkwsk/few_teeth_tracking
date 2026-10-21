# -*- coding: utf-8 -*-
"""録画クリップをトラッカーで再生する（オフライン調整，2026-10-21）。

    python experiment/replay.py 20261021_home --tag base
    python experiment/replay.py 20261021_home --tag depth5 --env M3T_XXX=5 --env M3T_YYY=1
    python experiment/replay.py 20261021_home/occ_rot_01 --tag base        # 1 本だけ

クリップごとにトラッカーを 1 プロセス起動し（状態の持ち越しなし）、<クリップ>/replay_<tag>.csv に CSV を、
<クリップ>/replay_<tag>.log に標準出力を書く。-j で並列数。採点は experiment/score.py。
トラッカーは conda 環境（M3T_CONDA_ENV、既定 pym3t39）の python で、Unity と同じく PYM3T_DIR を引き継いで動かす。
"""
import argparse
import concurrent.futures as cf
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SESSIONS_DIR = os.path.join(HERE, "sessions")
DUMP = False


def _user_env(name):
    """ユーザー環境変数（setup_pc.ps1 が設定した M3T_CONDA_ENV / PYM3T_DIR）。起動前に開いたシェルにも効くように直接読む。"""
    if name in os.environ or os.name != "nt":
        return os.environ.get(name)
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, name)[0]
    except OSError:
        return None


for _k in ("M3T_CONDA_ENV", "PYM3T_DIR"):
    _v = _user_env(_k)
    if _v:
        os.environ[_k] = _v


def conda_env_python():
    env_name = os.environ.get("M3T_CONDA_ENV", "pym3t39")
    for root in (os.path.expanduser("~/anaconda3"), os.path.expanduser("~/miniconda3"),
                 os.path.expanduser("~/miniforge3"), r"C:\ProgramData\anaconda3", r"C:\ProgramData\miniconda3"):
        d = os.path.join(root, "envs", env_name)
        exe = os.path.join(d, "python.exe" if os.name == "nt" else "bin/python")
        if os.path.exists(exe):
            return d, exe
    sys.exit(f"conda 環境 {env_name} の python が見つからない")


def find_clips(targets):
    clips = []
    for t in targets:
        p = t if os.path.isabs(t) else os.path.join(SESSIONS_DIR, t)
        if os.path.exists(os.path.join(p, "video.avi")):
            clips.append(p)
        else:
            clips += sorted(os.path.dirname(v) for v in glob.glob(os.path.join(p, "*", "video.avi")))
    return [c for c in clips if os.path.exists(os.path.join(c, "start_pose.npy"))]


def run_one(clip, tag, extra_env, env_dir, exe):
    env = dict(os.environ)
    if os.name == "nt":   # conda activate 相当（DLL の検索パス）
        env["PATH"] = os.pathsep.join([env_dir, os.path.join(env_dir, "Library", "mingw-w64", "bin"),
                                       os.path.join(env_dir, "Library", "usr", "bin"),
                                       os.path.join(env_dir, "Library", "bin"), os.path.join(env_dir, "Scripts"),
                                       env.get("PATH", "")])
    env.update({"M3T_REPLAY_CLIP": clip, "M3T_REPLAY_LOG": os.path.join(clip, f"replay_{tag}.csv"),
                "M3T_HEADLESS": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
    env.update(extra_env)
    if DUMP:
        env["M3T_PNP_DUMP"] = os.path.join(clip, f"pnp_{tag}.pkl")
    t0 = time.time()
    with open(os.path.join(clip, f"replay_{tag}.log"), "w", encoding="utf-8") as logf:
        rc = subprocess.call([exe, "-u", os.path.join(REPO, "tracker_four_group.py")], cwd=REPO, env=env,
                             stdout=logf, stderr=subprocess.STDOUT)
    return clip, rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="セッション名 / セッション/クリップ / フォルダのパス")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE（繰り返し可）")
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--dump", action="store_true", help="特徴点 PnP の診断データを <クリップ>/pnp_<tag>.pkl に保存")
    args = ap.parse_args()
    global DUMP
    DUMP = args.dump
    extra = dict(kv.split("=", 1) for kv in args.env)
    clips = find_clips(args.targets)
    if not clips:
        sys.exit("クリップが見つからない")
    env_dir, exe = conda_env_python()
    print(f"{len(clips)} クリップを tag={args.tag} {extra or ''} で再生（並列 {args.j}）")
    with cf.ThreadPoolExecutor(args.j) as ex:
        for clip, rc, dt in ex.map(lambda c: run_one(c, args.tag, extra, env_dir, exe), clips):
            print(f"  {'OK' if rc == 0 else f'NG(rc={rc})'} {os.path.relpath(clip, SESSIONS_DIR)}  {dt:.0f} s")


if __name__ == "__main__":
    main()
