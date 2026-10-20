# -*- coding: utf-8 -*-
"""カメラの開き方と PC ごとの設定（m3t_local.env）を OS に依らず扱う共通部品。

tracker_four_group.py / tools/calibrate_camera.py / tools/list_cameras.py から使う。

設定の優先順位: プロセスの環境変数 > m3t_local.env（リポジトリ直下、gitignore）> 既定値
  M3T_CAMERA_INDEX    カメラ番号。Mac の連係カメラ（iPhone）は 0 とは限らないので tools/list_cameras.py で確かめる。
                      未設定のとき Windows（dshow）は M3T_CAMERA_NAME の名前で探し、見つからなければ 0、他の OS は 0
  M3T_CAMERA_NAME     Windows で番号を名前（部分一致、大小無視）で探すときの名前（既定 iVCam。pygrabber を使う）。
                      DirectShow の番号は PC・接続状況で変わる（OMEN では内蔵カメラと iVCam の番号が入れ替わった）
  M3T_CAMERA_BACKEND  dshow / msmf / avfoundation / v4l2 / any（既定: Windows=dshow, Mac=avfoundation, 他=any）
  M3T_CAMERA_WIDTH    要求する幅（既定 1280）。キャリブ（camera_calibration.npz）の解像度と一致させること
  M3T_CAMERA_HEIGHT   要求する高さ（既定 720）
Mac では GUI から起動した Unity にシェルの環境変数が渡らないので、PC ごとの値は m3t_local.env に書く
（tools/setup_mac.sh が作る）。Windows でも使える。
"""
import os
import sys

import cv2

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_ENV_PATH = os.path.join(REPO_DIR, "m3t_local.env")

_BACKENDS = {
    "dshow": getattr(cv2, "CAP_DSHOW", 0),
    "msmf": getattr(cv2, "CAP_MSMF", 0),
    "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", 0),
    "v4l2": getattr(cv2, "CAP_V4L2", 0),
    "any": getattr(cv2, "CAP_ANY", 0),
}


def load_local_env(path=LOCAL_ENV_PATH):
    """KEY=VALUE 形式（# はコメント）の m3t_local.env を読み、まだ設定されていない環境変数だけを埋める。
    読み込んだキーの一覧を返す。"""
    loaded = []
    if not os.path.isfile(path):
        return loaded
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and value and not os.environ.get(key):
                os.environ[key] = os.path.expanduser(value)
                loaded.append(key)
    return loaded


def default_backend_name():
    if sys.platform == "win32":
        return "dshow"
    if sys.platform == "darwin":
        return "avfoundation"
    return "any"


def camera_backend():
    name = os.environ.get("M3T_CAMERA_BACKEND", "").strip().lower() or default_backend_name()
    if name not in _BACKENDS:
        print(f"M3T_CAMERA_BACKEND={name} は不明なので {default_backend_name()} を使う")
        name = default_backend_name()
    return name, _BACKENDS[name]


def camera_index(default=0):
    value = os.environ.get("M3T_CAMERA_INDEX", "").strip()
    if value:
        try:
            return int(value)
        except ValueError:
            print(f"M3T_CAMERA_INDEX={value} は数字でないので無視する")
    if camera_backend()[0] == "dshow":
        found = _find_dshow_index_by_name(os.environ.get("M3T_CAMERA_NAME", "iVCam"))
        if found is not None:
            return found
    return default


def _find_dshow_index_by_name(name):
    """DirectShow の入力デバイス一覧から名前を含む最初の番号を返す。一覧が取れない／見つからなければ None。"""
    try:
        from pygrabber.dshow_graph import FilterGraph
        devices = FilterGraph().get_input_devices()
    except Exception as e:  # noqa: BLE001
        print(f"カメラ一覧を取得できない（{e}）→ M3T_CAMERA_INDEX か既定の番号を使う")
        return None
    for index, device in enumerate(devices):
        if name.lower() in device.lower():
            print(f"カメラ: {index} 番 '{device}' を使用（M3T_CAMERA_NAME='{name}'）")
            return index
    print(f"カメラ: '{name}' が見つからない。一覧: {list(enumerate(devices))}")
    return None


def camera_size(default=(1280, 720)):
    try:
        w = int(os.environ.get("M3T_CAMERA_WIDTH", default[0]))
        h = int(os.environ.get("M3T_CAMERA_HEIGHT", default[1]))
        return w, h
    except ValueError:
        return default


def open_camera(index=None, width=None, height=None, verbose=True):
    """OS に合ったバックエンドでカメラを開き、解像度を要求する。返り値: (cap, (実際の幅, 高さ))。
    開けなければ RuntimeError（Mac ではカメラの使用許可がないときもここで失敗する）。"""
    if index is None:
        index = camera_index()
    if width is None or height is None:
        width, height = camera_size()
    name, api = camera_backend()
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened():
        hint = ""
        if sys.platform == "darwin":
            hint = ("（Mac: システム設定 → プライバシーとセキュリティ → カメラ で、起動元のアプリ"
                    "〔ターミナル／Unity〕を許可したか、M3T_CAMERA_INDEX が正しいか確認）")
        raise RuntimeError(f"カメラ {index} を開けない（backend={name}）{hint}")
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if verbose:
        print(f"カメラ {index} を開いた（backend={name}、要求 {width}x{height} → 実際 {aw}x{ah}）")
        if width and height and (aw, ah) != (width, height):
            print("  要求した解像度にならなかった。M3T_CAMERA_WIDTH/HEIGHT を実際の値に合わせ、その解像度でキャリブすること")
    return cap, (aw, ah)
