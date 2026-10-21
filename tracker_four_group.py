"""
few_teeth_tracking/tracker_four_group.py

4グループ Anchor Switching トラッカー。tracker_anchor_switching.py をベースに、
Blender で切り出した4グループの歯列モデルを使う。
  A: group_1_molar         (左側3大臼歯)
  B: group_2_premolar_canine (左側小臼歯+犬歯)
  C: group_3_canine_lateral  (左側犬歯+側切歯)
  D: group_4_incisor         (右側4本)
毎フレーム全アンカーを追跡し、FIT QUALITY が最も高いアンカーを Unity へ出力する。
どの向きでも少なくとも1グループが見えていれば追跡が途切れにくい。

モデルの準備:
    models/ に group_1_molar.obj ~ group_4_incisor.obj を置くこと。
    初回起動時に M3T が .meta ファイルを自動生成する（モデルが大きいので数分かかる）。
"""

import os
import sys

# Unity から起動されたとき（stdout がパイプ）は UTF-8 で出す。既定の cp932 だと Unity 側の
# ログが文字化けする（m3t_tracker.log）。コンソール直接起動時はそのまま。
if not sys.stdout.isatty():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# conda 環境を activate せずに envs\pym3t\python.exe を直接叩くと、numpy の MKL
# （Library\bin\mkl_rt.2.dll）が見つからず、最初の np.linalg 呼び出しで
# "Windows fatal exception: code 0xc06d007f"（traceback なしで即終了）になる（2026-09-21 に発生）。
# どちらの起動方法でも動くよう、numpy を import する前に DLL 検索パスへ追加する。
_conda_lib_bin = os.path.join(sys.prefix, "Library", "bin")
if os.name == "nt" and os.path.isdir(_conda_lib_bin):
    os.environ["PATH"] = _conda_lib_bin + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_conda_lib_bin)

# few_teeth_tracking は pym3t とは別ディレクトリに配置されているため、
# pym3t.pyd 等の所在地を明示的に指定する。
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
# PC ごとの設定（PYM3T_DIR・カメラ番号など）。Mac では GUI の Unity にシェルの環境変数が渡らないのでファイルで持つ。
# 環境変数が既にあればそちらを優先する（camera_io.py 参照）。
import camera_io  # noqa: E402
_local_env_keys = camera_io.load_local_env()
if _local_env_keys:
    print(f"m3t_local.env から読み込み: {', '.join(_local_env_keys)}")
PYM3T_DIR = os.path.abspath(os.environ.get("PYM3T_DIR", os.path.join(os.path.dirname(REPO_DIR), "pym3t")))
sys.path.insert(0, PYM3T_DIR)

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


# Unity から起動されたトラッカーはウィンドウのない裏のプロセスなので、Windows 11 が省電力制御
# （EcoQoS）をかけ、ハイブリッド CPU（OMEN の i9-12900HX など）では E コアに回されることがある。
# 2026-10-19 の OMEN では m3t_step 38 ms / quality 21 ms（研究室 PC は m3t_step 15 ms）で 13 fps しか出なかった。
# 自プロセスの省電力制御を切り、優先度を上げる。M3T_PRIORITY=normal / above（既定）/ high、
# M3T_POWER_THROTTLING=1 で省電力制御を Windows 任せに戻す。
def _configure_windows_performance():
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    process = kernel32.GetCurrentProcess()

    if os.environ.get("M3T_POWER_THROTTLING", "0") != "1":
        class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        ProcessPowerThrottling = 4
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
        PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4
        # ControlMask で指定した項目を StateMask=0 にする＝その省電力制御を常に OFF にする
        state = PROCESS_POWER_THROTTLING_STATE(
            1, PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, 0)
        kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        if kernel32.SetProcessInformation(process, ProcessPowerThrottling, ctypes.byref(state), ctypes.sizeof(state)):
            print("性能設定: 省電力制御（EcoQoS）を OFF")
        else:
            print(f"性能設定: 省電力制御の OFF に失敗（error {ctypes.get_last_error()}）")

    priority_classes = {"normal": 0x00000020, "above": 0x00008000, "high": 0x00000080}
    priority = os.environ.get("M3T_PRIORITY", "above").lower()
    if priority in priority_classes:
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        if kernel32.SetPriorityClass(process, priority_classes[priority]):
            print(f"性能設定: 優先度 {priority}")
        else:
            print(f"性能設定: 優先度 {priority} の設定に失敗（error {ctypes.get_last_error()}）")
    else:
        print(f"性能設定: M3T_PRIORITY='{priority}' は不明（normal / above / high）→ 変更しない")


_configure_windows_performance()

import csv
import json
import socket
import struct
import threading
import traceback
import time

import queue as _queue

import cv2
import numpy as np
import pym3t

# ─── 録画クリップの再生（オフライン調整，2026-10-21） ─────────────────────────────
# M3T_REPLAY_CLIP=<クリップのフォルダ>（experiment/record.py で録ったもの）を渡すと、カメラの代わりにその録画を読み、
# 録画開始時の姿勢から追跡を始め、最後のフレームで終了する。設定を変えて同じ映像を何度でも処理して比べるため。
# 滞在時間・クールダウンなどの時間依存の処理が録画時と同じに動くよう、time.time() は「いま処理しているフレームの
# 録画時刻」を返す。Unity への送信・ポートの待受・診断フレームの保存はしない（並列に何本も動かせる）。
# ログは M3T_REPLAY_LOG（無ければクリップのフォルダの replay.csv）。
REPLAY_CLIP_DIR = os.environ.get("M3T_REPLAY_CLIP", "")
REPLAY = bool(REPLAY_CLIP_DIR)
_replay_clock = [0.0]
if REPLAY:
    time.time = lambda: _replay_clock[0]

# ─── クラッシュ捕捉 ─────────────────────────────────────────────────────────
# 原因不明の「勝手に終了する」現象を調査するため、メインスレッド・別スレッド
# 双方の未処理例外を必ずファイルに記録する（標準出力のバッファリングで
# 見えなくなることがあるため、ファイルへの記録は open() 直後に flush する）。

_CRASH_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "crash.log")


def _write_crash_log(where, exc_type, exc_value, exc_tb):
    os.makedirs(os.path.dirname(_CRASH_LOG_PATH), exist_ok=True)
    with open(_CRASH_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n--- {where} crash at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        f.flush()
    print(f"[CRASH] {where}: {exc_type.__name__}: {exc_value}  (詳細: {_CRASH_LOG_PATH})", flush=True)


def _excepthook(exc_type, exc_value, exc_tb):
    _write_crash_log("main thread", exc_type, exc_value, exc_tb)
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def _threading_excepthook(args):
    _write_crash_log(f"thread '{args.thread.name}'", args.exc_type, args.exc_value, args.exc_traceback)


sys.excepthook = _excepthook
threading.excepthook = _threading_excepthook

# ─── 設定 ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 追跡する2つのクラスタモデル（犬歯を挟んで隣接し、範囲が重なる）。
# それぞれ tracker_premolar_canine.py / tracker_canine.py で個別に検証済みの
# 実寸・しきい値をそのまま引き継ぐ。
ANCHOR_A_NAME              = "group_1_molar.obj"
ANCHOR_A_SPHERE_RADIUS     = 0.030
ANCHOR_A_SHARP_EDGE_DEG    = 30.0
ANCHOR_A_COLOR             = (50, 50, 230)   # 赤（アクティブ時）

ANCHOR_B_NAME              = "group_2_premolar_canine.obj"
ANCHOR_B_SPHERE_RADIUS     = 0.025
ANCHOR_B_SHARP_EDGE_DEG    = 35.0
ANCHOR_B_COLOR             = (50, 220, 50)   # 緑（アクティブ時）

ANCHOR_C_NAME              = "group_3_canine_lateral.obj"
ANCHOR_C_SPHERE_RADIUS     = 0.025
ANCHOR_C_SHARP_EDGE_DEG    = 40.0
ANCHOR_C_COLOR             = (230, 130, 50)  # 青（アクティブ時）

ANCHOR_D_NAME              = "group_4_incisor.obj"
ANCHOR_D_SPHERE_RADIUS     = 0.028
ANCHOR_D_SHARP_EDGE_DEG    = 40.0
ANCHOR_D_COLOR             = (0, 220, 255)   # 黄（アクティブ時）

INACTIVE_ANCHOR_COLOR      = (90, 90, 90)    # 非アクティブ側は暗いグレーで薄く表示

# クリック位置合わせ（'c' キーの solvePnP）に使うアンカー。
# 4グループはすべて同じ座標系を共有するので1グループだけで十分。
# 点が多すぎて煩わしい場合はここを変えて点数を調整する。
# "A"=大臼歯(16点前後), "B"=小臼歯+犬歯(8点前後), "C"=犬歯+側切歯, "D"=右側4本
LANDMARK_ANCHOR_KEY = "B"

# models/<name>_smooth.obj を使うか。
#   True : 歯肉境界（CT ボクセルの階段状の縁）を平滑化し、穴を塞いで閉じたメッシュにした版。
#          カメラに映る歯頸線の輪郭が実物に近づくので、M3T の輪郭照合が安定する想定。
#   False: 元の group_*.obj（before/after の録画比較用に残してある）。
# 座標系・スケールは元と同一なので initial_poses/ のスロットはそのまま使える。
USE_SMOOTH_MODELS = True

def _model_path(name):
    if USE_SMOOTH_MODELS:
        stem, ext = os.path.splitext(name)
        name = f"{stem}_smooth{ext}"
    return os.path.join(SCRIPT_DIR, "models", name)

anchor_a_path = _model_path(ANCHOR_A_NAME)
anchor_b_path = _model_path(ANCHOR_B_NAME)
anchor_c_path = _model_path(ANCHOR_C_NAME)
anchor_d_path = _model_path(ANCHOR_D_NAME)

# ── 単体ボディモード ─────────────────────────────────────────────────────────
# pym3t ラッパーはモデルごとに独立した Link/Optimizer を作るため、4 グループを別モデルにすると
# 4 つの 6DoF 最適化が独立に走り、Python 側で「良い方の pose に全員を同期」するしかなかった
# （3 本歯 1 グループの輪郭だけで全体の姿勢を決めることになり、奥行きが決まらない）。
# 4 グループを結合した 1 つの OBJ（models/arch_4groups_smooth.obj）を単一ボディとして渡すと、
# M3T 内部で歯列全体の輪郭を使った 1 つの最適化になる。アンカー機構はキー "A" のみで動く。
# 2026-09-09 のマーカー基準計測（開始 2〜4 秒で dz −100〜−180 mm）の対策として追加。
SINGLE_BODY_MODE = True
ARCH_MODEL_NAME  = "arch_4groups_smooth.obj"
ARCH_SHARP_EDGE_DEG = 35.0
ARCH_QUALITY_MAX_EDGES = 1000      # 品質判定用ワイヤーフレームの本数（1500 で fps 13。速度優先で少し減らす）

# ── 階層アンカー（Anchor Switching デモ用、2026-09-29）────────────────────────
# 歯列全体（キー "F"）を主アンカーとして追跡しつつ、4 歯群（A〜D）も副アンカーとして同時に M3T で
# 追跡する。通常は F の姿勢を出力し、F の適合度が落ちた（＝手や器具で歯列の大部分が隠れた）ときだけ、
# まだ良く見えている歯群の姿勢に切り替える。全アンカーは毎フレーム出力姿勢に同期されるので、
# 副アンカーは「遮蔽の間だけ、直前の良好姿勢から短時間引き継ぐ」役割になる（2026-09-09 に見た
# 「小さいアンカー単独では奥行きが決まらず発散する」問題を避ける）。
HIERARCHICAL_ANCHORS      = True
# 比較実験用: Anchor Switching を無効にして歯列全体 F だけで姿勢を出すモード（歯群は追跡・表示するが姿勢源にしない）。
# 起動時は環境変数 M3T_ANCHOR_SWITCHING=0 で OFF、実行中は Unity コマンド anchor_switching:on|off で切替。
# CSV の末尾列 anchor_switching（1/0）に記録され、analyze_runs.py が run のラベルに使う。
ANCHOR_SWITCHING_DEFAULT  = os.environ.get("M3T_ANCHOR_SWITCHING", "1") != "0"
# 統合モード（2026-10-03 歯科ミーティングの提案）: 「スコア最大の 1 歯群を選ぶ」のではなく、「遮蔽されてスコアが
# 落ちた歯群を外し、見えている歯群すべてを使って歯列全体の姿勢を推定する」。各歯群の歯の重心位置（位置は信頼
# できる、回転は弱い）を対応点として重み付き剛体フィット（Kabsch）を解く。基準姿勢（F を離れた時点）から予測した
# 各重心を弱い事前情報として加え、歯群が 1 つしか見えなくても回転が決まるようにする。
# 起動時は M3T_ANCHOR_FUSION=0 で無効。実行中は Unity コマンド anchor_switching:fusion|on|off。
ANCHOR_FUSION_DEFAULT     = os.environ.get("M3T_ANCHOR_FUSION", "1") != "0"
# 2026-10-05 00:13 の run で統合の構成が毎フレーム入れ替わり（0.99〜5.4 回/s）、C が頻繁に混ざり、
# 「使える歯群なし → F 保持」と「統合解除 → F」を往復していた。対策:
#   * 歯群の採用にヒステリシス（ENTER 0.55 を 3 フレーム連続で採用、EXIT 0.42 を割るまで維持）
#   * C を統合から外す（姿勢源としても除外済み。重心が B/D に近く情報が少ない割にスコアが不安定）
#   * 並進は見えている歯群だけで決め、事前情報は回転にだけ効かせる（以前は並進も基準姿勢へ 4 割引き戻していた）
#   * 使える歯群がない間は F の生の姿勢ではなく直前の出力姿勢を保持する
#   * 統合姿勢の並進を 1 次 IIR で平滑化（構成が変わった瞬間の段差を抑える）
# 2026-10-05 00:34 の run（上の対策後）でも構成の入れ替わりと 100° 級の回転跳びが残った。原因:
#   * 歯群重心はどれもモデル原点から約 30 mm 離れている（A 29.8 / B 29.4 / D 30.1 mm）ので、歯群ボディの回転誤差
#     （20° 程度は普通）がそのまま 10 mm 級の重心位置誤差になる。A-B-D は細長い三角形（B は A-D 線から 11.6 mm）
#     なので、この誤差を含む 3 点に Kabsch をかけると回転が大きく外れる（跳び量ゲートが 98〜118° を拒否していた）。
#     → 統合は並進のみ。回転は F を離れた時点（最後の良好姿勢）のまま固定する（best-one の GROUP_ROTATION_FROM_F と同じ）。
#   * F を離れる瞬間、出力は遮蔽で流れた F 姿勢、統合は last_good 基準なので 34° の差を跳び量ゲートが拒否し続け、
#     12 フレームで強制再ロックに落ちていた。→ 離脱フレームはゲートを通さず last_good へ戻す（HOLD と同じ扱い）。
#   * 採用判定の 20 mm 逸脱チェックが単一閾値だったので、閾値付近で歯群が出入りした。→ 15/25 mm のヒステリシス、
#     基準は直前の出力姿勢（ボディは毎フレーム出力姿勢へ戻されるので、1 フレームで 15 mm 動いた歯群は誤適合）。
#   * 統合中は F ボディが統合姿勢に置かれるので F のスコアが 0.6〜0.7 に上がり、1.5 s ごとに F へ戻っては遮蔽で
#     落ちて再離脱していた。→ F への復帰条件を「全歯群が見えている（遮蔽が終わった）」か「F が 2 s 以上良好」か
#     「使える歯群が 1 s 以上ない」に変更。
FUSION_KEYS               = ("A", "B", "D")
FUSION_GROUP_ENTER_QUALITY = 0.55  # この値以上が FUSION_ENTER_FRAMES フレーム続いたら統合に加える
FUSION_GROUP_EXIT_QUALITY  = 0.42  # 統合中の歯群がこれを割ったら外す（0.55/0.42 のヒステリシス）
FUSION_ENTER_FRAMES       = 3
FUSION_GROUP_WEIGHT       = {"A": 1.0, "B": 1.0, "D": 1.0}
FUSION_GROUP_DEV_ENTER_M  = 0.015  # 直前出力から予測した重心との差がこれ以下なら採用候補
FUSION_GROUP_DEV_EXIT_M   = 0.025  # 採用中の歯群がこれを超えてずれたら外す（誤適合・跳び）
FUSION_TRANS_SMOOTH       = 0.5    # 統合姿勢の並進の平滑化係数（1.0 = 平滑化なし）
# 2026-10-05 00:46 run の解析: 歯群が姿勢を出している間の歯列中心のずれは、カメラ光線方向（奥行き）が中央値 7〜9 mm /
# p90 12〜15 mm なのに対し、光線に直交する方向（画像上の位置）は中央値 1〜2 mm。小さな歯群 1 つの輪郭からは
# 奥行き（＝見かけの大きさ）がほぼ決まらないということ。→ 歯群の重心は「観測した光線上で、基準姿勢と同じ奥行き」
# に置く（画像上の位置は歯群に従い、奥行きは F を離れた時点のまま）。遮蔽中に模型が奥行き方向へ動くと追えないが、
# 7〜15 mm のノイズを毎フレーム出すよりよい。best-one（GROUP_ROTATION_FROM_F）と統合モードの両方に効く。
GROUP_DEPTH_FROM_REF      = True
FUSION_F_RETURN_SEC       = float(os.environ.get("M3T_F_RETURN_SEC", "2.0"))   # F のスコアがこれだけ連続して回復閾値以上なら歯群の見え方によらず F へ復帰
FUSION_NO_MEMBER_SEC      = 1.0    # 使える歯群がこれだけ無く F が OK 閾値以上なら F へ戻す
ANCHOR_F_OK_QUALITY       = float(os.environ.get("M3T_F_OK_Q", "0.45"))   # F の単体適合度がこれ以上なら F を出力（切替しない）
# ↑ 案 E（2026-10-14 13:00 run）: 統合中も追跡している F ボディ自身の誤差は qF 0.35〜0.45 で中央値 1.3〜1.4 mm / p90 ≤3.4 mm
#   （ただし毎フレーム統合姿勢から 1 step した値なので、F 単独で何秒も粘れる証拠ではない）。M3T_F_OK_Q=0.38 で試して比較する。
ANCHOR_F_RECOVER_QUALITY  = float(os.environ.get("M3T_F_RECOVER_Q", "0.55"))   # 歯群へ切替後、F がこれ以上に戻ったら F へ復帰（0.45 で復帰すると往復した）
ANCHOR_GROUP_MIN_QUALITY = 0.60   # 0.55→0.60 (2026-10-01)。副アンカーへ切り替える最低適合度
ANCHOR_SWITCH_MAX_ROT_DEG = 15.0   # 20→15 (2026-10-01: 歯群 D が切替後 0.5 s で 63 mm ずれた)
ANCHOR_SWITCH_MAX_TRANS_M = 0.015   # 20 mm → 15 mm (2026-10-01)
ANCHOR_MIN_DWELL_SEC      = 1.5    # 切替後、最低これだけは同じアンカーに留まる（0.5 s では 5 分で 53 回切り替わった）
# 姿勢源にしてよい歯群。C（犬歯+側切歯の 2 本）は丸い形状で回転が決まらず、q 0.65 でも 44° 外れて
# 毎回ロストの引き金になった（2026-09-29 17:50 録画: ロスト 3 回すべて C 起因）。強調表示には残す。
ANCHOR_POSE_SOURCE_KEYS   = ("A", "B", "D")
# 歯群で姿勢を出している間の逸脱監視: 切替時点の姿勢からこれ以上離れたら F（保持姿勢）へ戻す。
# 歯群ボディは全体に比べて拘束が弱く、遮蔽中にじわじわ回って最後に吹っ飛ぶ。
ANCHOR_GROUP_MAX_DRIFT_DEG = 15.0   # 30→15 (2026-10-01)
ANCHOR_GROUP_MAX_DRIFT_M = 0.020   # 40→20 mm (2026-10-01)
# 逸脱で F に戻した直後に同じ歯群へ再切替すると、基準が更新されて 12 mm ずつ流れ続ける（12:44 のログで
# F→A→F→A… を繰り返して 400 mm まで流れた）。戻した後はこの秒数だけ歯群への切替を禁止する。
ANCHOR_DRIFT_COOLDOWN_SEC = 1.5
# 歯群ボディの回転推定は弱い（2026-10-01 14:27 run: A 23°, B 32° の回転誤差、位置は 10〜20 mm）。
# 回転 20° の誤りは歯列の反対側を 17 mm 動かす。歯群で肩代わりする間は「歯群の歯の位置」だけを採用し、
# 回転は F を離れた時点のものを保つ（遮蔽中に模型を回す場面は捨てる）。逸脱監視も歯群中心の並進だけで行う。
GROUP_ROTATION_FROM_F     = True
# ── ① F 復帰時の回転再探索（2026-10-09）──────────────────────────────────────
# 統合中は回転を F 離脱時のまま固定するため、遮蔽中に模型が回ると F 復帰後も古い回転の局所解に落ちる
# （ArUco 実測: 復帰後 11〜16° が残り、q は 0.56〜0.64 と「良好」に見えたまま）。復帰の瞬間だけ、歯列重心
# まわりに回転した候補を M3T で数ステップ精密化し、剛体整合スコアが無回転より十分良いときだけ採用する。
# 2026-10-14 12:45 run で既定を OFF にした: 採用判定が q 頼みのため、45.3 s の復帰で +21°（q 0.67→0.80）を採用して
# 実際には 5.5°→31.8° に悪化、F はその局所解に q 0.91〜0.93 のまま 15 s 留まった（ArUco 32 mm / 44°）。108 s でも
# 0.7°→13.4° に悪化。改善した例（90 s: 10.8°→0.8°）もあるが、safety-critical では「q が高い静かな誤り」を作る方が重い。
# 遮蔽中の回転は統合中の回転追従（FUSION_ROT_*）で扱う。検証したいときは M3T_RETURN_SEARCH=1。
RETURN_ROT_SEARCH_ENABLED = os.environ.get("M3T_RETURN_SEARCH", "0") != "0"
RETURN_ROT_SEARCH_DEGS    = {"y": (6.0, 12.0, 18.0), "x": (8.0,), "z": (8.0,)}   # カメラ座標の軸ごとの候補角（±）。y=鉛直まわり（机上で回す）が主
RETURN_ROT_SEARCH_STEPS   = 2      # 候補ごとの M3T 反復。3→2 (2026-10-09 14:42 run: 11 候補 × 3 step = 0.48 s が重い)
RETURN_ROT_ACCEPT_MARGIN  = 0.03   # 無回転候補の剛体スコアよりこれ以上良いときだけ採用（q は 15° ずれても 0.6 出るので慎重に）
# 再探索は 1 回 0.48 s 止まる。14:34 run では F の離脱→復帰が 150 s に 22 回（5〜7 s に 1 回）あり、全部で走らせると
# 約 7% の時間が停止になった。1〜2 s の短い離脱では回転はほぼ動かないので、長い遮蔽か大きな移動のときだけ走らせる。
RETURN_ROT_SEARCH_MIN_OCCL_SEC = 1.5    # 離脱がこれ以上続いたときだけ再探索（1.0 では 14:42 run の 19 回全部が通り、1.0〜1.4 s の 7 回は補正 0°）
RETURN_ROT_SEARCH_MIN_MOVE_M   = 0.006  # または離脱中に歯列重心がこれ以上動いたとき（回転で重心も動くため）
# 2026-10-14 11:41 run: 再探索 12 回すべてが 1 フレーム 440〜470 ms の停止になっていた（「たまに起こるラグ」の正体）。
# 候補を 1 フレームにまとめて評価せず、復帰時の画像を控えておいて毎フレーム数候補ずつ評価する（ライブ追跡は
# その間も続く）。全候補が終わったら、採用候補の「基準姿勢からの回転（モデル座標）」を今の姿勢に掛けて適用する。
RETURN_ROT_SEARCH_PER_FRAME    = int(os.environ.get("M3T_RETURN_SEARCH_PER_FRAME", "2"))   # 1 フレームに評価する候補数（1 候補 ≈ 36 ms）
# ── 統合中の回転追従（2026-10-14）────────────────────────────────────────────
# 統合モードは回転を F 離脱時のまま固定していたので、遮蔽中に模型を回すと追従しなかった。⑤ の実測で A（臼歯 3 本）の
# 回転は遮蔽なしで F 並み（1.9° vs 2.5°）、統合中も 3.7° だったので、A（なければ B）の回転へ固定回転を少しずつ寄せる。
# 歯群の回転は時々 20° 以上外れるため、(1) 採用は q が閾値以上かつ回転が安定（前フレームとの差が小さい）な状態が
# 連続したときだけ、(2) 現在の回転との差が大き過ぎる歯群は誤適合として無視、(3) 1 フレームの変化量と離脱時からの
# 累計に上限、(4) 歯群が統合から外れたら停止（回転はその時点で保持）。CSV の fus_rot_src / fus_rot_deg で確認できる。
FUSION_ROT_FROM_GROUP     = os.environ.get("M3T_FUSION_ROT", "1") != "0"
FUSION_ROT_SOURCE_KEYS    = ("A",)                  # 13:12 run で B を外した（偏り 11〜16°、B(cap) で走った）。D も使わない
FUSION_ROT_PAIR_CHECK     = True    # 2 歯群以上のときだけ追従し、重心の並びの向き（画像上）で回転を検算する（13:12 run の暴走対策）
FUSION_ROT_PAIR_MAX_DEG   = 8.0     # 予測と観測の向きの差がこれを超えたら追従を取り消して離脱時の回転へ戻す
FUSION_ROT_MIN_QUALITY    = {"A": 0.60, "B": 0.65}  # 回転源として使う最低適合度（統合メンバーの条件 0.55 より厳しく）
FUSION_ROT_EXIT_QUALITY   = {"A": 0.52, "B": 0.58}  # 追従中はこの値を割るまで続ける（開始 0.60/0.65 とのヒステリシス）
FUSION_ROT_MIN_FRAMES     = 3       # 上の条件＋安定がこのフレーム数続いたら追従開始
FUSION_ROT_MAX_JITTER_DEG = 4.0     # 歯群ボディの回転が前フレームからこれ以上変わったら「不安定」として連続カウントを戻す
FUSION_ROT_MAX_DIFF_DEG   = 25.0    # 歯群の回転が現在の出力回転からこれ以上離れていたら誤適合とみなして無視
FUSION_ROT_BLEND          = 0.25    # 1 フレームで残差のこの割合だけ寄せる（≈4 フレームで 1/e）
FUSION_ROT_MAX_STEP_DEG   = 3.0     # 1 フレームの回転変化量の上限（2→3: 12:09 run で 1.6 s に 20° 回したとき追従が 10° 遅れた）
# 離脱時の回転からの累計上限（25→35: 12:09 run で実際に 30° 回して上限に当たった）。2026-10-21 の大きく回した録画では
# 真値 42° に対し PnP は 40° と正しく出していたのに 35° で止まり 6〜7° ずれた → M3T_FUSION_ROT_MAX_TOTAL_DEG で変更
FUSION_ROT_MAX_TOTAL_DEG  = float(os.environ.get("M3T_FUSION_ROT_MAX_TOTAL_DEG", "35.0"))
# 歯群ボディの回転は F と系統的にずれる（2026-10-14 11:53 run: F が q 0.77 で安定していても A は出力から中央値 5.3°、
# B 8.6° 離れた向きに収束する。メッシュと実物の差や 3 本歯の拘束の弱さによる偏り）。そのまま追従すると遮蔽のたびに
# その偏りの分だけ回ってしまうので、F 追跡中に「歯群の回転 → 出力回転」の相対回転（モデル座標）を EMA で覚えておき、
# 統合中はそれで補正した歯群の回転を追従先にする。
FUSION_ROT_BIAS_COMP      = os.environ.get("M3T_FUSION_ROT_BIAS", "1") != "0"
FUSION_ROT_BIAS_ALPHA     = 0.05    # 偏りの EMA 係数（≈20 フレーム ≈ 1 s の時定数）
FUSION_ROT_BIAS_MIN_QF    = 0.60    # 偏りを学習するのは F がこの適合度以上で出力しているときだけ
# 静止時の追従の積み上がり対策（2026-10-19 OMEN 15:16 run、静止模型で遮蔽→復帰 13 回）: A+B 統合中の誤差が追従の累計に比例
# （≤1°: 2.1 mm / 1.3°、>10°: 7.9 mm / 14.3°）。A の回転の小さな偏りへ毎フレーム寄せ、全アンカーをその姿勢へ同期するので
# 模型が動かなくても積み上がる。A の投影範囲の特徴点を「基準フレームから」LK で追い（ノイズは積み上がらないが動きは積み上がる）、
# 中央値の移動が FUSION_ROT_MOTION_MIN_PX を超えたら「動いた」として FUSION_ROT_MOTION_HOLD_SEC だけ追従を許し、基準を取り直す。
# 保存フレーム（20 フレーム間隔、縮小画像）の実測: 静止 0.01〜0.03 px、手で隠している最中 0.02〜0.1 px、模型の移動 0.3〜15 px。
# CSV fus_motion_px、fus_rot_src の "(still)" で確認。M3T_FUSION_ROT_GATE=0 で従来（常に追従）。
FUSION_ROT_MOTION_GATE     = os.environ.get("M3T_FUSION_ROT_GATE", "1") != "0"
FUSION_ROT_MOTION_MIN_PX   = float(os.environ.get("M3T_FUSION_ROT_MOTION_PX", "0.25"))   # 縮小画像（QUALITY_SCALE）上の px
FUSION_ROT_MOTION_HOLD_SEC = 0.5
FUSION_ROT_MOTION_MIN_POINTS = 8
# ── 回転源: キーフレーム特徴点 PnP（2026-10-19、文献調査に基づく）────────────────────────────────
# 15:24 run で遮蔽中に模型を回すと A 自身の回転が 9〜13° 外れ、A への追従は 9.5〜30.5° 回った（真値 7.6〜14.7°）。
# A を出力へ毎フレーム同期してから次の 1 step を「観測」として使う構造そのものが偏りを積分する（予測の暴走と同じフィードバック）。
# 代わりに、遮蔽直前の F 良好フレームをキーフレームにし、F の投影範囲内の特徴点を z バッファで最前面のメッシュ頂点に結び付け、
# 毎フレーム LK（前後追跡チェック、Kalal+ ICPR 2010）で追って、回転＋並進を重み付き Gauss-Newton で解く
# （キーフレーム基準なので偏りが積分されない: Vacchetti, Lepetit, Fua TPAMI 2004）。並進には統合（歯群重心）の値を
# 事前分布として入れる（2 mm の重心誤差が 4° の回転に化ける結合があるため）。静止判定（ZUPT, Skog+ TBME 2010）として、
# キーフレームからの回転が max(FUSION_PNP_MIN_DEG, 3σ) 未満なら回転は離脱時のまま保持する。
# オフライン検証（15:24 run の保存フレーム、20 フレーム間隔でつないだ追跡）: 静止 0.1〜1.4°、回転 6〜13° の区間で
# 5.7〜10.7°（A 追従 14〜22°、保持 6.8〜13.5°）、4〜15° の区間で 3.2〜9.5°（A 追従 4〜12°、保持 5〜15°）。
# M3T_FUSION_ROT_SOURCE=a で従来の A 追従（動きゲート付き）に戻す。CSV fus_pnp_n / fus_pnp_std_deg / fus_pnp_deg。
FUSION_ROT_SOURCE        = os.environ.get("M3T_FUSION_ROT_SOURCE", "pnp").lower()
FUSION_PNP_MAX_FEATURES  = 300
FUSION_PNP_MIN_POINTS    = 12      # 生きている点がこれ未満なら回転は保持
# ⑨ 特徴点 PnP の画像の倍率（2026-10-20）。以前は品質評価と同じ半分の大きさ（0.5）で検出・追跡していた。元の解像度（1.0）にすると
#   歯の上の点が増え、LK の位置精度も上がる想定。以下の px のしきい値は「半分の大きさの画像での px」として書いてあり、
#   使う場所で FUSION_PNP_PX（= 倍率 / 0.5）を掛けて換算する。
#   実機（20:18・20:28 run）では 1.0 の方が悪かった（20:01 の 0.5 は統合中 43% が 2 mm・2° 未満、1.0 は 0〜13%）ので既定は 0.5 に戻した。
#   M3T_FUSION_PNP_SCALE=1.0 で元の解像度。
FUSION_PNP_SCALE         = float(os.environ.get("M3T_FUSION_PNP_SCALE", "0.5"))
FUSION_PNP_PX            = FUSION_PNP_SCALE / 0.5
FUSION_PNP_MAX_POINTS    = int(round(FUSION_PNP_MAX_FEATURES * FUSION_PNP_PX))   # 元の解像度では 600 点まで


def _pnp_px(v):
    """半分の大きさの画像で決めた px を、特徴点 PnP の画像の px に換算する。"""
    return v * FUSION_PNP_PX


def _pnp_odd(v):
    """カーネル・窓の大きさ（奇数）を倍率に合わせる。"""
    n = int(round(v * FUSION_PNP_PX))
    return n if n % 2 == 1 else n + 1
FUSION_PNP_FB_MAX_PX     = 0.8     # 前後追跡の往復誤差（縮小画像 px）。0.5 は奥行き方向の移動（拡大縮小）で点を捨て過ぎた
# ⑤ 点の補充（11:50 run）: 遮蔽中に奥行き方向へ動かすと点が 1.5 s で 43→11 と減り PnP(bad) → 歯群重心（奥行き固定）に
#   戻って追えなかった（142.5 s: 真値 −13.6 mm に対し出力 +9.2 mm）。生存点がこれ未満になったら、解が信用できる
#   （残差中央値 ≤ FUSION_PNP_REFRESH_MAX_RES_PX）ときに今の姿勢で歯色の画素から点を取り直して足す。
FUSION_PNP_REFRESH_BELOW     = 40
FUSION_PNP_REFRESH_MAX_RES_PX = 1.0
# ⑧ 見えている歯の輪郭を特徴点と一緒に使う（2026-10-20 20:01 run、Choi & Christensen IJRR 2012 のキーポイント＋エッジの考え方）:
#   遮蔽中に 25〜35° 回すと、歯の上の特徴点が 35〜39 まで減り、補充のたびに今のずれた姿勢で点を結び付けるので回転の取りこぼし
#   （9〜11°）が固まった。統合に使っている歯群（見えている歯）の稜線を投影し、向き付きエッジ距離マップ（品質評価と同じ）で
#   画像のエッジまでの距離を残差に加えて、特徴点と一緒に姿勢を解く。距離は FUSION_PNP_EDGE_MAX_PX で打ち切る（手や別の
#   エッジに引かれないように）。輪郭の重みは「輪郭の合計 ≈ 特徴点の合計 × FUSION_PNP_EDGE_REL」になるよう自動で決める。
#   補充は、その姿勢で見えている歯群の輪郭の適合度が FUSION_PNP_REFRESH_MIN_EDGE_Q 以上のときだけ。歯肉・台座など歯以外は使わない。
#   **結果（20:01 run の保存フレームで再生）: 悪化したので既定 OFF**（M3T_FUSION_PNP_EDGE=1 で有効）。回転区間の PnP 回転誤差
#   中央値 0.7→1.3° / 6.7→23.3° / 4.8→15.6°。正しい姿勢でも稜線の標本点からエッジまで中央値 2.4〜3.0 px（1 px 以内 14〜19%）で、
#   打ち切り 4 / 2 / 1 px のどれでも、4 フレームとも「カメラ x 軸まわり +6°」の方がコストが小さい（系統的な偏り。切り出した
#   歯群メッシュの切り口の稜線や、歯頸部・台座の密なエッジに引かれる）。歯の稜線は 2 mm・2° の精度の手がかりにならない。
FUSION_PNP_USE_EDGES          = os.environ.get("M3T_FUSION_PNP_EDGE", "0") == "1"
FUSION_PNP_EDGE_MAX_PER_GROUP = 200     # 歯群ごとに使う稜線の数（1 本あたり 3 点を標本化）
FUSION_PNP_EDGE_MAX_PX        = 4.0     # 縮小画像 px（品質評価の QUALITY_EDGE_MAX_DIST_PX と同じ）
FUSION_PNP_EDGE_REL           = float(os.environ.get("M3T_FUSION_PNP_EDGE_REL", "1.0"))
FUSION_PNP_REFRESH_MIN_EDGE_Q = 0.55
FUSION_PNP_OUTLIER_PX    = 2.0     # 剛体運動に合わない点（手・反射）はこの残差で永久に捨てる（1.5 は回転中に点が減り過ぎた）
# 15:48 run（実機、静止模型）: 手が入る瞬間に生存点が 125→78 / 65→41 と減り、残った点の一部が手に引きずられて
# PnP が σ 0.2〜0.4° のまま 6〜16° 回った（前後追跡チェックは滑らかな手の動きを捕まえられない）。
# ① 静止判定: 生存点のキーフレームからの移動量の中央値がこれ未満なら模型は動いていない → 回転保持＋解をキーフレーム姿勢へ戻す
#   （手に引きずられる点は少数なので中央値はほぼ動かない。3° の回転で歯の点は約 5 px 動く）
# ② 前回の解とキーフレーム姿勢の両方から解き、1 px 以内の点が多い方を採る（手と歯の折衷解に居座らないように）
FUSION_PNP_STILL_PX      = float(os.environ.get("M3T_FUSION_PNP_STILL_PX", "0.6"))   # 縮小画像 px
# ⑩ 手に引きずられた点の除去（2026-10-20 20:18 run、元の解像度で点が 3 倍になった後）: 静止判定のあいだ、手に覆われた点の一部は
#   指の表面を追って動き続けた（模型の点が多数なので中央値は動かず静止のまま）。模型が少し動いた瞬間に解くと手の点に合う解
#   （真値 0.9° に対し 17〜24°）に落ち、外れ値除去で模型の点（多数派）を捨てて 212→89 点、σ 0.2° のまま誤った回転を出し続けた。
#   (a) 静止判定のあいだにキーフレームから FUSION_PNP_STILL_DROP_PX 以上動いた点は模型の上の点ではないので捨てる。
#   (b) 解に合う点が解く前の点の FUSION_PNP_MIN_INLIER_FRAC 未満なら、その解は信用せず点も捨てない（少数派の解を採らない）。
#       20:18 の誤った解は 212 点中 104 点（49%）。正しく追えている区間の再生では 97〜100%。
#   20:28 run（元の解像度＋⑩）で誤った急な回転は消えたが、全体は 20:01 より悪かったので、半分の解像度に戻すのに合わせて既定は切る
#   （0 で無効）。元の解像度で使うときは M3T_FUSION_PNP_STILL_DROP_PX=1.5 M3T_FUSION_PNP_MIN_INLIER_FRAC=0.6。
FUSION_PNP_STILL_DROP_PX   = float(os.environ.get("M3T_FUSION_PNP_STILL_DROP_PX", "0"))      # 縮小画像 px、0 で無効
FUSION_PNP_MIN_INLIER_FRAC = float(os.environ.get("M3T_FUSION_PNP_MIN_INLIER_FRAC", "0"))    # 0 で無効
# ③ 見た目の確認（KLT の NCC 検証）: 15:53 run 45.6 s で手に覆われた点が手と一緒に動き、静止判定もすり抜けて
#   PnP が σ 0.3〜0.5° のまま 11〜12° 回った（真値 1°）。キーフレームでのその点まわりの小画像と今の小画像の
#   正規化相互相関がこれ未満の点（歯→手に変わった点）は以後使わない。
FUSION_PNP_NCC_MIN       = float(os.environ.get("M3T_FUSION_PNP_NCC", "0.6"))
FUSION_PNP_NCC_HALF      = 5       # 小画像の半径 px（11×11、縮小画像）
# ④ 統合中の並進も PnP から: 11:43 run で模型を隠したまま横にずらした 2 区間、歯群重心（奥行き固定）の出力は
#   最大 21.6 / 22.8 mm ずれたが、同じ保存フレームで PnP の F 重心は 0.7〜1.5 mm だった（回転区間でも 1.8〜2.7 mm）。
#   PnP の解が信用できるときは、出力の F 重心を PnP の F 重心に置き換える（回転は回転追従の結果）。M3T_FUSION_PNP_TRANS=0 で従来。
FUSION_PNP_USE_TRANSLATION = os.environ.get("M3T_FUSION_PNP_TRANS", "1") != "0"
# 統合姿勢の逸脱監視（ANCHOR_GROUP_MAX_DRIFT_M = 20 mm）は歯群ボディの流れ対策。並進を PnP で出しているときは、
# 実際に 20 mm 前後ずらした模型を追えるよう（11:43 run）この値まで広げる。
FUSION_PNP_MAX_DRIFT_M = 0.060
# ⑥ F 追跡中の見張り（2026-10-20 18:53 run）: 模型をカメラへ近づけながら少し上へ動かすと（約 25 mm/s）、F が逆向き（遠ざかる側）
#   へ 0.5 s で 30〜46 mm 跳び、約 1 s 後に戻った（qF 0.7→0.5 で F_OK 0.45 を割らないので統合にも入らない）。遮蔽中に使う
#   キーフレーム特徴点 PnP を F 追跡中も並走させ、F が PnP から F_GUARD_DIVERGE_MM / _DEG 以上離れた状態が
#   F_GUARD_DIVERGE_FRAMES 続いたら全アンカーを PnP の姿勢へ戻す。キーフレームは F 良好・PnP と一致しているときに
#   F_GUARD_REKEY_SEC ごとに取り直す（補充した点の誤差を溜めない）。静止中に F が滑る場合（11:50 run 155〜165 s）も同じ仕組みで戻る。
#   M3T_F_GUARD=0 で無効。ログ（event 列）に f_guard_reset。
F_GUARD_ENABLED        = os.environ.get("M3T_F_GUARD", "1") != "0"
F_GUARD_MIN_QF         = 0.60    # キーフレームを作る／取り直すときの F の最低適合度
F_GUARD_MIN_F_SEC      = 0.3     # F で連続してこの秒数以上追跡できてからキーフレームを作る
F_GUARD_REKEY_SEC      = 2.0
F_GUARD_AGREE_MM       = 2.0     # 取り直してよい一致度
F_GUARD_AGREE_DEG      = 2.0
F_GUARD_DIVERGE_MM     = float(os.environ.get("M3T_F_GUARD_MM", "8.0"))
F_GUARD_DIVERGE_DEG    = float(os.environ.get("M3T_F_GUARD_DEG", "8.0"))
F_GUARD_DIVERGE_FRAMES = 2
F_GUARD_MIN_POINTS     = 25
# ⑯ 統合から F に戻った直後は、F が統合のずれ（5〜8°）を引き継いでいる。0.3 s でそこから見張りのキーフレームを作ると、
#   その後 F が正しい姿勢へ収束しようとするたびに見張りがずれた姿勢へ戻した（2026-10-21 occ_rot_01 の再生: 0.2 s ごとに
#   5° の「戻し」が続き 8.6° のまま）。戻った後は F_GUARD_AFTER_RETURN_SEC 待ってから作る。M3T_F_GUARD_AFTER_RETURN で変更。
F_GUARD_AFTER_RETURN_SEC = float(os.environ.get("M3T_F_GUARD_AFTER_RETURN", "0.3"))
# ⑮ 統合に入るときに直近の F 良好フレームが無ければ、F の見張りが追っている点の集合を遮蔽中の PnP に引き継ぐ。
#   M3T_FUSION_PNP_FROM_GUARD=1 で有効。
FUSION_PNP_FROM_GUARD = os.environ.get("M3T_FUSION_PNP_FROM_GUARD", "0") == "1"
# ⑰ 統合に入るときの姿勢（離脱時の姿勢）を、手が入りかけてずれ始めた F（last_good）ではなく、そのフレームで F の見張りの
#   PnP が解いた姿勢にする（last_good から 10 mm / 10° 以内のとき）。回転源 PnP も見張りの点をそのまま使う。
#   2026-10-21 occ_trans_02: 直前まで 0.3° だった F が統合に入るフレームで 2.7° ずれ、その回転が遮蔽中ずっと保持された。
#   M3T_FUSION_DEPART_FROM_GUARD=1 で姿勢ごと（回転源 PnP も見張りの点）、=2 で回転だけ見張りから（F 重心の位置は last_good、
#   回転源 PnP は従来どおり直近の良好フレームから。見張りのキーフレームは並進が 2〜3 mm 偏ることがあった: occ_static_01 で 86→40%）。
FUSION_DEPART_FROM_GUARD = int(os.environ.get("M3T_FUSION_DEPART_FROM_GUARD", "0") or 0)
# ⑱ 切り分け用（2026-10-21）: 統合中の姿勢に部分モデル（歯群）を使わず、特徴点 PnP だけで出す。歯群は切替・復帰の判断と
#   表示にだけ使う（PnP の重心の事前分布・位置の平滑化・逸脱監視・PnP が使えないときの歯群からの姿勢を外す）。
#   M3T_FUSION_PNP_ONLY=1 で有効。
FUSION_PNP_ONLY = os.environ.get("M3T_FUSION_PNP_ONLY", "0") == "1"
# ⑲ Anchor Switching の本体（2026-10-21）: 遮蔽中の特徴点 PnP を「見えているアンカー（歯群）」の上の点だけで解く。
#   各点がどの歯群の上にあるかをキーフレームを作るときに決めておき、適合度 s が FUSION_PNP_ANCHOR_Q 未満の歯群（隠れていると
#   判定したアンカー）の点はそのフレームの解から外す。見えている歯群の点が FUSION_PNP_ANCHOR_MIN_POINTS 未満なら全部の点で解く。
#   使ったアンカーは Unity のパネル・歯群の色分けに出す。M3T_FUSION_PNP_ANCHORS=1 で有効、しきい値は M3T_FUSION_PNP_ANCHOR_Q。
FUSION_PNP_ANCHORS = os.environ.get("M3T_FUSION_PNP_ANCHORS", "0") == "1"
FUSION_PNP_ANCHOR_Q = float(os.environ.get("M3T_FUSION_PNP_ANCHOR_Q", "0.55"))
FUSION_PNP_ANCHOR_MIN_POINTS = 30
F_GUARD_MAX_RES_PX     = 1.0
# ⑦ 統合から抜け出せない状態の検知（2026-10-20 19:47 run）: 遮蔽中に回して回転が 17.6° ずれたまま PnP の点が 0 になり、
#   手を離して A/B は q 0.7 で見えているのに、F は出力（ずれた姿勢）へ毎フレーム同期されるので q 0.52〜0.55 から上がらず
#   復帰条件（q ≥ 0.55 が 2 s 連続 or 全歯群が見える）を満たせないまま 53 s 統合に居座った。統合が STUCK_MIN_FUSION_SEC
#   以上続き、PnP が STUCK_PNP_BAD_SEC 以上使えず、A と B が STUCK_GROUP_Q 以上見えているのに F が戻れないときは、
#   全探索で再ロックする（STUCK_COOLDOWN_SEC に 1 回まで）。M3T_FUSION_STUCK_RELOCK=0 で無効。
FUSION_STUCK_RELOCK       = os.environ.get("M3T_FUSION_STUCK_RELOCK", "1") != "0"
FUSION_STUCK_MIN_FUSION_SEC = 5.0
FUSION_STUCK_PNP_BAD_SEC  = 3.0
FUSION_STUCK_GROUP_Q      = 0.60
FUSION_STUCK_COOLDOWN_SEC = 5.0
FUSION_PNP_MAX_RES_PX    = 1.5     # 残差中央値がこれを超えたら解を信用しない
# σ は残差中央値から出すので過大（15:24 run の保存フレームで実誤差 1.3〜2.0° のとき σ 2.0〜2.7°）。3σ / 3° だと
# 4〜8° 回した区間を保持し続け、9° 超の良い解も捨てていたので 2σ / 5° にした。
FUSION_PNP_MAX_STD_DEG   = 5.0     # 回転の 1σ（最大固有値）がこれを超えたら解を信用しない
FUSION_PNP_MIN_DEG       = float(os.environ.get("M3T_FUSION_PNP_MIN_DEG", "3.0"))   # ZUPT: これ未満の回転は保持
FUSION_PNP_ZUPT_SIGMA    = 2.0     # ZUPT: キーフレームからの回転が max(MIN_DEG, この倍数 × σ) 未満なら保持
# 並進の事前分布（統合の F 重心、奥行きは離脱時に固定）の 1σ。2 mm だと模型を回したとき重心の移動を許さず回転を
# 押さえ込んだ（11:35 run 78.5 s の区間を保存フレームで再生: 終了時 9.4°→10 mm で 4.4°。他の区間はほぼ不変）
FUSION_PNP_PRIOR_SIGMA_M = 0.010
FUSION_PNP_BLEND         = 0.5     # 1 フレームで PnP 回転へ寄せる割合
# ⑭ 回転を模型座標の 1 軸まわりに制限（2026-10-21 自宅の録画で評価）: 遮蔽中に動かすと、点の当てはまりは正しい回転でも
#   4〜5° ずれた回転でも 0.35 px 以内しか違わず（回転がほぼ決まらない）、点のわずかな偏りで誤った側に倒れていた
#   （平行移動を回転と取り違え、回転を傾きとして出す）。模型を机に置いて動かす限り回転は机の法線まわりだけで、
#   録画の真値の回転軸は模型の +y から 1.26°（ばらつき 0.65°）。保存した点で解き直すと、遮蔽中に動かす区間の
#   2 mm・2° 未満が 27% → 96%。臨床（頭が自由に動く）では成り立たない仮定なのでデモ・評価用。
#   M3T_FUSION_PNP_AXIS="0,1,0.022"（模型座標の軸）で有効、空で 3 軸（従来）。特徴点を使う F の見張りにも効く。
_pnp_axis_env = os.environ.get("M3T_FUSION_PNP_AXIS", "").strip()
FUSION_PNP_AXIS = None
if _pnp_axis_env:
    FUSION_PNP_AXIS = np.array([float(v) for v in _pnp_axis_env.split(",")], np.float64)
    FUSION_PNP_AXIS /= np.linalg.norm(FUSION_PNP_AXIS)
# 遮蔽直前の F 良好フレームがこれより古ければキーフレームにしない（1.0→3.0: 11:43 run で遮蔽の合間の F が 0.04〜0.8 s と
# 短く、一瞬 F に戻ってから遮蔽が始まると 1 s を超えて 20 区間中 9 区間でキーフレームが作れず回転を保持した）
FUSION_PNP_KEYFRAME_MAX_AGE_SEC = 3.0
FUSION_PNP_KEYFRAME_MIN_QF = 0.50       # 15:24 run の F 追跡中 qF は中央値 0.57 / p10 0.35
# F に戻った直後は姿勢が収束しきっておらず、11:35 run では 9〜15° ずれた F をキーフレームにして次の遮蔽区間に誤差を
# 引き継いだ（90.5 s・31.1 s の区間、復帰 0.26 s 後に次の遮蔽）。F で連続してこの秒数以上追跡できたフレームだけを
# キーフレームにする。1.0 s では 11:43 run の 20 区間中 11 区間しか作れず、0.3 s・3.0 s なら 19 区間（同 run の記録で計算、
# キーフレーム時点の誤差 中央値 1.2°、最大 10.7°。F 自体が外れていた時間のフレームは q では区別できない）。
FUSION_PNP_KEYFRAME_MIN_F_SEC = 0.3
# ⑪ キーフレームの検算（2026-10-21 自宅 10:06 run）: 遮蔽直前の 0.5 s で F が x・z 軸まわりに 9° 滑り（qF 0.58〜0.63 のまま）、
#   ⑥ の見張りが戻した直後に遮蔽が始まった。キーフレームは見張りが戻す前のフレーム（ArUco 8.7°）になり、戻した後のフレームは
#   qF < 0.50 で控えられなかった。遮蔽中は静止のあいだ離脱時の回転（正しい）を保持していたが、動き出して PnP の回転を採った瞬間に
#   キーフレームのずれへ寄り、遮蔽の終わりまで 9〜13° ずれた。対策: (a) 見張りの PnP と F の差が下の値を超えるフレームは
#   キーフレームにしない（F 自体のずれを q の代わりに見張りで判定）、(b) 見張りが F を戻したフレームは戻した姿勢をキーフレームにする。
#   M3T_FUSION_PNP_KEY_GUARD_DEG=0 で (a) を無効。
FUSION_PNP_KEYFRAME_GUARD_DEG = float(os.environ.get("M3T_FUSION_PNP_KEY_GUARD_DEG", "3.0"))
FUSION_PNP_KEYFRAME_GUARD_MM  = float(os.environ.get("M3T_FUSION_PNP_KEY_GUARD_MM", "4.0"))
# ⑫ 遮蔽中は PnP が生きている間、品質低下による自動再ロックを待つ（2026-10-21 自宅 10:22 run）: 手で歯列の半分を 10 s 覆い、
#   使える歯群なし・q 0.33〜0.47 のまま PnP で保持（ArUco 1〜3 mm）していたのに、HOLD が続いたので自動再ロックが走った。
#   手が乗ったままの近傍探索は 2 回失敗（q 0.27 / 0.13）→ 並進探索へ格上げ → 手がどいた後、白い照明で歯色に見える台座の模様に
#   q 0.79 で誤ロック（37 mm / 35°）。PnP（遮蔽前の良好フレーム基準）が解けている間は見失っていないので探索しない。
#   誤ロック検知・統合から抜け出せない状態（force_relock）は従来どおり。FUSION_PNP_DEFER_MAX_SEC を超えたら従来どおり再ロック。
#   M3T_PNP_DEFER_RELOCK=0 で従来。
FUSION_PNP_DEFER_RELOCK  = os.environ.get("M3T_PNP_DEFER_RELOCK", "1") != "0"
FUSION_PNP_DEFER_MAX_SEC = 30.0
# ⑬ 統合中の F の試走（2026-10-21 録画の再生で評価）: 統合中は全アンカーを毎フレーム出力姿勢へ戻していたので、出力が 5〜8° ずれた
#   まま手を離しても F は 1 step しか進めず q が 0.42〜0.50 から上がらず、F への復帰に 1.2〜3.4 s かかった（その間ずれたまま）。
#   統合中は F だけ出力へ戻さずに追わせ、出力から F_FREE_MAX_MM / _DEG 以上離れたら出力へ戻す。M3T_F_FREE=1 で有効。
F_FREE_IN_FUSION = os.environ.get("M3T_F_FREE", "0") == "1"
# 診断（2026-10-21）: M3T_PNP_DUMP=<path.pkl> で、統合中の特徴点 PnP の点（2D 位置・3D 点・結び付けた姿勢）を毎フレーム保存する。
#   experiment/pnp_diag.py が ArUco の真値姿勢で投影し直して、点が壊れているのか解き方の問題かを切り分ける。
PNP_DUMP_PATH = os.environ.get("M3T_PNP_DUMP", "")
_pnp_dump_segments = []
_pnp_dump_frames = []
F_FREE_MAX_MM    = float(os.environ.get("M3T_F_FREE_MM", "15"))
F_FREE_MAX_DEG   = float(os.environ.get("M3T_F_FREE_DEG", "15"))
# マーカー基準が古い疑いの検知（2026-10-14: 10/09 の登録のまま測ったら全フレーム 15 mm / 17° の一定誤差だった。
# ボードか模型が動いた＝物理的な関係が変わったのに、q は 0.6 台で見た目は普通に追えていた）。
# F が良好に追跡しているのに誤差が大きい状態が続いたら、Unity パネルに登録し直しを促す。
MARKER_STALE_ERR_MM       = 8.0
MARKER_STALE_MIN_QF       = 0.60
MARKER_STALE_SEC          = 3.0
# ── ② 回転不確か警告 ────────────────────────────────────────────────────────
# 統合中（回転固定）が長い／歯列が大きく動いた／HOLD 中は、出力の回転が古い可能性があることを Unity に伝える。
ROT_UNCERTAIN_AFTER_SEC   = 2.0    # 統合（回転固定）がこれ以上続いたら警告
ROT_UNCERTAIN_MOVE_M      = 0.010  # 統合中に歯列重心が離脱時からこれ以上動いたら警告（動いたなら回ってもいる可能性）
ROT_UNCERTAIN_AFTER_RETURN_SEC = 2.0   # F 復帰後、この秒数は警告を維持（再探索の結果を人が確認できるように）
# Unity の強調表示: True = F 出力中も最良歯群に色を付ける（旧動作、色がちらつく）。False = 肩代わり中の歯群だけ色を付け、
# F 出力中は全体を同色で表示する。
DISPLAY_BEST_GROUP_WHEN_F = False
# 出力姿勢の 1 フレーム跳び量ゲート（全モード共通）。前フレームの出力からこれ以上跳んだ姿勢は捨て、
# 前フレーム姿勢に全ボディを戻して M3T にやり直させる。12 fps で 30 mm/フレーム = 0.36 m/s。
# 2026-10-01 ログ: 1 フレームで 38〜94 mm、奥行き 62 mm まで突っ込む「吹っ飛び」が 66 回。
POSE_JUMP_MAX_TRANS_M     = 0.030
POSE_JUMP_MAX_ROT_DEG     = 30.0
POSE_DEPTH_MIN_M          = 0.09   # デモ配置は 150 mm 前後。カメラ距離を変えたらここも変える
POSE_DEPTH_MAX_M          = 0.28
# ゲートの固着防止: 跳び先の剛体整合スコアが GOOD 以上で、かつ保持姿勢よりこのマージン以上良ければ
# 跳びを受け入れる。2026-10-01 12:24 のログで、歯群 D の誤姿勢（63 mm ずれ）を出力した直後に F が毎フレーム
# 正しい位置（49 mm 先）へ収束するのをゲートが 50 回以上拒否し続け、再ロックまで 3 秒固まった。
POSE_JUMP_ACCEPT_MARGIN   = 0.10
POSE_JUMP_ACCEPT_FRAMES   = 3       # 同じ跳び先（5 mm/5° 以内）が連続この回数そろったら受け入れる
# ゲートが連続してこの回数拒否し続けたら（0.5 s）、保持姿勢の方が古いとみなして即・自動再ロックに入る。
# 2026-10-01 14:0x: 模型を手で動かした後、M3T は毎フレーム 35 mm/37° 先に収束するのにゲートが 20 回以上
# 拒否し、品質はデッドゾーン（0.36〜0.45）で再ロックも走らず、Unity から停止するまで固まった。
POSE_JUMP_STUCK_FRAMES    = 12
# 12:35 のログ: 跳び先 q 0.68 vs 保持 0.37 で受け入れた先が 29 mm ずれ（奥行き 25 mm 手前）だった。
# エッジだけの剛体整合は手前に寄った大きめの投影でも高く出るため、歯色（内部点・外側帯）を掛けた
# スコアで比較し、かつ 3 フレーム連続で同じ跳び先に収束したときだけ受け入れる。
ARCH_ANCHOR_KEY = "F" if SINGLE_BODY_MODE else "A"
arch_model_path = os.path.join(SCRIPT_DIR, "models", ARCH_MODEL_NAME)

# M3T の texture modality（実画像の特徴点で相対運動を拘束。モデルにテクスチャは不要）。
# 奥行き方向の暴走に効く可能性があるが未検証。
# 2026-10-09 ArUco 実測（texture OFF, F のみ, 18.5 cm）: 誤差 中央値 2.7 mm / 2.9°、軸別ゆらぎは x（≒歯列方向）1.8 mm >
# z 1.6 mm > y 0.7 mm、フレーム間変化は中央値 0.27 mm（高周波ノイズではなくゆっくり滑る）。歯列方向の滑り（歯の
# 周期性によるあいまいさ）を実画像の特徴点で拘束できるかを ON で A/B 検証中。env M3T_TEXTURE=0/1 で上書き可。
# A/B 結果（遮蔽なし F, 各 1 run）: OFF 中央値 2.7 mm / 2.9°（p90 4.7 mm）→ ON 1.6 mm / 2.5°（p90 3.7 mm）。
# m3t_step 6→15 ms だがカメラ 24 fps 律速のままで fps 低下なし。暫定で既定 ON（再現確認待ち）。
USE_TEXTURE_MODALITY = os.environ.get("M3T_TEXTURE", "1") != "0"

# 2アンカーは同じ typodont 座標系を共有するため姿勢スロットも共有でよいが、
# 単独運用の tracker_premolar_canine.py / tracker_canine.py の initial_poses/
# を汚さないよう、Anchor Switching 専用のキーで別ディレクトリに保存する。
_model_key        = "four_group_molar_premolar_canine_incisor"
pose_slots_dir    = os.path.join(SCRIPT_DIR, "initial_poses", _model_key)
calibration_path  = os.path.join(PYM3T_DIR, "camera_calibration.npz")
# 別のキャリブを使う（古い録画を当時の K で再生するときなど。2026-10-21）
calibration_path  = os.environ.get("M3T_CALIBRATION") or calibration_path
log_dir           = os.path.join(SCRIPT_DIR, "logs")

# OBJ の 1 モデル単位 = 何メートルか（typodont と同じスケールなら 0.04）
# 2026-09-09 22:30: 0.04 → 0.025 に修正。
# 根拠: 台座に固定した ArUco ボード（実寸既知）を基準に 3 視点で追跡姿勢を取り、
# 「追跡の並進を s 倍したときにボード座標系での姿勢が視点間で一致する s」を探すと
# s=0.625 で RMS 0.7 mm に収束（s=1.0 では 25.7 mm）。回転は元から ≤2° で一致。
# つまりモデルが 1.6 倍大きく、追跡・PnP の奥行きが 1.6 倍遠く出ていた（270 mm → 実際 ~165 mm）。
# これまでの mm 表記の誤差はすべて 1.6 倍に膨らんでいたことになる。
# 変更後は 4 グループの直線スパンが約 53 mm（0.04 では 84 mm）。定規で実測して確認すること。
MODEL_UNIT_IN_METER = 0.025

# 2アンカー間でどちらを「今のトラッキング結果」として採用するか切り替える際の
# ヒステリシス。品質が僅差でチラチラ切り替わる（ちらつき）のを防ぐため、
# 挑戦側がこのマージン以上優れていない限りは現在のアンカーを維持する。
ANCHOR_SWITCH_MARGIN = 0.05

# Unity 連携
UNITY_UDP_ENABLED   = True
UNITY_UDP_HOST      = "127.0.0.1"
UNITY_UDP_PORT      = 5005
UNITY_COMMAND_PORT  = 5006
UNITY_VIDEO_ENABLED = True
if REPLAY:
    UNITY_UDP_ENABLED = UNITY_VIDEO_ENABLED = False
UNITY_VIDEO_PORT    = 5010
UNITY_VIDEO_FPS     = 30   # 10→20→30: カメラは 24 fps 固定（2026-10-09 実測）なので 20 で間引くと
                           # テンポずれと最大 50 ms の表示遅れが出る。30 なら全フレーム送る（エンコードは別スレッド）
# Unity へ送る映像の幅。カメラが 720p になったので等倍で送る（640 だと Game ビュー拡大時にぼける）。
# Unity 側 M3TCameraIntrinsics の imageWidth/imageHeight/fx/fy/cx/cy はこの送信解像度に合わせること
# （現在: 1280x720, newK fx=1001.6 fy=1024.0 cx=640.2 cy=334.5）。
UNITY_VIDEO_WIDTH   = 1280
UNITY_VIDEO_JPEG_QUALITY = 70   # 80→70: Unity 側の LoadImage（メインスレッド）負荷とサイズを下げる

POSE_SLOT_COUNT = 12
POSE_SLOT_KEYS = {
    ord("1"): 1, ord("2"): 2, ord("3"): 3, ord("4"): 4,
    ord("5"): 5, ord("6"): 6, ord("7"): 7, ord("8"): 8,
    ord("9"): 9, ord("0"): 10, ord("-"): 11, ord("="): 12,
}

SNAP_START_STEPS = 30
SNAP_FRAME_STEPS = 3
SNAP_FRAME_COUNT = 20   # リカバリ後のスナップ期間を長くして安定化
AUTO_SELECT_STEPS = 8
# pym3t 内部の n_corr_iterations/n_update_iterations (5/2) は固定で Python
# から変更できないため、1フレームにつき tracker.step() を複数回呼ぶことで
# 探索範囲を広げる（特に速い回転で有効）。コストは概ね倍率分かかる。
# ただし molar_premolar クラスタのように歯冠が小さく輪郭が弱い対象では、
# 1フレームで3回も補正をかけると弱いエッジのノイズを増幅して暴走しやすい
# ことがログ解析（tz が1秒未満で100mm超ジャンプ）で見えたため、まずは
# 1（pym3tネイティブの1回分）に戻して暴走が減るか検証する。
TRACKING_STEPS_PER_FRAME = 1
ROTATION_REJECT_DEG = 120.0
ROTATION_PENALTY_WEIGHT = 4.0
QUALITY_SELECT_WEIGHT = 300.0  # 候補順位を支配する: (1 - edge_match) * これ

# 自動姿勢選択の候補生成: 保存済み12スロットだけでは隙間（10〜30度以上）
# ができることがあり、AUTO_SELECT_STEPS の範囲では埋まらない。各スロットに
# 安価な回転摂動を加えて候補を密にし、Cannyエッジとの粗い一致度で上位だけ
# 実際にトラッカーで精密化する。
COARSE_YAW_STEPS_DEG = (-20.0, -10.0, 0.0, 10.0, 20.0)
COARSE_PITCH_STEPS_DEG = (-10.0, 0.0, 10.0)
COARSE_TOP_K = 5

# Unity の Quick Align（auto_align）用の並進探索。従来の候補生成は保存スロットの
# 「回転だけ」を振るため、模型をスロット保存時と違う場所に置くと全く合わなかった
# （M3T の収束範囲は 1〜2 cm / 10〜15° 程度）。ここでは各基準姿勢（保存スロット・
# 現在姿勢・最後の良好姿勢）を中心に、画面平行方向・奥行き・yaw/pitch の格子で候補を
# 作り、粗採点（投影エッジの一致率、1 候補 0.4 ms）→ 上位セルの局所細分 → M3T 精密化
# の 3 段で絞る。粗採点は 1 基準姿勢あたり約 3600 候補 ≒ 1.5 s。
ALIGN_TRANS_SEARCH_ENABLED = True
ALIGN_XY_RANGE_M       = 0.040                       # 画面平行方向 ±40 mm
ALIGN_XY_STEP_M        = 0.010                       # 10 mm 刻み → 9×9
ALIGN_DEPTH_FACTORS    = (0.80, 0.90, 1.00, 1.12, 1.25)  # 奥行き倍率（同じ視線上で前後）
ALIGN_YAW_STEPS_DEG    = (-20.0, 0.0, 20.0)
ALIGN_PITCH_STEPS_DEG  = (-15.0, 0.0, 15.0)
ALIGN_LOCAL_CELLS      = 12                          # 局所細分に回す上位セル数
ALIGN_LOCAL_XY_M       = (-0.005, 0.0, 0.005)        # 細分: ±5 mm
ALIGN_LOCAL_DEPTH      = (0.95, 1.0, 1.05)
ALIGN_LOCAL_YAW_DEG    = (-10.0, 0.0, 10.0)
ALIGN_LOCAL_PITCH_DEG  = (-8.0, 0.0, 8.0)
ALIGN_NMS_TRANS_M      = 0.012                       # 近すぎるセルは同一候補とみなす
ALIGN_NMS_ROT_DEG      = 12.0
ALIGN_TOP_K            = 8                           # M3T 精密化に回す候補数
ALIGN_REFINE_STEPS     = 12
ALIGN_MIN_QUALITY      = 0.45                        # これ未満なら「見つからず」と警告
ALIGN_CENTER_MARGIN_PX = 60                          # 物体中心がこの縁幅より外の候補は捨てる
ALIGN_COARSE_TOL_PX    = 12.0   # 粗格子（10 mm 刻み）の採点許容 [縮小画像 px]。640 幅・150 mm で 1 px ≒ 0.3 mm
ALIGN_FINE_TOL_PX      = 6.0    # 局所細分（5 mm 刻み）の採点許容
# エッジ一致だけでは歯列の周期性（歯が約 8 mm ごとに繰り返す）のせいで 13 mm / 23° ずれた姿勢が
# 真値より高得点になる（オフライン検証 2026-09-29）。そこで「モデルの内部点が歯の色（低彩度・明）
# の画素に乗っている割合」を掛け合わせる。上下方向・pitch のずれで歯肉のピンクに乗ると急落する。
ALIGN_FILL_SAMPLES     = 3000   # モデル内部点として使う頂点数（全頂点からランダム抽出）
ALIGN_WHITE_S_MAX      = 0.35   # HSV 彩度の上限（歯: 中央値 0.16, 75% 0.30 / 歯肉: 0.65 以上）
ALIGN_WHITE_V_MIN      = 0.45   # HSV 明度の下限
# 白い台座やぼかした ArUco 板の上に乗った候補は内部点が全部「歯色」になってしまうので、
# 咬合面のすぐ外（模型座標で +y = 歯冠側。薄い軸が y で、-y が歯肉側）の帯も見て
# 「そこは歯色でない（背景の机・上顎のピンク）」ことを要求する。歯肉側の帯は使えない:
# この模型は歯の下が白い石膏で、赤いテープはさらに 1 cm 下にある（オフライン検証で below q=0.22）。
ALIGN_BAND_AXIS        = (0.0, 1.0, 0.0)
ALIGN_BAND_FRAC        = 0.30   # 内部点のうち歯冠側 30% を帯の元にする
ALIGN_BAND_OFFSET_M    = 0.004  # 帯を歯冠側へずらす量
# 歯列をほぼ真横（エッジオン）から見た候補は投影が細い帯に潰れ、歯の咬合縁に沿って置くだけで
# エッジ・歯色・外側帯の全部を満たしてしまう（オフライン検証: 真値より高得点）。
# 投影面積 × 奥行き² を基準姿勢（保存スロット）と比べ、大きく縮む候補は捨てる。
ALIGN_MIN_AREA_RATIO   = 0.60
ALIGN_MIN_SHORT_RATIO  = 0.50   # 投影 bbox の短辺 × 奥行き も基準の半分未満なら捨てる（細い帯の検出）
# 歯列は歯が約 8 mm 周期で並ぶため、歯 1 本分横にずれた偽姿勢が採点上ほぼ同点になる。
# デモでは模型を保存時とほぼ同じ場所に置く前提なので、基準姿勢（保存スロット等）からの距離に
# 応じた弱いペナルティで「近い解」を優先する（10 mm で 0.025、40 mm で 0.10）。
ALIGN_PRIOR_PER_M      = 2.5
ALIGN_PRIOR_FINAL_PER_MM = 1.5  # M3T 精密化後の最終採点 score（(1-q)*300 ベース）への加算 [点/mm]
VIEW_MODES = ("M3T", "WIREFRAME", "BLEND")
FRAME_READ_RETRY_LIMIT = 120

# 継続的な自動再ロック（品質が落ちたら毎回全探索）は暴れやすい（pose が
# ふらつく）ことが実運用で分かったため、デフォルトは OFF。品質計算と
# last_good_pose によるフリーズ送信自体は常に有効（Unity 側が飛ばないための
# 安全策）。't' キーでライブに切り替え可能。
QUALITY_CHECK_ENABLED = True
# 投影したエッジが（同じ向きの）実エッジにこの距離以内なら「ヒット」。
# 2026-09-09 の録画フレームで総当たり: 4px・隣接±1ビンで、正解=1.00 / 2〜4px ズレ=0.98〜0.91 /
# 45°・90° 誤配置=0.64・0.48 / 完全に外れた配置=0.21。6px だと誤配置が 0.75・0.64 まで上がる。
QUALITY_EDGE_MAX_DIST_PX = 4.0
# 投影エッジの向き（法線）と画像勾配の向きが一致するエッジだけをヒットと数える。
# 2026-09-09 のログ解析: 向きを問わない従来方式では、モデルが完全に外れていても
# 歯列付近はエッジが密なので q=0.2〜0.5、3本歯のアンカーが歯列の別の場所に乗ると
# q=1.0 になり、ロスト判定も良好判定も機能していなかった。
QUALITY_ORIENTED_EDGES = True
# 閾値は方向付きマッチング導入後の実測（2026-09-09 17:01 のログ、CSV の qA〜qD 列と
# 保存フレームを目視ラベル付け）から決めた。A/B/C 平均は
#   合っている: 中央値 0.47（p10 0.39〜p90 0.52） / 臼歯が空中に浮いた誤姿勢: 0.37〜0.38
# と差が小さいので、1フレームの値ではなく EMA 平滑化した値で判定し、連続フレーム数も長めに取る。
# 単体ボディモードでは値域が上がる（2026-09-09 21:46 の計測: 正解 1〜2 mm のとき 0.73〜0.76、
# 遮蔽で 38 mm ずれたとき 0.50〜0.63）ので閾値を別に持つ。
# 2026-09-09 22:51: MODEL_UNIT 0.025 にした後の実測では、合っているとき（基準誤差 <8 mm）の
# q_raw は中央値 0.53（p10 0.45〜p90 0.61）、外れているとき（>12 mm）は 0.32（p90 0.49）と、
# 旧スケール時（0.73〜0.76）より全体に低い。0.60/0.70 のままだと常時 HOLD → 3 秒ごと自動再ロック
# の空回りになった（28 回/3 分）ので、実測に合わせて下げる。値域がスケール・視点で動くので、
# 将来的には追跡開始直後の q を基準にした相対閾値にするのが望ましい。
QUALITY_LOSS_THRESHOLD  = 0.40 if SINGLE_BODY_MODE else 0.40   # 平滑化スコアがこれ未満で続いたらロスト
QUALITY_GOOD_THRESHOLD  = 0.48 if SINGLE_BODY_MODE else 0.46   # これ以上で lost_streak をリセット/「良好」
QUALITY_EMA_ALPHA       = 0.25   # 剛体整合スコアの EMA 係数（約 4 フレームの時定数）
# ロスト直後は「遮蔽かもしれない」ので、まず last_good_pose を保持して毎フレームそこから追跡を
# 再試行する（HOLD）。遮蔽が去れば姿勢は正しいままなので即座に再ロックする。
# HOLD_BEFORE_RECOVERY_SEC を過ぎても戻らなければ本当のロストとみなして自動再ロック（全探索）へ。
HOLD_BEFORE_RECOVERY_SEC = 3.0
LAST_GOOD_MAX_AGE_SEC   = HOLD_BEFORE_RECOVERY_SEC   # これより古い last_good_pose へは同期で引き戻さない
# ロスト/良好判定に使う「剛体整合スコア」: 同じ pose を RIGID_QUALITY_KEYS の各アンカーの
# ジオメトリで評価し、上位 RIGID_QUALITY_TOP_K の平均を取る。3 = 単純平均。
# 上位 2 平均だと、臼歯（A）が完全に外れていても B/C だけで 0.54 になり見逃した。
# D（切歯）は取り外し式義歯パーツ＋ガーゼで見た目がモデルと一致しないため既定では除外する。
RIGID_QUALITY_KEYS  = ("F",) if SINGLE_BODY_MODE else ("A", "B", "C")
RIGID_QUALITY_TOP_K = 3
# 自動アンカー切替・スナップ時のアンカー選択の候補。D は Unity から手動指定した
# ときだけ使う（D に切り替わった瞬間に pose が 100mm 以上飛ぶのをログで確認）。
AUTO_ANCHOR_KEYS    = ("F",) if SINGLE_BODY_MODE else ("A", "B", "C")
QUALITY_LOSS_CONSECUTIVE = 20    # 何フレーム連続で悪いと再ロックを試みるか（約 1 秒 @20fps）
RECOVERY_COOLDOWN_SEC    = 0.8   # 自動再ロックの最短間隔（短くして素早く再試行）
# 自動再ロックが失敗（見つけた姿勢の品質 < LOSS）したら回数を数え、1 回目失敗後は回転ガードなしの全探索、
# RELOCK_FAILS_BEFORE_SEARCH 回目以降は Quick Align と同じ並進探索（3〜5 s）に切り替える。
# 2026-10-01 14:27 run: 模型を大きく動かした後、スロット近傍だけの再ロックが q 0.01〜0.08 で 30 秒以上
# 失敗し続けた（しかも失敗結果を last_good_pose にして quality=1.0 にリセットしていた）。
RELOCK_FAILS_BEFORE_SEARCH = 2
RELOCK_MAX_TRANS_M        = 0.020   # guard=on の再ロックで、基準姿勢（last_good）からこれ以上離れた候補は採用しない
# guard=on の再ロックが失敗しても、ロストがこの秒数続くまでは全探索・並進探索へ上げない（遮蔽が続いている間に
# 範囲を広げると、見えている別の歯や台座に合ってしまう）。
RELOCK_ESCALATE_AFTER_SEC = 5.0
# ── ① HOLD 中の惰性追跡（2026-10-14）────────────────────────────────────────
# 12:09 run のロスト 14 回はすべて「手で模型を持って動かす → 歯群が 2〜3 個同時に隠れる → 剛体スコア < 0.40 → HOLD」
# で、HOLD は毎フレーム last_good_pose に戻す（姿勢の凍結）ため、模型が動き続けると収束範囲（1〜2 cm）を外れて
# 全探索（SNAP）に落ちていた。真の移動速度は 6〜14 mm/s と遅く、遮蔽なしの F は 2〜12 mm/s で追従比 ≈1.0（遅れなし）。
# → q が落ちた瞬間に模型が動いていた（直前の出力速度 ≥ COAST_MIN_SPEED）なら、COAST_MAX_SEC の間は last_good に戻さず
#   「直前の速度で 1 フレーム先を予測した姿勢」から M3T を進め続ける（惰性追跡）。暴走は跳び量ゲートと
#   HOLD_BEFORE_RECOVERY_SEC 後の再ロックで止める。静止中の遮蔽（速度ほぼ 0）は従来どおり凍結する。
COAST_ENABLED               = os.environ.get("M3T_COAST", "1") != "0"
COAST_MIN_SPEED_M_PER_FRAME = 0.00025  # 0.25 mm/フレーム ≈ 6 mm/s @24 fps。これ未満なら「静止中の遮蔽」とみなして凍結
COAST_MAX_SEC               = 1.5      # 惰性で追う最長時間（これを過ぎたら従来の HOLD → 再ロック）
COAST_VEL_ALPHA             = 0.3      # 出力速度の EMA 係数
COAST_VEL_DECAY             = 0.92     # 惰性中は毎フレーム速度を減衰（予測が飛び続けないように）
COAST_VEL_MAX_M             = 0.010    # 1 フレームにこれ以上動いた出力は「跳び」なので速度に入れない
# ── ② ロスト判定スコア（2026-10-14）──────────────────────────────────────────
# 従来は A/B/C の単純平均。1 歯群が手で隠れただけで 0.40 を割って HOLD に落ちていた。"top2" では A/B/C/D の上位 2 つの
# 平均も計算し、大きい方を HOLD 判定に使う（遮蔽に強い）。既知のリスク（上のコメント）: A が完全に外れていても B/C で
# 0.54 になる誤適合を見逃しやすい。last_good_pose の更新・再ロックの合否・復帰時再探索は従来の厳しいスコアのまま。
# ArUco で「q ≥ 0.4 なのに誤差 > 5 mm」が増えないか確認して既定を決める。"abc3" で従来動作。
# ── 案 A: 単独歯群の統合を禁止（2026-10-14 12:45 run）────────────────────────
# 統合中の大きなずれ（回転 15° 超 75 フレーム、並進 p90 20 mm）はほぼ B 単独統合。B は奥行き 6.5〜8 mm の偏りと
# 回転 10° 超の履歴がある。「A を含む」か「2 歯群以上」のときだけ統合し、B/D だけなら保持（＋等速予測）にする。
FUSION_REQUIRE_A_OR_PAIR = os.environ.get("M3T_FUSION_SOLO", "0") != "1"   # M3T_FUSION_SOLO=1 で従来（単独でも統合）
# ── 案 B: 等速予測（2026-10-14）────────────────────────────────────────────────
# 出力速度（惰性追跡と同じ EMA）で 1 フレーム先を予測した姿勢から M3T を進める。統合中は常に、F 追跡中は
# PREDICT_F_MIN_SPEED_M 以上で動いているときだけ（静止時にノイズを足さない）。12:45 run で 24〜48 mm/s の
# F 誤差が 4.0 mm / 3.6°（〜24 mm/s では 1.5 mm / 0.7°）だったのを縮める狙い。
PREDICT_ENABLED           = os.environ.get("M3T_PREDICT", "1") != "0"
PREDICT_F_MIN_SPEED_M     = 0.0005  # 0.5 mm/フレーム ≈ 12 mm/s
FUSION_HOLD_PREDICT_MAX_SEC = 1.0   # 使える歯群がない間、保持姿勢を速度で進める最長時間（速度は毎フレーム減衰）
# 予測の暴走対策（2026-10-19 OMEN 14:43 run）: 静止した模型で統合 A+B → F に戻った瞬間の出力の跳ね（2〜3 mm/フレーム）が
# 速度に入り、F 追跡中の予測が 0.5 mm/フレームを超えたまま自己維持して、奥行きが 1 s で 165→188 mm（ArUco 49 mm / 23°）に
# 滑った（q は 0.62→0.90 に上がる「静かな誤り」）。奥行きは画像からの拘束が弱く、予測で進めた分を M3T がほとんど
# 引き戻さないため止まらない。① 姿勢源が F に切り替わったフレームは速度を 0 にする、② F 追跡中は奥行き（カメラ z）を
# 予測しない。統合中・保持中の予測は従来どおり（遮蔽中の追従に使う）。M3T_PREDICT_DEPTH=1 で②を無効化。
PREDICT_RESET_ON_RETURN_TO_F = os.environ.get("M3T_PREDICT_RESET", "1") != "0"
PREDICT_F_DEPTH              = os.environ.get("M3T_PREDICT_DEPTH", "0") == "1"
# ③ 15:11 run（①②導入後、静止模型・F 追跡中・切替なし）: 15.0 s のわずかな動き（+1 mm/フレーム）で速度が閾値を超え、
# 今度は上下（y）に 1→3.6 mm/フレームと加速しながら 1.2 s で 47 mm / 40°（q 0.57→0.91）。奥行きだけ止めても横方向・回転で
# 同じ自己維持が起きるので、F 追跡中の予測は既定 OFF にする（M3T_PREDICT_F=1 で従来）。統合中・保持中の予測は残す。
# 本来は「予測で進めた分を M3T が押し戻したら速度を減衰」（補正量と速度の内積で判定）のほうがよい。
PREDICT_F_ENABLED            = os.environ.get("M3T_PREDICT_F", "0") == "1"
# ── 歯でない場所への誤ロック検知（2026-10-14 13:00 run）────────────────────────
# 手で隠しながら動かした後、F が台座の白黒模様＋スタンドの縁に合い、q 0.72〜0.93 のまま 13 s 外れ続けた（ArUco 68 mm / 35°）。
# 剛体スコアは厳格版でも 0.66 を保ち、q 系の指標では捕まらない。歯色マスク（ALIGN_WHITE_*、並進探索の採点に既存）に
# F モデルの内部点が乗っている割合は、正しい追跡で 0.42〜0.81（中央値 0.68〜0.77）、この誤ロックで 0.13〜0.19 と分かれる。
# F 追跡中（統合でない）に q が良好なのにこの割合が低い状態が続いたら、誤ロックとみなして並進探索の再ロックへ落とす。
# 遮蔽中（統合）は手が歯を覆って割合が 0.07 まで落ちるので判定しない。歯の上でずれる誤ロック（12:45 run の 31 mm、割合 0.62）は捕まらない。
NONTOOTH_CHECK_ENABLED   = os.environ.get("M3T_NONTOOTH_CHECK", "1") != "0"
NONTOOTH_FILL_MIN        = 0.30    # 歯色マスクに乗っている内部点の割合（EMA）がこれ未満で
NONTOOTH_SEC             = 1.0     # この秒数続いたら誤ロック（13:12 run: 統合で 1 s ごとに中断されタイマーが戻って発火しなかった →
                                   #   統合中はタイマーを止めるだけでリセットしない。1.5→1.0 s）
NONTOOTH_FILL_SEVERE     = 0.20    # これ未満なら NONTOOTH_SEVERE_SEC で即判定（13:12 run の誤ロックは 0.13〜0.17）
NONTOOTH_SEVERE_SEC      = 0.5
# 奥行きの窓: 追跡開始後、最初に良好になった姿勢の奥行きを基準に、この比率の外へ出た姿勢は跳び量ゲートで拒否する
DEPTH_WINDOW_MIN_RATIO   = 0.65
DEPTH_WINDOW_MAX_RATIO   = 1.5
NONTOOTH_MIN_QUALITY     = 0.46    # 判定するのは q がこれ以上（追跡できているつもり）のときだけ
NONTOOTH_EVAL_EVERY      = 2       # 評価間隔（フレーム）。歯色マスクは 720p で 3〜4 ms
NONTOOTH_FILL_EMA_ALPHA  = 0.3
LOST_SCORE_MODE          = os.environ.get("M3T_LOST_SCORE", "top2")
LOST_SCORE_LENIENT_KEYS  = ("A", "B", "C", "D")
LOST_SCORE_LENIENT_TOP_K = 2

# 良い品質が続いた姿勢を自動でスロットに登録する機能。患者の頭や模型の向きが
# 変わるたびにクリック初期化をやり直す手間を減らすため、追跡中に安定して
# 良品質が続いた姿勢を自動保存し、再ロック候補の姿勢バリエーションを
# 使うたびに自然に増やしていく。スロット1はクリック初期化用に予約し、
# 自動登録では触らない。
# 2026-09-09: GOOD 閾値が 0.20 と甘かったため、完全にロストした姿勢が「良品質」
# と判定され、スロット 2〜9 がすべて不良姿勢で上書きされた（initial_poses/.../
# polluted_20260909_autocapture/ に退避）。新しい品質指標の値域を確認するまで
# 既定 OFF。ON にする場合も下の 3 条件をすべて満たしたときだけ登録する。
AUTO_POSE_CAPTURE_ENABLED = False
AUTO_CAPTURE_MIN_GOOD_STREAK = 60     # 剛体整合スコア >= AUTO_CAPTURE_MIN_QUALITY がこのフレーム数連続
AUTO_CAPTURE_MIN_QUALITY = 0.85       # GOOD 閾値より厳しい登録専用の閾値
AUTO_CAPTURE_AFTER_RECOVERY_SEC = 3.0  # 再ロック直後はスナップ中の見かけ上の高品質を登録しない

# オフラインで品質指標・閾値を調整するため、追跡中の生フレームを N フレームごとに
# logs/tracking_<ts>_frames/ に保存する（CSV の frame 列と対応）。0 で無効。
DIAG_FRAME_DUMP_EVERY = 0 if REPLAY else 20   # 再生時は元の録画があるので保存しない
AUTO_CAPTURE_MIN_ANGLE_DEG = 10.0     # 既存スロットとこの角度以上離れていたら新規とみなす
AUTO_CAPTURE_MIN_TRANSLATION_M = 0.01  # 既存スロットとこの距離以上離れていたら新規とみなす
AUTO_CAPTURE_COOLDOWN_SEC = 3.0       # 自動登録の最短間隔
AUTO_CAPTURE_SLOTS = list(range(2, 10))  # 自動登録で使うスロット（ラウンドロビン）

# ─── カメラ初期化 ─────────────────────────────────────────────────────────

# 歯・顎模型が白く光沢があるため、自動露出だとハイライトが白飛びして
# 輪郭（明暗の境界）が消えてしまうことがある。露出を手動で少し下げて
# クリッピングを抑えられるか試すためのスイッチ。
# None = 自動露出のまま（元の挙動）。数値を入れると自動露出をOFFにして
# その値を設定する（DirectShowは露出を2の指数スケールで扱うことが多く、
# -4 前後が明るめ、-8 前後で大分暗くなる。カメラ機種依存なので実際の
# 映像を見ながら調整すること）。
CAMERA_MANUAL_EXPOSURE = -6

# カメラに要求する解像度。iPhone SE（iVCam 等の仮想カメラ経由）は既定で 640x480 しか出さないが、
# 配信アプリ側の設定を上げれば 1280x720 / 1920x1080 が取れる。1 px の実寸が半分になるので
# 追跡・基準の両方の精度下限が改善する。解像度を変えたら calibrate_camera.py で再キャリブレーション
# すること（camera_calibration.npz の image_width/height と一致しないと仮の K になる）。
# None にすると配信アプリの既定値をそのまま使う。
# カメラ番号・バックエンド・解像度は camera_io.py（環境変数 / m3t_local.env）で PC ごとに変えられる。
# 既定は Windows=DirectShow / Mac=AVFoundation、カメラ 0、1280x720。
CAMERA_WIDTH, CAMERA_HEIGHT = camera_io.camera_size((1280, 720))
# 露出の手動設定は Windows（DirectShow）だけ。AVFoundation は未対応の値を受けて挙動が変わることがあるので触らない
if os.name != "nt":
    CAMERA_MANUAL_EXPOSURE = None

# カメラ番号は camera_io.camera_index(): M3T_CAMERA_INDEX があればそれ、無ければ Windows では DirectShow の
# デバイス名（M3T_CAMERA_NAME、既定 iVCam）で探す（OMEN では内蔵カメラが 0、iVCam が 0 か 1 で接続状況により変わる）。
class ReplaySource:
    """録画クリップ（video.avi + frames.csv）を読むカメラ代わり。read() のたびに仮想時計をそのフレームの録画時刻に進める。"""

    def __init__(self, clip_dir):
        self._cap = cv2.VideoCapture(os.path.join(clip_dir, "video.avi"))
        with open(os.path.join(clip_dir, "frames.csv"), encoding="utf-8") as f:
            self._ts = [float(r["timestamp"]) for r in csv.DictReader(f)]
        self._fi = 0
        self._pending = None
        if self._ts:
            _replay_clock[0] = self._ts[0]

    def peek(self):
        """最初のフレームを読んで返す（次の read() でも同じフレームを返す。画像サイズの取得用）。"""
        ok, img = self.read()
        if ok:
            self._pending = img
        return ok, img

    def read(self):
        if self._pending is not None:
            img, self._pending = self._pending, None
            return True, img
        ok, img = self._cap.read()
        if not ok or self._fi >= len(self._ts):
            return False, None
        _replay_clock[0] = self._ts[self._fi]
        self._fi += 1
        return True, img

    def set(self, *args):
        return False

    def get(self, prop):
        return self._cap.get(prop)

    def release(self):
        self._cap.release()


if REPLAY:
    cap = ReplaySource(REPLAY_CLIP_DIR)
    CAMERA_MANUAL_EXPOSURE = None
    print(f"[replay] {REPLAY_CLIP_DIR}")
else:
    cap, _ = camera_io.open_camera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)

if CAMERA_MANUAL_EXPOSURE is not None:
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_MANUAL_EXPOSURE)

ret, frame = cap.peek() if REPLAY else cap.read()
if not ret:
    raise RuntimeError("カメラの読み取りに失敗しました")

current_exposure = CAMERA_MANUAL_EXPOSURE if CAMERA_MANUAL_EXPOSURE is not None else cap.get(cv2.CAP_PROP_EXPOSURE)

if CAMERA_MANUAL_EXPOSURE is not None:
    actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    actual_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    print(f"手動露出を設定: 要求={CAMERA_MANUAL_EXPOSURE} 実際の値(exposure={actual_exposure}, auto_exposure={actual_auto})")
    print("  ドライバによっては反映されないことがある。'[' で露出を下げる/']' で上げる（実行中に調整可能）。")

h, w = frame.shape[:2]


# ─── カメラ読み取りの別スレッド化（ラグ対策） ─────────────────────────────────
# cap.read() をメインループで直接呼ぶと、M3T と品質評価の間にドライバ側へ溜まった
# 「古い」フレームを次に受け取ることになり、画面の遅れ（ラグ）として見える。
# 読み取りを専用スレッドに移し、メインループは常に最新の 1 枚だけを処理する。
# このカメラは解像度・露出・バックエンドに依らず 24 fps 固定（2026-10-09 実測）なので
# fps は変わらないが、処理・表示するフレームの鮮度が上がる。
# 注意: '[' ']' の露出変更は別スレッドの read と並行して cap.set を呼ぶ（DSHOW では実用上問題なし）。
class _LatestFrameGrabber:
    def __init__(self, capture):
        self._cap = capture
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            with self._cond:
                if ret:
                    self._frame = frame
                    self._seq += 1
                self._cond.notify_all()
            if not ret:
                time.sleep(0.005)

    def read(self, last_seq, timeout=0.5):   # 0.5 s × FRAME_READ_RETRY_LIMIT(120) ≒ 60 s でカメラ停止を検知
        """last_seq より新しいフレームが届くまで待ち (ret, frame, seq) を返す。timeout で (False, None, last_seq)。"""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq <= last_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None, last_seq
                self._cond.wait(remaining)
            return True, self._frame, self._seq

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)


class _ReplayGrabber:
    """再生時は録画したフレームを 1 枚ずつ順に渡す（間引かない）。"""

    def __init__(self, capture):
        self._cap = capture
        self._seq = 0

    def read(self, last_seq, timeout=0.5):
        ret, frame = self._cap.read()
        if ret:
            self._seq += 1
        return ret, frame, self._seq

    def stop(self):
        pass


_grabber = _ReplayGrabber(cap) if REPLAY else _LatestFrameGrabber(cap)
_frame_seq = 0

# 品質評価（方向付きエッジ距離マップ）は 640 幅相当に縮小して計算する。
#  - 速度: 720p では 9 枚の距離変換が 3 倍重くなり fps が 13 → 6.7 に落ちた（2026-09-21）
#  - 値の一貫性: QUALITY_EDGE_MAX_DIST_PX（4 px）や閾値は 640x480 の実測で決めたもの。
#    フル解像度で評価すると同じズレでも px が 2 倍になり q が 0.7 → 0.25 に落ちて、
#    HOLD/自動再ロックが空回りした。縮小して評価すれば解像度に依らず同じ値域になる。
# 追跡（M3T）自体はフル解像度で動く。
QUALITY_MAX_WIDTH = 640
QUALITY_EVAL_EVERY = 2      # 距離マップの再計算間隔（フレーム）。1 = 毎フレーム
dist_map = None
QUALITY_SCALE = min(1.0, QUALITY_MAX_WIDTH / float(w))
if QUALITY_SCALE < 1.0:
    print(f"品質評価は {int(w*QUALITY_SCALE)}x{int(h*QUALITY_SCALE)} に縮小して計算（scale={QUALITY_SCALE:.3f}）")

K = np.array([
    [800,   0, w / 2],
    [  0, 800, h / 2],
    [  0,   0,     1],
], dtype=np.float32)
dist = None
undistort_map = None

# 歪み補正の ON/OFF（A/B 検証用、2026-10-09）。モニター表示のチェッカーボードで取った npz は高次の
# 歪み係数（k2=-1.6, k3=3.0）が大きく、remap が画像を微妙に歪めて全アンカーの適合度を一律に下げている
# 疑いがある。OFF のときはキャリブの K をそのまま使い、歪み係数は使わない（画像は無補正）。
# 環境変数 M3T_UNDISTORT=1/0 で上書き可。
# A/B 結果 (2026-10-09 12:44 run): OFF でも剛体スコア中央値 0.351→0.347 と不変 → 歪み補正は原因ではない。既定 ON に戻した。
UNDISTORT_ENABLED = os.environ.get("M3T_UNDISTORT", "1") != "0"

if os.path.exists(calibration_path):
    calib = np.load(calibration_path)
    calib_w = int(calib["image_width"][0])
    calib_h = int(calib["image_height"][0])
    if calib_w == w and calib_h == h:
        K    = calib["K"].astype(np.float32)
        dist = calib["dist"].astype(np.float32)
        if UNDISTORT_ENABLED:
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
            undistort_map = cv2.initUndistortRectifyMap(
                K, dist, None, new_K.astype(np.float32), (w, h), cv2.CV_16SC2)
            K = new_K.astype(np.float32)
            print("カメラキャリブレーション読み込み完了（歪み補正 ON）")
        else:
            dist = None
            print("カメラキャリブレーション読み込み完了（歪み補正 OFF: K のみ使用）")
    else:
        print(f"キャリブレーション解像度不一致: {calib_w}x{calib_h} vs {w}x{h}")
else:
    print("camera_calibration.npz が見つからないため仮の K を使用")


def prepare_frame(raw):
    if undistort_map is not None:
        return cv2.remap(raw, undistort_map[0], undistort_map[1], cv2.INTER_LINEAR)
    return raw


# ─── マーカー基準（精度評価用。追跡自体はマーカーを使わない） ─────────────────
# 台座に貼った ArUco（markers/aruco_sheet_A4.pdf）の姿勢を毎フレーム推定し、追跡が合っている
# ときに 'm' を押して「マーカー→歯牙モデル」の固定変換を登録すると、以後はマーカーから予測した
# 歯牙姿勢とマーカーレス追跡の姿勢の差を mm / 度で HUD と CSV に出す。
# フレームは prepare_frame で歪み補正済みなので、K は補正後の値・歪み係数ゼロで扱う。
MARKER_REF_ENABLED = True
MARKER_PRINT_SCALE = 1.0   # 印刷倍率が 100% でなかったときの補正（実測 mm / 設計 mm）
# 追跡器と品質評価に渡す画像からマーカー領域を塗り消す。2026-09-09 20:32 の計測で、飛んだモデルが
# ボードの強いエッジ上で「良い」と評価されて固定された（B の q だけ上昇）ため。検出は元画像で行う。
MARKER_MASK_FOR_TRACKER = True
MARKER_MASK_METHOD = "blur"      # blur(6.8ms, 残エッジ最少) / inpaint(23ms) / fill(12ms)  ※保存フレームで計測
MARKER_MASK_PAD_PX = int(round(25 * w / 640.0))   # マーカー外側の白い台紙の縁まで消す（640 幅で 25 px 相当）
from marker_reference import MarkerReference
marker_ref = None
if MARKER_REF_ENABLED:
    _marker_reg_path = os.path.join(pose_slots_dir, "marker_registration.json")
    if REPLAY and os.path.exists(os.path.join(REPLAY_CLIP_DIR, "marker_registration.json")):
        _marker_reg_path = os.path.join(REPLAY_CLIP_DIR, "marker_registration.json")   # 録画時点の登録で評価する
    marker_ref = MarkerReference(K, None, _marker_reg_path, scale=MARKER_PRINT_SCALE)
    if marker_ref.T_ref_model:
        for _k in marker_ref.T_ref_model:
            _s = marker_ref.summary(_k)
            print(f"マーカー基準の登録を読み込み: {_k} サンプル {_s['n']} 件"
                  + (f"（ばらつき RMS {_s['rms_mm']:.1f} mm / {_s['rms_deg']:.1f} deg）" if _s['n'] > 1 else ""))


# ─── OBJ エッジ読み込み ───────────────────────────────────────────────────

def load_obj_edges(path, max_edges=500, sharp_angle_deg=90.0):
    """OBJ を読み込み、幾何学的に鋭いエッジ（歯間の谷・稜線など実際の折れ目）
    だけを残す。歯肉/歯牙の色境界のような、3Dの折れ目が無い平坦面の境界を
    誤って拾わないよう、二面角（隣接する2面の法線のなす角）でフィルタする。
    """
    vertices, faces = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                p = line.split()
                if len(p) >= 4:
                    vertices.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                ids = []
                for tok in line.split()[1:]:
                    try:
                        ids.append(int(tok.split("/")[0]) - 1)
                    except Exception:
                        pass
                if len(ids) >= 3:
                    faces.append(ids)

    vertices = np.array(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError(f"OBJ の頂点を正しく読み込めませんでした: {vertices.shape}")

    n_verts = vertices.shape[0]
    tri_faces = [f for f in faces if len(f) == 3]
    other_faces = [f for f in faces if len(f) != 3]

    tri_arr = np.array(tri_faces, dtype=np.int64)
    p0, p1, p2 = vertices[tri_arr[:, 0]], vertices[tri_arr[:, 1]], vertices[tri_arr[:, 2]]
    tri_normals = np.cross(p1 - p0, p2 - p0)
    tri_norm_len = np.linalg.norm(tri_normals, axis=1, keepdims=True)
    tri_norm_len[tri_norm_len < 1e-12] = 1.0
    tri_normals = (tri_normals / tri_norm_len).astype(np.float32)

    tri_edge_pairs = np.stack([
        tri_arr[:, [0, 1]], tri_arr[:, [1, 2]], tri_arr[:, [2, 0]],
    ], axis=1).reshape(-1, 2)
    tri_face_ids = np.repeat(np.arange(len(tri_faces), dtype=np.int64), 3)

    other_normals, other_edge_a, other_edge_b, other_face_ids = [], [], [], []
    face_offset = len(tri_faces)
    for face_i, face in enumerate(other_faces):
        pts = vertices[face]
        n = len(pts)
        normal = np.zeros(3, dtype=np.float64)
        for i in range(n):
            cur, nxt = pts[i], pts[(i + 1) % n]
            normal[0] += (cur[1] - nxt[1]) * (cur[2] + nxt[2])
            normal[1] += (cur[2] - nxt[2]) * (cur[0] + nxt[0])
            normal[2] += (cur[0] - nxt[0]) * (cur[1] + nxt[1])
        norm_len = np.linalg.norm(normal)
        other_normals.append(normal / norm_len if norm_len > 1e-12 else np.array([0.0, 0.0, 1.0]))
        for i in range(n):
            other_edge_a.append(face[i])
            other_edge_b.append(face[(i + 1) % n])
            other_face_ids.append(face_offset + face_i)

    if other_faces:
        all_normals = np.concatenate(
            [tri_normals, np.array(other_normals, dtype=np.float32).reshape(-1, 3)], axis=0)
        edge_a_arr = np.concatenate([tri_edge_pairs[:, 0], np.array(other_edge_a, dtype=np.int64)])
        edge_b_arr = np.concatenate([tri_edge_pairs[:, 1], np.array(other_edge_b, dtype=np.int64)])
        face_idx_arr = np.concatenate([tri_face_ids, np.array(other_face_ids, dtype=np.int64)])
    else:
        all_normals = tri_normals
        edge_a_arr = tri_edge_pairs[:, 0]
        edge_b_arr = tri_edge_pairs[:, 1]
        face_idx_arr = tri_face_ids

    a_can = np.minimum(edge_a_arr, edge_b_arr).astype(np.int64)
    b_can = np.maximum(edge_a_arr, edge_b_arr).astype(np.int64)
    keys = a_can * n_verts + b_can

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_faces = face_idx_arr[order]

    change = np.concatenate([[True], sorted_keys[1:] != sorted_keys[:-1]])
    group_starts = np.nonzero(change)[0]
    group_ends = np.concatenate([group_starts[1:], [len(sorted_keys)]])
    two_face_mask = (group_ends - group_starts) == 2
    two_face_starts = group_starts[two_face_mask]
    face_a_idx = sorted_faces[two_face_starts]
    face_b_idx = sorted_faces[two_face_starts + 1]

    na = all_normals[face_a_idx]
    nb = all_normals[face_b_idx]
    cos_angle = np.clip(np.sum(na * nb, axis=1), -1.0, 1.0)
    dihedral_deg = np.degrees(np.arccos(cos_angle))

    sharp_mask = dihedral_deg >= sharp_angle_deg
    sharp_keys = sorted_keys[two_face_starts[sharp_mask]]

    sharp_a = (sharp_keys // n_verts).astype(int)
    sharp_b = (sharp_keys % n_verts).astype(int)
    edges = list(zip(sharp_a.tolist(), sharp_b.tolist()))

    print(f"  シャープエッジ (dihedral >= {sharp_angle_deg} deg): {len(edges)} / {len(group_starts)} 総エッジ")

    if len(edges) == 0:
        raise RuntimeError("シャープエッジが見つかりませんでした。sharp_angle_deg を下げてください。")

    if len(edges) > max_edges:
        idx = np.linspace(0, len(edges) - 1, max_edges).astype(int)
        edges = [edges[i] for i in idx]

    return vertices, edges


def rot_x(deg):
    a = np.deg2rad(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], dtype=np.float32)


def rot_y(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], dtype=np.float32)


def rot_z(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=np.float32)


# ─── One-Euro フィルタ（Unity 送信用の姿勢だけ平滑化） ───────────────────

def rotmat_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw, qx, qy, qz = 0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw, qx, qy, qz = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw, qx, qy, qz = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw, qx, qy, qz = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_to_rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float32)


def slerp(q0, q1, t):
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot) * t
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def one_euro_smoothing_factor(t_e, cutoff):
    r = 2.0 * np.pi * cutoff * t_e
    return r / (r + 1.0)


class OneEuroFilter:
    """標準的な One-Euro フィルタ（位置用）。静止時は強く平滑化し、動きが
    速くなるほど生の値に近づく。"""

    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.x_prev = self.dx_prev = self.t_prev = None

    def filter(self, x, t):
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()
        t_e = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / t_e
        a_d = one_euro_smoothing_factor(t_e, self.dcutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        speed = float(np.linalg.norm(dx_hat))
        cutoff = self.mincutoff + self.beta * speed
        a = one_euro_smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class OneEuroRotationFilter:
    """回転版 One-Euro フィルタ。角速度でカットオフを駆動し、slerp で補間する。"""

    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.q_prev, self.speed_prev, self.t_prev = None, 0.0, None

    def filter(self, R, t):
        q = rotmat_to_quat(R)
        if self.q_prev is None:
            self.q_prev, self.t_prev = q, t
            return R.copy()
        t_e = max(t - self.t_prev, 1e-3)
        dot = np.clip(abs(float(np.dot(q, self.q_prev))), -1.0, 1.0)
        speed = (2.0 * np.arccos(dot)) / t_e
        a_d = one_euro_smoothing_factor(t_e, self.dcutoff)
        speed_hat = a_d * speed + (1.0 - a_d) * self.speed_prev
        cutoff = self.mincutoff + self.beta * speed_hat
        a = one_euro_smoothing_factor(t_e, cutoff)
        q_hat = slerp(self.q_prev, q, a)
        self.q_prev, self.speed_prev, self.t_prev = q_hat, speed_hat, t
        return quat_to_rotmat(q_hat)


# cutoff = mincutoff + beta*speed なので、beta/mincutoffが大きいほど「速い動き」
# を生の値に近く通してしまう。今回のように速い動き=ほぼ全てトラッキング誤差
# （実際の頭部の動きではない）である場合、betaを大きく下げて、動きが速く見えても
# 平滑化を強く残すようにする（完全フリーズではなく、大きな飛びを鈍らせる方向）。
# 2026-09-24: mincutoff 0.5 Hz は時定数 ~0.3 s で、Unity 側の模型が 0.5〜1 秒遅れて追従して見えた
# （「ラグ」の主因）。単体ボディ化で追跡自体が暴れなくなったので、応答優先に上げる。
# beta を上げると速い動きはほぼ生値で通り、静止時だけジッターを抑える One-Euro 本来の動きになる。
pos_filter = OneEuroFilter(mincutoff=2.5, beta=0.6)
rot_filter = OneEuroRotationFilter(mincutoff=2.5, beta=0.3)


def smooth_pose(raw_pose):
    """Unity へ送る姿勢だけ平滑化する。トラッカー自身が使う生の pose 変数は
    そのままにして、FIT QUALITY 判定が平滑化のラグに引きずられないようにする。"""
    now = time.time()
    smoothed = np.eye(4, dtype=np.float32)
    smoothed[:3, 3] = pos_filter.filter(raw_pose[:3, 3], now)
    smoothed[:3, :3] = rot_filter.filter(raw_pose[:3, :3], now)
    return smoothed


# ─── M3T セットアップ（2アンカー） ────────────────────────────────────────

for _name, _path in [(ANCHOR_A_NAME, anchor_a_path), (ANCHOR_B_NAME, anchor_b_path),
                     (ANCHOR_C_NAME, anchor_c_path), (ANCHOR_D_NAME, anchor_d_path)]:
    if not os.path.exists(_path):
        raise FileNotFoundError(
            f"クラスタモデルが見つかりません: {_path}\n"
            "extract_cluster_mesh.py / segment_teeth.py でモデルを生成してから起動してください。"
        )

# contra 版は use_texture=True だが、クラスタOBJにはテクスチャ/色情報が
# 無いため、ここでは region（輪郭）のみを使う。
tracker = pym3t.Tracker(w, h, K, use_region=True, use_depth=False, use_texture=USE_TEXTURE_MODALITY)
if SINGLE_BODY_MODE and not os.path.exists(arch_model_path):
    raise FileNotFoundError(f"単体ボディモード用の結合モデルが見つかりません: {arch_model_path}")


def build_anchor(key, name, path, sphere_radius, sharp_deg, active_color, max_edges=500):
    """1アンカー分の pym3t.Model と、品質判定用/表示用のエッジ集合を作る。
    注: pym3t ラッパーは sphere_radius 引数を読み捨てる（RegionModel は既定 0.8 m で生成される）。"""
    m = pym3t.Model(name, path, "", "", MODEL_UNIT_IN_METER, sphere_radius)

    print(f"[{key}] OBJ エッジ読み込み中... ({name})")
    vertices, edges = load_obj_edges(path, max_edges=max_edges, sharp_angle_deg=sharp_deg)
    vertices = vertices * MODEL_UNIT_IN_METER
    used_ids = sorted(set(i for e in edges for i in e))
    vertex_map = {old: new for new, old in enumerate(used_ids)}
    draw_vertices = vertices[used_ids]
    draw_edges = [(vertex_map[a], vertex_map[b]) for a, b in edges]
    print(f"  [{key}] 品質判定用 頂点:{draw_vertices.shape} エッジ:{len(draw_edges)}")

    # 表示専用の密なエッジセット（人間が目視で位置合わせできるよう、角度に
    # 関係なく全エッジを対象にする。品質判定用のシャープエッジだけだと疎すぎる）。
    viz_vertices, viz_edges = load_obj_edges(path, max_edges=2500, sharp_angle_deg=0.0)
    viz_vertices = viz_vertices * MODEL_UNIT_IN_METER
    viz_used_ids = sorted(set(i for e in viz_edges for i in e))
    viz_vertex_map = {old: new for new, old in enumerate(viz_used_ids)}
    viz_draw_vertices = viz_vertices[viz_used_ids]
    viz_draw_edges = [(viz_vertex_map[a], viz_vertex_map[b]) for a, b in viz_edges]
    print(f"  [{key}] 表示用 頂点:{viz_draw_vertices.shape} エッジ:{len(viz_draw_edges)}")

    edge_a_idx = np.array([a for a, b in draw_edges], dtype=np.int32)
    edge_b_idx = np.array([b for a, b in draw_edges], dtype=np.int32)
    fill_idx = np.random.default_rng(0).choice(len(vertices), size=min(ALIGN_FILL_SAMPLES, len(vertices)),
                                               replace=False)
    fill_vertices = vertices[fill_idx]
    band_axis = np.array(ALIGN_BAND_AXIS, dtype=np.float32)
    along = fill_vertices @ band_axis
    band = along >= np.percentile(along, 100.0 * (1.0 - ALIGN_BAND_FRAC))
    outside_vertices = fill_vertices[band] + band_axis * ALIGN_BAND_OFFSET_M

    return {
        "key": key, "name": name, "path": path, "model": m,
        "centroid": draw_vertices.mean(axis=0).astype(np.float32),
        "draw_vertices": draw_vertices, "draw_edges": draw_edges,
        "fill_vertices": fill_vertices, "outside_vertices": outside_vertices,
        "edge_a_idx": edge_a_idx, "edge_b_idx": edge_b_idx,
        "viz_draw_vertices": viz_draw_vertices, "viz_draw_edges": viz_draw_edges,
        "all_vertices": vertices.astype(np.float32),   # 全頂点（キーフレーム特徴点をメッシュに結びつける z バッファ用）
        "active_color": active_color,
    }


if SINGLE_BODY_MODE:
    # 歯列全体を 1 ボディ（キー "F"）で追跡。階層アンカー有効時は 4 歯群（A〜D）も副アンカーとして追加。
    ANCHORS = [
        build_anchor("F", ARCH_MODEL_NAME, arch_model_path, ANCHOR_A_SPHERE_RADIUS, ARCH_SHARP_EDGE_DEG,
                     (255, 255, 255), max_edges=ARCH_QUALITY_MAX_EDGES),
    ]
    if HIERARCHICAL_ANCHORS:
        ANCHORS += [
            build_anchor("A", ANCHOR_A_NAME, anchor_a_path, ANCHOR_A_SPHERE_RADIUS, ANCHOR_A_SHARP_EDGE_DEG, ANCHOR_A_COLOR),
            build_anchor("B", ANCHOR_B_NAME, anchor_b_path, ANCHOR_B_SPHERE_RADIUS, ANCHOR_B_SHARP_EDGE_DEG, ANCHOR_B_COLOR),
            build_anchor("C", ANCHOR_C_NAME, anchor_c_path, ANCHOR_C_SPHERE_RADIUS, ANCHOR_C_SHARP_EDGE_DEG, ANCHOR_C_COLOR),
            build_anchor("D", ANCHOR_D_NAME, anchor_d_path, ANCHOR_D_SPHERE_RADIUS, ANCHOR_D_SHARP_EDGE_DEG, ANCHOR_D_COLOR),
        ]
        print("階層アンカーモード: 歯列全体 F を主アンカー、4 歯群 A〜D を副アンカーとして追跡します")
        # ⑲ 特徴点 PnP の点を歯群（アンカー）に振り分けるための表（F の頂点 → 最も近い歯群の頂点の歯群）
        #   頂点は 3 つおきに間引く（点の間隔 < 0.5 mm なので振り分けは変わらず、1 キーフレームあたり数 ms）
        _ANCHOR_LABEL_V = np.concatenate([a["all_vertices"][::3] for a in ANCHORS[1:]]).astype(np.float32)
        _ANCHOR_LABEL_K = np.concatenate([np.full(len(a["all_vertices"][::3]), a["key"]) for a in ANCHORS[1:]])
    else:
        print("単体ボディモード: 4 グループ結合モデルを 1 つの M3T ボディとして追跡します")
else:
    ANCHORS = [
        build_anchor("A", ANCHOR_A_NAME, anchor_a_path, ANCHOR_A_SPHERE_RADIUS, ANCHOR_A_SHARP_EDGE_DEG, ANCHOR_A_COLOR),
        build_anchor("B", ANCHOR_B_NAME, anchor_b_path, ANCHOR_B_SPHERE_RADIUS, ANCHOR_B_SHARP_EDGE_DEG, ANCHOR_B_COLOR),
        build_anchor("C", ANCHOR_C_NAME, anchor_c_path, ANCHOR_C_SPHERE_RADIUS, ANCHOR_C_SHARP_EDGE_DEG, ANCHOR_C_COLOR),
        build_anchor("D", ANCHOR_D_NAME, anchor_d_path, ANCHOR_D_SPHERE_RADIUS, ANCHOR_D_SHARP_EDGE_DEG, ANCHOR_D_COLOR),
    ]
ANCHOR_BY_KEY = {a["key"]: a for a in ANCHORS}

EDGE_SAMPLE_T = np.linspace(0.0, 1.0, 4, dtype=np.float32)


def _group_center_cam(key, pose_mat):
    """歯群 key の歯の重心（モデル座標）をカメラ座標に写した位置。"""
    c = ANCHOR_BY_KEY[key]["centroid"]
    return pose_mat[:3, :3] @ c + pose_mat[:3, 3]


def _depth_locked(c_cam, c_ref):
    """観測重心 c_cam を、その光線上で基準重心 c_ref と同じ奥行き（z）に置き直す。"""
    if not GROUP_DEPTH_FROM_REF or c_cam[2] <= 1e-6 or c_ref[2] <= 1e-6:
        return c_cam
    return c_cam * (c_ref[2] / c_cam[2])


def _group_pose_with_rotation(key, pose_group, ref_pose):
    """歯群の推定から歯の重心の画像上の位置だけを採り、回転（と GROUP_DEPTH_FROM_REF なら奥行き）は
    基準姿勢 ref_pose のものに差し替えた姿勢を返す。"""
    c = ANCHOR_BY_KEY[key]["centroid"]
    R_ref = ref_pose[:3, :3]
    c_cam = pose_group[:3, :3] @ c + pose_group[:3, 3]
    c_cam = _depth_locked(c_cam, R_ref @ c + ref_pose[:3, 3])
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = R_ref
    out[:3, 3] = c_cam - R_ref @ c
    return out


def _fuse_groups(keys, poses_by_key, ref_pose):
    """見えている歯群の重心位置から歯列全体の並進を決める（回転は ref_pose のまま）。
    各歯群ボディの推定姿勢で歯群重心をカメラ座標に写し、「その重心が ref_pose の回転のもとで来るべき並進」を
    歯群ごとに求めて重み付き平均する。回転を推定しないのは、歯群重心がモデル原点から約 30 mm 離れていて
    歯群ボディの回転誤差が 10 mm 級の重心誤差になり、3 点の Kabsch では回転が大きく外れるため（2026-10-05）。"""
    R = ref_pose[:3, :3].astype(np.float64)
    t_ref = ref_pose[:3, 3].astype(np.float64)
    t_sum, w_sum = np.zeros(3), 0.0
    for k in keys:
        c = np.asarray(ANCHOR_BY_KEY[k]["centroid"], np.float64)
        pk = poses_by_key[k]
        c_cam = pk[:3, :3] @ c + pk[:3, 3]
        c_cam = _depth_locked(c_cam, R @ c + t_ref)   # 奥行きは基準姿勢のまま（画像上の位置だけ歯群に従う）
        w = FUSION_GROUP_WEIGHT.get(k, 1.0)
        t_sum += w * (c_cam - R @ c)
        w_sum += w
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = R
    out[:3, 3] = t_sum / w_sum
    return out


def _rotate_pose_about_arch_center(pose_mat, axis, deg):
    """歯列全体の重心（カメラ座標）を中心に、カメラ座標の軸 axis まわりに deg 回した姿勢を返す。"""
    Rd = _rot_axis_matrix(axis, deg)
    R, t = pose_mat[:3, :3], pose_mat[:3, 3]
    c_cam = R @ ANCHOR_BY_KEY["F"]["centroid"] + t
    out = pose_mat.copy().astype(np.float32)
    out[:3, :3] = Rd @ R
    out[:3, 3] = Rd @ (t - c_cam) + c_cam
    return out


def return_rotation_search(frame_bgr, dist_map, base_pose):
    """統合解除 → F の瞬間に呼ぶ。base_pose（回転は F 離脱時のまま）の周りに回転候補を置き、各候補を
    M3T で RETURN_ROT_SEARCH_STEPS 回精密化して剛体整合スコアで比べる。無回転候補より
    RETURN_ROT_ACCEPT_MARGIN 以上良い候補があればそれを採用する。
    返り値: (採用姿勢, base からの回転量[deg], 無回転のスコア, 採用スコア, 所要秒)"""
    t0 = time.perf_counter()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    cands = [("none", 0.0, base_pose.copy().astype(np.float32))]
    for axis, degs in RETURN_ROT_SEARCH_DEGS.items():
        for d in degs:
            for s in (1.0, -1.0):
                cands.append((axis, s * d, _rotate_pose_about_arch_center(base_pose, axis, s * d)))
    results = []
    for axis, d, cand in cands:
        reset_all_anchors(cand)
        for _ in range(RETURN_ROT_SEARCH_STEPS):
            tracker.step(frame_rgb)
        refined = ANCHOR_BY_KEY["F"]["model"].pose_cv.copy().astype(np.float32)
        results.append((estimate_rigid_quality(dist_map, refined), axis, d, refined))
    base_q = results[0][0]
    best = max(results, key=lambda r: r[0])
    chosen = best if best[0] >= base_q + RETURN_ROT_ACCEPT_MARGIN else results[0]
    reset_all_anchors(chosen[3])
    delta = rotation_angle_deg(chosen[3], base_pose)
    dt = time.perf_counter() - t0
    top = sorted(results, key=lambda r: -r[0])[:4]
    print(f"復帰時回転再探索: 無回転 q={base_q:.2f} → 採用 {chosen[1]}{chosen[2]:+.0f}° q={chosen[0]:.2f} "
          f"(base からの回転 {delta:.1f}°, {dt*1000:.0f} ms)  上位: "
          + ", ".join(f"{a}{d:+.0f}°={q:.2f}" for q, a, d, _ in top))
    return chosen[3], delta, base_q, chosen[0], dt


def _slerp_R(Ra, Rb, alpha):
    """回転 Ra から Rb へ割合 alpha だけ進めた回転（軸角で補間）。"""
    rv, _ = cv2.Rodrigues(np.asarray(Rb, np.float64) @ np.asarray(Ra, np.float64).T)
    Rd, _ = cv2.Rodrigues(rv * float(alpha))
    return Rd @ np.asarray(Ra, np.float64)


def _rotate_pose_about_point(pose_mat, Rd, c_cam):
    """カメラ座標の点 c_cam を中心に、カメラ座標の回転 Rd を掛けた姿勢を返す。"""
    R, t = pose_mat[:3, :3], pose_mat[:3, 3]
    out = pose_mat.copy().astype(np.float32)
    out[:3, :3] = (Rd @ R).astype(np.float32)
    out[:3, 3] = (Rd @ (t - c_cam) + c_cam).astype(np.float32)
    return out


def return_search_begin(frame_bgr, dist_map, base_pose, occl_text):
    """① 復帰時回転再探索の分割版: 復帰時の画像と候補を控えるだけで、この場では評価しない。
    毎フレーム return_search_advance() が RETURN_ROT_SEARCH_PER_FRAME 候補ずつ評価する。"""
    cands = [("none", 0.0, base_pose.copy().astype(np.float32))]
    for axis, degs in RETURN_ROT_SEARCH_DEGS.items():
        for d in degs:
            for s in (1.0, -1.0):
                cands.append((axis, s * d, _rotate_pose_about_arch_center(base_pose, axis, s * d)))
    return {"frame_rgb": cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), "dist_map": dist_map,
            "base_pose": base_pose.copy().astype(np.float32), "cands": cands, "results": [],
            "t0": time.perf_counter(), "busy_ms": 0.0, "occl_text": occl_text}


def return_search_advance(job):
    """控えた画像に対して候補を数個評価する。ライブ追跡のボディ姿勢は呼ぶ前の状態に戻してから返す。
    全候補が終わったら (採用姿勢の base からの相対変換[モデル座標] or None, 採用回転量[deg], 無回転 q, 採用 q, 経過秒, 計算時間秒)
    を返し、未完なら None を返す。"""
    t0 = time.perf_counter()
    saved = {a["key"]: a["model"].pose_cv.copy() for a in ANCHORS}
    n_done = len(job["results"])
    for axis, d, cand in job["cands"][n_done:n_done + max(1, RETURN_ROT_SEARCH_PER_FRAME)]:
        reset_all_anchors(cand)
        for _ in range(RETURN_ROT_SEARCH_STEPS):
            tracker.step(job["frame_rgb"])
        refined = ANCHOR_BY_KEY["F"]["model"].pose_cv.copy().astype(np.float32)
        job["results"].append((estimate_rigid_quality(job["dist_map"], refined), axis, d, refined))
    for a in ANCHORS:
        a["model"].reset_pose_cv(saved[a["key"]])
    job["busy_ms"] += (time.perf_counter() - t0) * 1000.0
    if len(job["results"]) < len(job["cands"]):
        return None
    results = job["results"]
    base_q = results[0][0]
    best = max(results, key=lambda r: r[0])
    chosen = best if best[0] >= base_q + RETURN_ROT_ACCEPT_MARGIN else results[0]
    delta = rotation_angle_deg(chosen[3], job["base_pose"])
    elapsed = time.perf_counter() - job["t0"]
    top = sorted(results, key=lambda r: -r[0])[:4]
    print(f"復帰時回転再探索: 無回転 q={base_q:.2f} → 採用 {chosen[1]}{chosen[2]:+.0f}° q={chosen[0]:.2f} "
          f"(base からの回転 {delta:.1f}°, {len(results)} 候補を {elapsed*1000:.0f} ms に分割, 計算 {job['busy_ms']:.0f} ms)  上位: "
          + ", ".join(f"{a}{d:+.0f}°={q:.2f}" for q, a, d, _ in top))
    # 採用候補を「基準姿勢からの相対変換（モデル座標）」にして返す。評価中もライブ追跡は進んでいるので、
    # 候補姿勢そのものではなく、この相対変換を今の姿勢に掛けて適用する。
    rel = None
    if chosen is not results[0]:
        base = job["base_pose"].astype(np.float64)
        rel = (np.linalg.inv(base) @ chosen[3].astype(np.float64)).astype(np.float32)
    return rel, delta, base_q, chosen[0], elapsed, job["busy_ms"] / 1000.0


def reset_all_anchors(pose_mat):
    for a in ANCHORS:
        a["model"].reset_pose_cv(pose_mat)


# ─── クリック位置合わせ (solvePnP) 用ランドマーク点 ──────────────────────
#
# 手動での6DoF調整(a/d,q/e,w/s,i/k,j/l,u/o)が手間なので、実物でも視認・
# クリックしやすい点をモデル自身から自動抽出し、画像上で対応する点を
# クリックするだけで cv2.solvePnP に姿勢を計算させる。
#
# 咬頭（歯冠の最も高い点）だけだと、この3本クラスタでは実質4点しか取れず、
# しかもそのうち2点が7mm程度しか離れておらず、solvePnP(P3P/AP3P、最小構成)
# が不安定で反転解を選ぶ問題があった。各歯の左右の縁（クラウン外郭の端点）
# も候補に加えて点数を増やし、cv2.SOLVEPNP_ITERATIVE（過剰決定・最小二乗）
# で解けるようにすることで、1点あたりのクリック誤差の影響を減らし安定させる。
# 咬頭検出は segment_teeth.py の
#   build_adjacency（メッシュ隣接） + laplacian_smooth（局所ノイズ平滑化）
#   + find_crown_peaks（平滑化した高さの局所極大 = 歯冠候補）
# をそのまま流用（車輪の再発明をしない）。


def _connected_components(n, faces):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for f in faces:
        union(f[0], f[1])
        union(f[1], f[2])

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _nms_by_distance(candidate_ids, positions, priority, min_dist):
    """近接する候補をまとめ、各クラスタで優先度が最も高い1点だけ残す
    （距離ベースの Non-Maximum Suppression）。"""
    order = sorted(candidate_ids, key=lambda i: -priority[i])
    kept_ids, kept_pos = [], []
    for i in order:
        p = positions[i]
        if all(np.linalg.norm(p - kp) >= min_dist for kp in kept_pos):
            kept_ids.append(i)
            kept_pos.append(p)
    return kept_ids


def compute_landmark_points(path):
    from extract_cluster_mesh import load_obj, triangulate_faces
    from segment_teeth import build_adjacency, laplacian_smooth, find_crown_peaks

    raw_vertices, raw_faces, _ = load_obj(path)
    V = np.array(raw_vertices, dtype=np.float64)
    F = np.array(triangulate_faces(raw_faces), dtype=np.int64)
    neighbors = build_adjacency(len(V), F)
    # iterations=8 だと小臼歯の副咬頭が主咬頭に吸収されて3点(<4)になり
    # solvePnPが使えなくなることを実測で確認したため、3に抑える
    # （iterations=1〜5の範囲で安定して4点検出できることを確認済み）。
    Vs = laplacian_smooth(V, F, iterations=3)
    height = Vs[:, 1]  # このモデル群は一貫して Y が歯冠方向（他クラスタと同じ）
    region_set = set(range(len(V)))
    min_height = float(np.percentile(height, 50))
    peak_ids = list(find_crown_peaks(V, height, neighbors, region_set, min_height))

    # 各歯（連結成分）ごとに左右の縁の頂点も候補に加える。クリックのしやすさは
    # 咬頭より劣るが、咬頭同士が近すぎる場合の分散確保に効く。
    islands = _connected_components(len(V), F)
    corner_ids = []
    for island in islands:
        arr = np.array(island)
        corner_ids.append(int(arr[np.argmin(V[arr, 0])]))
        corner_ids.append(int(arr[np.argmax(V[arr, 0])]))

    candidate_ids = sorted(set(peak_ids) | set(corner_ids))
    if not candidate_ids:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=bool)

    # 優先度: 咬頭を常に縁より上位にして、近接時は咬頭を残す。
    priority = np.zeros(len(V))
    for i in peak_ids:
        priority[i] = 1000.0 + height[i]
    for i in corner_ids:
        if priority[i] == 0.0:
            priority[i] = height[i]

    span_x = V[:, 0].max() - V[:, 0].min()
    min_dist = max(span_x / 6.0, 1e-6)
    kept_ids = _nms_by_distance(candidate_ids, V, priority, min_dist)

    pts = V[np.array(kept_ids)] * MODEL_UNIT_IN_METER
    is_cusp = np.array([i in set(peak_ids) for i in kept_ids])
    order = np.argsort(pts[:, 0])
    return pts[order].astype(np.float32), is_cusp[order]


# 4アンカーはすべて同じ typodont 座標系を共有するため、クリック位置合わせは
# LANDMARK_ANCHOR_KEY で指定した1グループのランドマークだけで十分。
# 求まった pose を reset_all_anchors() で全モデルに同時にセットする。
_landmark_anchor_path_map = {
    "A": anchor_a_path, "B": anchor_b_path,
    "C": anchor_c_path, "D": anchor_d_path,
}
_landmark_src_path = _landmark_anchor_path_map[LANDMARK_ANCHOR_KEY]
_merged_pts, _merged_is_cusp = compute_landmark_points(_landmark_src_path)

if len(_merged_pts) > 0:
    _priority = np.where(_merged_is_cusp, 1000.0, 0.0) + _merged_pts[:, 1]
    _kept = _nms_by_distance(list(range(len(_merged_pts))), _merged_pts, _priority, min_dist=0.003)
    _order = np.argsort(_merged_pts[np.array(_kept)][:, 0])
    LANDMARK_POINTS_MODEL = _merged_pts[np.array(_kept)][_order].astype(np.float32)
    LANDMARK_IS_CUSP = _merged_is_cusp[np.array(_kept)][_order]
else:
    LANDMARK_POINTS_MODEL = np.zeros((0, 3), dtype=np.float32)
    LANDMARK_IS_CUSP = np.zeros(0, dtype=bool)

print(f"クリック位置合わせ用ランドマーク: {len(LANDMARK_POINTS_MODEL)} 点（アンカー {LANDMARK_ANCHOR_KEY} のみ使用）"
      f"（うち歯冠の尖った点 {int(LANDMARK_IS_CUSP.sum())} / 歯の縁 {int((~LANDMARK_IS_CUSP).sum())}）"
      + ("" if len(LANDMARK_POINTS_MODEL) >= 4 else "（4点未満のため 'c' キーでのPnP位置合わせは無効）"))


def _look_at(eye, target, up_hint):
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    right = np.cross(forward, up_hint)
    right = right / (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, forward)
    R = np.stack([right, -up, forward], axis=0)  # OpenCV convention: x右, y下, z前
    t = -R @ eye
    return R.astype(np.float64), t.astype(np.float64)


# ─── 参照ウィンドウ インタラクティブビュー ────────────────────────────────────
# 左ドラッグ: 視点回転（Blender風オービット）
# スクロール: ズームイン/アウト
# ダブルクリック: 初期視点にリセット

_ref_view = {
    "az": 0.0, "el": 10.0, "zoom": 1.0,
    "init_az": 0.0, "init_el": 10.0,
    "drag": False, "last_xy": None, "dirty": False,
}
_ref_mesh_cache = None


def _ensure_ref_mesh():
    """OBJを一度だけ読み込み、面データをキャッシュする。初期視点も設定する。"""
    global _ref_mesh_cache
    if _ref_mesh_cache is not None:
        return
    from extract_cluster_mesh import load_obj, triangulate_faces

    anchor_paths_colors = [
        (anchor_a_path, [225.0, 215.0, 200.0]),
        (anchor_b_path, [210.0, 225.0, 205.0]),
        (anchor_c_path, [210.0, 215.0, 225.0]),
        (anchor_d_path, [225.0, 225.0, 205.0]),
    ]
    all_V, all_F, all_C = [], [], []
    v_offset = 0
    for _p, _col in anchor_paths_colors:
        raw_v, raw_f, _ = load_obj(_p)
        V_i = np.array(raw_v, dtype=np.float64) * MODEL_UNIT_IN_METER
        F_i = np.array(triangulate_faces(raw_f), dtype=np.int64)
        if len(V_i) == 0 or len(F_i) == 0:
            continue
        all_V.append(V_i)
        all_F.append(F_i + v_offset)
        all_C.append(np.tile(np.array(_col), (len(F_i), 1)))
        v_offset += len(V_i)

    V = np.concatenate(all_V, axis=0) if all_V else np.zeros((1, 3), dtype=np.float64)
    F = np.concatenate(all_F, axis=0) if all_F else np.zeros((0, 3), dtype=np.int64)
    face_base_color = np.concatenate(all_C, axis=0) if all_C else np.zeros((0, 3))

    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    face_normal = np.cross(p1 - p0, p2 - p0)
    face_area2 = np.linalg.norm(face_normal, axis=1, keepdims=True)
    face_normal_unit = face_normal / (face_area2 + 1e-12)
    face_centroid = (p0 + p1 + p2) / 3.0
    centroid = V.mean(axis=0)
    radius = float(np.linalg.norm(V - centroid, axis=1).max()) if len(V) > 1 else 1.0

    # 水平方向の正面を面積重み平均法線から求め、初期アジマスを設定する
    mean_normal = (face_normal_unit * face_area2).sum(axis=0)
    mean_normal[1] = 0.0
    horiz_norm = np.linalg.norm(mean_normal)
    if horiz_norm < 1e-6:
        mean_normal = np.array([0.0, 0.0, 1.0])
    else:
        mean_normal /= horiz_norm
    init_az = float(np.degrees(np.arctan2(mean_normal[0], mean_normal[2])))
    init_el = float(np.degrees(np.arctan2(0.6, 3.5)))  # ≈ 9.7° 見下ろし角

    _ref_mesh_cache = {
        "V": V, "F": F, "face_base_color": face_base_color,
        "face_normal_unit": face_normal_unit,
        "face_centroid": face_centroid,
        "centroid": centroid, "radius": radius,
    }
    _ref_view["az"] = _ref_view["init_az"] = init_az
    _ref_view["el"] = _ref_view["init_el"] = init_el


def _render_reference(width=680, height=560):
    """現在の _ref_view 角度でモデルをレンダリングして返す。"""
    _ensure_ref_mesh()
    m = _ref_mesh_cache
    V, F = m["V"], m["F"]
    face_base_color = m["face_base_color"]
    face_normal_unit = m["face_normal_unit"]
    face_centroid = m["face_centroid"]
    centroid, radius = m["centroid"], m["radius"]

    az_rad = np.deg2rad(_ref_view["az"])
    el_rad = np.deg2rad(_ref_view["el"])
    dir_x = np.cos(el_rad) * np.sin(az_rad)
    dir_y = np.sin(el_rad)
    dir_z = np.cos(el_rad) * np.cos(az_rad)
    eye_dist = radius * 3.5 / max(_ref_view["zoom"], 0.01)
    eye = centroid + np.array([dir_x, dir_y, dir_z]) * eye_dist

    if abs(_ref_view["el"]) > 80.0:
        up_hint = np.array([np.sin(az_rad + np.pi / 2), 0.0, np.cos(az_rad + np.pi / 2)])
    else:
        up_hint = np.array([0.0, 1.0, 0.0])

    R, t = _look_at(eye, centroid, up_hint)

    def project_all(pts):
        cam = (R @ pts.T).T + t
        return cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2], cam[:, 2]

    fx_raw, fy_raw, _ = project_all(V)
    margin = 50
    x_min, x_max = float(fx_raw.min()), float(fx_raw.max())
    y_min, y_max = float(fy_raw.min()), float(fy_raw.max())
    scale = min((width - 2 * margin) / max(x_max - x_min, 1e-6),
                (height - 2 * margin) / max(y_max - y_min, 1e-6))

    def to_canvas_xy(px, py):
        return margin + (px - x_min) * scale, margin + (py - y_min) * scale

    face_cam = (R @ face_centroid.T).T + t
    face_depth = face_cam[:, 2]
    light_dir = np.array([0.3, -0.5, -1.0])
    light_dir /= np.linalg.norm(light_dir)
    normal_cam = (R @ face_normal_unit.T).T
    shade = np.clip(-normal_cam @ light_dir, 0.0, 1.0)
    shade = 0.35 + 0.65 * shade

    fx_v, fy_v, _ = project_all(V)
    vx, vy = to_canvas_xy(fx_v, fy_v)
    tri2d = np.stack([vx, vy], axis=1)

    order = np.argsort(-face_depth)
    img = np.full((height, width, 3), 25, dtype=np.uint8)
    for fi in order:
        tri = tri2d[F[fi]].astype(np.int32)
        color = tuple(int(c) for c in (face_base_color[fi] * shade[fi]))
        cv2.fillConvexPoly(img, tri, color, lineType=cv2.LINE_AA)

    for anchor in ANCHORS:
        for a, b in anchor["draw_edges"]:
            va = anchor["draw_vertices"][a:a + 1]
            vb = anchor["draw_vertices"][b:b + 1]
            pa = to_canvas_xy(*project_all(va)[:2])
            pb = to_canvas_xy(*project_all(vb)[:2])
            pa = (int(pa[0][0]), int(pa[1][0]))
            pb = (int(pb[0][0]), int(pb[1][0]))
            cv2.line(img, pa, pb, (90, 70, 70), 1, cv2.LINE_AA)

    for i, p in enumerate(LANDMARK_POINTS_MODEL):
        lx, ly, _ = project_all(p.reshape(1, 3).astype(np.float64))
        cx_, cy_ = to_canvas_xy(lx, ly)
        cx_, cy_ = int(cx_[0]), int(cy_[0])
        color = (0, 255, 255) if LANDMARK_IS_CUSP[i] else (255, 255, 0)
        cv2.circle(img, (cx_, cy_), 7, (0, 0, 0), -1)
        cv2.circle(img, (cx_, cy_), 6, color, -1)
        cv2.putText(img, str(i + 1), (cx_ + 9, cy_ - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(img,
                f"REFERENCE  drag=rotate  scroll=zoom  dblclick=reset"
                f"  (az={_ref_view['az']:.0f} el={_ref_view['el']:.0f})",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
    cv2.putText(img, "yellow=cusp(sharp)  cyan=edge(approx ok)",
                (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return img


def on_mouse_reference(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _ref_view["drag"] = True
        _ref_view["last_xy"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        _ref_view["drag"] = False
    elif event == cv2.EVENT_MOUSEMOVE and _ref_view["drag"]:
        lx, ly = _ref_view["last_xy"]
        dx, dy = x - lx, y - ly
        _ref_view["az"] = (_ref_view["az"] + dx * 0.5) % 360.0
        _ref_view["el"] = float(np.clip(_ref_view["el"] - dy * 0.5, -89.0, 89.0))
        _ref_view["last_xy"] = (x, y)
        _ref_view["dirty"] = True
    elif event == cv2.EVENT_MOUSEWHEEL:
        step = 1.1 if flags > 0 else 1.0 / 1.1
        _ref_view["zoom"] = float(np.clip(_ref_view["zoom"] * step, 0.2, 8.0))
        _ref_view["dirty"] = True
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        _ref_view["az"] = _ref_view["init_az"]
        _ref_view["el"] = _ref_view["init_el"]
        _ref_view["zoom"] = 1.0
        _ref_view["dirty"] = True


_ensure_ref_mesh()

reset_all_anchors(np.eye(4, dtype=np.float32))  # 後で load_pose_slot の結果に差し替える
for a in ANCHORS:
    tracker.add_model(a["model"])
viewer = pym3t.Viewer(tracker)


# ─── 姿勢スロット ─────────────────────────────────────────────────────────

os.makedirs(pose_slots_dir, exist_ok=True)


def slot_pose_path(slot):
    return os.path.join(pose_slots_dir, f"initial_pose_{slot}.npy")


def load_pose_slot(slot):
    path = slot_pose_path(slot)
    if os.path.exists(path):
        return np.load(path).astype(np.float32), path
    return None, path


def save_pose_slot(slot, pose_mat):
    path = slot_pose_path(slot)
    np.save(path, pose_mat.astype(np.float32))
    return path


def pose_angle_diff_deg(Ra, Rb):
    qa, qb = rotmat_to_quat(Ra), rotmat_to_quat(Rb)
    dot = np.clip(abs(float(np.dot(qa, qb))), -1.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(dot)))


def is_pose_novel_for_capture(candidate_pose):
    """スロット1（クリック初期化用）と自動登録スロットの既存姿勢と比較し、
    回転・並進ともに十分違う場合だけ新規登録の対象にする（似た姿勢で
    スロットを埋め尽くさないようにする）。"""
    for slot in [1] + AUTO_CAPTURE_SLOTS:
        existing, _ = load_pose_slot(slot)
        if existing is None:
            continue
        angle_diff = pose_angle_diff_deg(candidate_pose[:3, :3], existing[:3, :3])
        trans_diff = float(np.linalg.norm(candidate_pose[:3, 3] - existing[:3, 3]))
        if angle_diff < AUTO_CAPTURE_MIN_ANGLE_DEG and trans_diff < AUTO_CAPTURE_MIN_TRANSLATION_M:
            return False
    return True


selected_pose_slot = 1
pose, loaded_pose_path = load_pose_slot(selected_pose_slot)
if pose is not None:
    print(f"初期姿勢スロット {selected_pose_slot} を読み込み: {loaded_pose_path}")
else:
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = 0.4
reset_all_anchors(pose)


# ─── 追跡品質評価・自動姿勢選択 ──────────────────────────────────────────

CANNY_AUTO_SIGMA = 0.33
EDGE_ORIENT_BINS = 8   # 勾配方向（0〜180°）の量子化数。22.5° 刻み、±1 ビン許容で約 ±34°
EDGE_DIST_NO_EDGE = 1.0e4  # あるビンにエッジが 1 つも無いときの距離


def compute_edge_distance_map(bgr):
    """Canny エッジまでの距離変換。shape = (1 + EDGE_ORIENT_BINS, H, W)
    [0]   : 向きを問わない従来の距離マップ
    [1+b] : 画像勾配の向きがビン b のエッジだけで作った距離マップ
    歯列のようにエッジが密な領域では「近くに何かエッジがある」だけなら
    どこに置いても当たってしまうため、投影エッジの向きと一致するものだけを
    ヒットと数えられるように向き別のマップも持つ。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if QUALITY_SCALE < 1.0:
        gray = cv2.resize(gray, None, fx=QUALITY_SCALE, fy=QUALITY_SCALE, interpolation=cv2.INTER_AREA)
    # 固定しきい値(50,150)だと、消灯時など画像全体が暗く勾配が弱いシーンで
    # エッジがほとんど検出されず、姿勢が合っていても品質スコアが不当に低く
    # 出てしまう。画像の中央値ベースでしきい値を自動決定し、明るさに
    # 依存しないエッジ密度にする（いわゆる auto-canny）。
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - CANNY_AUTO_SIGMA) * median))
    upper = int(min(255, (1.0 + CANNY_AUTO_SIGMA) * median))
    canny = cv2.Canny(gray, lower, upper)

    maps = np.empty((1 + EDGE_ORIENT_BINS,) + gray.shape, dtype=np.float32)
    maps[0] = cv2.distanceTransform(255 - canny, cv2.DIST_L2, 3)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    bins = np.round(ang / (np.pi / EDGE_ORIENT_BINS)).astype(np.int32) % EDGE_ORIENT_BINS
    edge = canny > 0
    for b in range(EDGE_ORIENT_BINS):
        sel = edge & (bins == b)
        if not sel.any():
            maps[1 + b] = EDGE_DIST_NO_EDGE
            continue
        mask = np.where(sel, 0, 255).astype(np.uint8)
        maps[1 + b] = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    return maps


def project_vertices(verts, candidate_pose):
    R = candidate_pose[:3, :3]
    t = candidate_pose[:3, 3]
    pts_cam = (R @ verts.T).T + t
    valid = pts_cam[:, 2] > 0.05
    x = K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]
    y = K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]
    pts2d = np.stack([x, y], axis=1)
    finite = np.isfinite(pts2d).all(axis=1)
    return pts2d, valid & finite


def estimate_tracking_quality_for(dist_map, candidate_pose, draw_verts, edge_a_idx, edge_b_idx,
                                   max_dist=QUALITY_EDGE_MAX_DIST_PX, count_out_of_bounds=False, soft=False):
    """投影したワイヤーフレームのうち、実画像のエッジ付近に乗っている割合。
    pym3t は残差/信頼度を返さないため、これが唯一のフィット判定手段になる。
    エッジごとサンプル数を固定してベクトル化し、候補が多くても軽い。
    Anchor Switching のため、どのアンカーの頂点/エッジ集合で評価するかを
    引数で切り替えられるようにしている。"""
    pts2d, valid = project_vertices(draw_verts, candidate_pose)
    if QUALITY_SCALE < 1.0:
        pts2d = pts2d * QUALITY_SCALE   # 距離マップは縮小画像上にあるので投影座標も合わせる
    a2d, b2d = pts2d[edge_a_idx], pts2d[edge_b_idx]
    edge_valid = valid[edge_a_idx] & valid[edge_b_idx]

    t = EDGE_SAMPLE_T[None, :, None]
    samples = a2d[:, None, :] * (1.0 - t) + b2d[:, None, :] * t

    hgt, wid = dist_map.shape[1:3]
    xi = np.round(samples[..., 0]).astype(np.int32)
    yi = np.round(samples[..., 1]).astype(np.int32)
    in_bounds = (xi >= 0) & (xi < wid) & (yi >= 0) & (yi < hgt) & edge_valid[:, None]
    xi_c, yi_c = np.clip(xi, 0, wid - 1), np.clip(yi, 0, hgt - 1)

    if QUALITY_ORIENTED_EDGES:
        # 投影エッジの法線方向 = そこにあるべき画像勾配の方向。その向き（±1ビン）の
        # エッジまでの距離だけを見る。
        d = b2d - a2d
        normal_ang = np.mod(np.arctan2(d[:, 1], d[:, 0]) + np.pi / 2.0, np.pi)
        nb = np.round(normal_ang / (np.pi / EDGE_ORIENT_BINS)).astype(np.int32) % EDGE_ORIENT_BINS
        dist = np.full(xi_c.shape, np.inf, dtype=np.float32)
        for off in (-1, 0, 1):
            layer = 1 + ((nb + off) % EDGE_ORIENT_BINS)
            dist = np.minimum(dist, dist_map[layer[:, None], yi_c, xi_c])
    else:
        dist = dist_map[0, yi_c, xi_c]
    hits = in_bounds & (dist <= max_dist)

    # count_out_of_bounds=True（並進探索用）: 画面外に出たサンプルも「外れ」として分母に入れる。
    # 既定の「画面内サンプルだけで割る」だと、物体の大半が画面外に出た候補が残り少数の
    # サンプルの偶然の一致で 1.0 になり、探索でそれが最上位に来てしまう。
    total = int(np.count_nonzero(np.broadcast_to(edge_valid[:, None], in_bounds.shape)))         if count_out_of_bounds else int(np.count_nonzero(in_bounds))
    if total <= 0:
        return 0.0
    if soft:
        # 距離に応じた連続スコア（max_dist で 0 に落ちる線形減衰）。粗い格子探索で
        # 「近い候補ほど高い」という勾配を持たせるため。ヒット/ミスの 2 値だと格子
        # 間隔（10 mm）に対して許容 4 px（約 1 mm）が狭すぎて真値近傍の候補が拾えない。
        soft_hit = np.where(in_bounds, np.clip(1.0 - dist / max_dist, 0.0, 1.0), 0.0)
        return float(soft_hit.sum()) / total
    return float(np.count_nonzero(hits)) / total


def estimate_anchor_quality(dist_map, candidate_pose, anchor, count_out_of_bounds=False):
    return estimate_tracking_quality_for(
        dist_map, candidate_pose, anchor["draw_vertices"], anchor["edge_a_idx"], anchor["edge_b_idx"],
        count_out_of_bounds=count_out_of_bounds)


def _motion_gray(bgr):
    """回転追従の動き検出用の縮小グレー画像（品質評価と同じ縮尺）。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if QUALITY_SCALE < 1.0:
        gray = cv2.resize(gray, None, fx=QUALITY_SCALE, fy=QUALITY_SCALE, interpolation=cv2.INTER_AREA)
    return gray


def motion_ref_begin(gray, pose, key):
    """歯群 key の投影範囲（外接矩形）から特徴点を取り、動き検出の基準にする。取れなければ None。"""
    pts2d, valid = project_vertices(ANCHOR_BY_KEY[key]["draw_vertices"], pose)
    pts2d = pts2d[valid] * QUALITY_SCALE
    if len(pts2d) < 10:
        return None
    hgt, wid = gray.shape
    x0, y0 = np.clip(np.floor(pts2d.min(axis=0)).astype(int), 0, [wid - 1, hgt - 1])
    x1, y1 = np.clip(np.ceil(pts2d.max(axis=0)).astype(int) + 1, 0, [wid, hgt])
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    mask = np.zeros_like(gray)
    mask[y0:y1, x0:x1] = 255
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=60, qualityLevel=0.01, minDistance=4, mask=mask)
    if pts is None or len(pts) < FUSION_ROT_MOTION_MIN_POINTS:
        return None
    return {"gray": gray, "pts": pts}


def motion_from_ref(ref, gray):
    """基準フレームからの特徴点移動量の中央値 [px, 縮小画像]。追えた点が少なければ None。"""
    p1, st, _ = cv2.calcOpticalFlowPyrLK(ref["gray"], gray, ref["pts"], None, winSize=(15, 15), maxLevel=3)
    if p1 is None:
        return None
    ok = st.ravel() == 1
    if int(ok.sum()) < FUSION_ROT_MOTION_MIN_POINTS:
        return None
    return float(np.median(np.linalg.norm((p1 - ref["pts"]).reshape(-1, 2)[ok], axis=1)))


def _pnp_gray(bgr):
    """特徴点 PnP 用のグレー画像（FUSION_PNP_SCALE の大きさ）。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if FUSION_PNP_SCALE < 1.0:
        gray = cv2.resize(gray, None, fx=FUSION_PNP_SCALE, fy=FUSION_PNP_SCALE, interpolation=cv2.INTER_AREA)
    return gray


def _pnp_project(Xc, Ks):
    return (Xc[:, :2] / Xc[:, 2:3]) * np.array([Ks[0, 0], Ks[1, 1]]) + np.array([Ks[0, 2], Ks[1, 2]])


def _pnp_patches(gray, pts):
    """各点まわりの (2h+1)² 小画像を平均 0・ノルム 1 にして返す（NCC 用）。shape = (N, (2h+1)²)。"""
    size = 2 * int(round(_pnp_px(FUSION_PNP_NCC_HALF))) + 1
    out = np.zeros((len(pts), size * size), np.float32)
    for i, (x, y) in enumerate(pts):
        p = cv2.getRectSubPix(gray, (size, size), (float(x), float(y))).astype(np.float32).ravel()
        p -= p.mean()
        n = float(np.linalg.norm(p))
        out[i] = p / n if n > 1e-6 else 0.0
    return out


def _pnp_Ks():
    Ks = K.astype(np.float64).copy()
    Ks[:2] *= FUSION_PNP_SCALE
    return Ks


def _pnp_detect(gray, R, t, max_features, extra_mask=None):
    """姿勢 (R, t) で F を投影した範囲（extra_mask があれば重ねる）から特徴点を取り、z バッファで最前面の
    メッシュ頂点に結び付ける。戻り値: (点 [N,2] 縮小画像 px, 頂点 [N,3] モデル座標)。"""
    Ks = _pnp_Ks()
    V = ANCHOR_BY_KEY["F"]["all_vertices"].astype(np.float64)
    Xc = V @ R.T + t
    front = Xc[:, 2] > 0.05
    uv = _pnp_project(Xc[front], Ks)
    vid = np.where(front)[0]
    z = Xc[front, 2]
    hgt, wid = gray.shape
    ui, vi = np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64)
    ok = (ui >= 0) & (ui < wid) & (vi >= 0) & (vi < hgt)
    order = np.argsort(-z[ok])                         # 遠い順に書き、近い頂点が上書きする（z バッファ）
    sel = np.where(ok)[0][order]
    zb = np.full((hgt, wid), np.inf, dtype=np.float32)
    idb = np.full((hgt, wid), -1, dtype=np.int64)
    zb[vi[sel], ui[sel]] = z[sel]
    idb[vi[sel], ui[sel]] = vid[sel]
    sil = np.isfinite(zb).astype(np.uint8) * 255
    # 元の解像度では頂点の投影が画素より疎になる（45k 頂点）ので、カーネル・探索範囲も倍率に合わせて広げる
    k_close, k_erode = _pnp_odd(5), _pnp_odd(3)
    sil = cv2.erode(cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8)),
                    np.ones((k_erode, k_erode), np.uint8))
    # 隠れた頂点がたまたま画素を持つことがあるので、近傍の最前面より 4 mm 以上奥の頂点は使わない
    zfront = cv2.erode(np.where(np.isinf(zb), 1e9, zb).astype(np.float32), np.ones((k_close, k_close), np.uint8))
    if extra_mask is not None:
        sil = cv2.bitwise_and(sil, extra_mask)
    empty = (np.zeros((0, 2), np.float32), np.zeros((0, 3)))
    if max_features <= 0:
        return empty
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=int(max_features), qualityLevel=0.001, minDistance=3, mask=sil)
    if pts is None:
        return empty
    pts = pts.reshape(-1, 2)
    keep_pts, keep_ids = [], []
    rad = int(round(_pnp_px(2)))
    for (x, y) in pts:
        xi, yi = int(round(x)), int(round(y))
        y0, y1, x0, x1 = max(0, yi - rad), min(hgt, yi + rad + 1), max(0, xi - rad), min(wid, xi + rad + 1)
        win_id = idb[y0:y1, x0:x1]
        valid = (win_id >= 0) & (zb[y0:y1, x0:x1] <= zfront[y0:y1, x0:x1] + 0.004)
        if not valid.any():
            continue
        yy, xx = np.nonzero(valid)
        k = int(np.argmin((yy + y0 - yi) ** 2 + (xx + x0 - xi) ** 2))
        keep_pts.append((x, y))
        keep_ids.append(int(win_id[yy[k], xx[k]]))
    if not keep_ids:
        return empty
    return np.array(keep_pts, np.float32), V[keep_ids]


_anchor_flann = None


def _anchor_labels(X):
    """モデル座標の点 X（F の頂点）→ それぞれが乗っている歯群（アンカー）のキー。歯群を読んでいなければ空文字。
    最近傍は k-d 木（OpenCV FLANN）で引く。総当たりだと F の見張りがキーフレームを作るたびに約 0.2 s 止まった（2026-10-21）。"""
    global _anchor_flann
    if not FUSION_PNP_ANCHORS or "_ANCHOR_LABEL_V" not in globals() or len(X) == 0:
        return np.full(len(X), "", dtype="<U1")
    if _anchor_flann is None:
        _anchor_flann = cv2.flann_Index(_ANCHOR_LABEL_V, {"algorithm": 1, "trees": 4})   # 1 = KDTREE
    idx, _ = _anchor_flann.knnSearch(np.ascontiguousarray(X, np.float32), 1, params={"checks": 32})
    return _ANCHOR_LABEL_K[idx.ravel()]


def pnp_keyframe_begin(gray, pose, kf_time=None, with_anchors=True):
    """遮蔽直前の F 良好フレーム（縮小グレー）と姿勢から、メッシュに結び付いた特徴点の集合を作る。取れなければ None。"""
    R, t = pose[:3, :3].astype(np.float64), pose[:3, 3].astype(np.float64)
    pts, X = _pnp_detect(gray, R, t, FUSION_PNP_MAX_POINTS)
    if len(pts) < FUSION_PNP_MIN_POINTS:
        return None
    return {"Ks": _pnp_Ks(), "X": X, "cur": pts.copy(), "kf_pts": pts.copy(),
            "alive": np.ones(len(pts), bool), "gray": gray, "kf_patches": _pnp_patches(gray, pts),
            "R0": R.copy(), "t0": t.copy(), "R": R.copy(), "t": t.copy(), "n0": len(pts), "refreshed": 0,
            # 診断用（M3T_PNP_DUMP）: 各点を結び付けた姿勢の番号と、その姿勢・時刻の一覧
            "born": np.zeros(len(pts), np.int32), "born_poses": [(kf_time, R.copy(), t.copy())],
            "born_uv": pts.copy(), "anchor": _anchor_labels(X) if with_anchors else np.full(len(X), "", dtype="<U1")}


def _pnp_refresh(st, gray, tooth_mask):
    """点が減ったら、今の PnP 姿勢で歯の上（歯色マスク）から点を取り直して足す（Vacchetti らのオンライン
    キーフレームの簡易版）。既存の生存点の近くは避ける。新しい点の kf_pts（静止判定の基準）はキーフレーム姿勢での投影。"""
    idx = np.where(st["alive"])[0]
    want = FUSION_PNP_MAX_POINTS - len(idx)
    mask = tooth_mask.copy()
    rad = int(round(_pnp_px(4)))
    for (x, y) in st["cur"][idx]:
        cv2.circle(mask, (int(round(x)), int(round(y))), rad, 0, -1)
    pts, X = _pnp_detect(gray, st["R"], st["t"], want, extra_mask=mask)
    if len(pts) == 0:
        return 0
    kf_proj = _pnp_project(X @ st["R0"].T + st["t0"], st["Ks"]).astype(np.float32)
    st["X"] = np.vstack([st["X"], X])
    st["cur"] = np.vstack([st["cur"], pts])
    st["kf_pts"] = np.vstack([st["kf_pts"], kf_proj])
    st["alive"] = np.concatenate([st["alive"], np.ones(len(pts), bool)])
    st["kf_patches"] = np.vstack([st["kf_patches"], _pnp_patches(gray, pts)])
    st["refreshed"] += 1
    st["born_poses"].append((time.time(), st["R"].copy(), st["t"].copy()))
    st["born"] = np.concatenate([st["born"], np.full(len(pts), len(st["born_poses"]) - 1, np.int32)])
    st["born_uv"] = np.vstack([st["born_uv"], pts])
    st["anchor"] = np.concatenate([st["anchor"], _anchor_labels(X)])
    return len(pts)


_PNP_EDGE_CACHE = {}


def _pnp_edge_set(key):
    """歯群 key の稜線（品質評価と同じ draw_vertices / edges）から、端点 A, B と 1 本 3 点の標本点 S を返す（キャッシュ）。"""
    if key not in _PNP_EDGE_CACHE:
        a = ANCHOR_BY_KEY[key]
        V = a["draw_vertices"].astype(np.float64)
        ia, ib = a["edge_a_idx"], a["edge_b_idx"]
        sel = np.random.default_rng(0).choice(len(ia), size=min(len(ia), FUSION_PNP_EDGE_MAX_PER_GROUP), replace=False)
        A, B = V[ia[sel]], V[ib[sel]]
        tt = np.array([0.25, 0.5, 0.75])
        S = (A[:, None, :] * (1.0 - tt)[None, :, None] + B[:, None, :] * tt[None, :, None]).reshape(-1, 3)
        _PNP_EDGE_CACHE[key] = (A, B, S)
    return _PNP_EDGE_CACHE[key]


def _pnp_edge_layers(edge_sets, R, t, Ks):
    """各標本点で引く向き付き距離マップの層（±1 ビン、品質評価と同じ）。反復ごとに今の姿勢で決めて数値微分中は固定する。"""
    layers = []
    for (A, B, S) in edge_sets:
        d = _pnp_project(B @ R.T + t, Ks) - _pnp_project(A @ R.T + t, Ks)
        nang = np.mod(np.arctan2(d[:, 1], d[:, 0]) + np.pi / 2.0, np.pi)
        nb = np.repeat(np.round(nang / (np.pi / EDGE_ORIENT_BINS)).astype(np.int64) % EDGE_ORIENT_BINS, 3)
        layers.append(np.stack([1 + ((nb + off) % EDGE_ORIENT_BINS) for off in (-1, 0, 1)], axis=1))
    return layers


def _pnp_edge_residual(dist_map, edge_sets, layers, R, t, Ks):
    """稜線の標本点から画像のエッジまでの向き付き距離（双線形補間、FUSION_PNP_EDGE_MAX_PX で打ち切り）。"""
    hgt, wid = dist_map.shape[1:3]
    out = []
    for (A, B, S), lay in zip(edge_sets, layers):
        uv = _pnp_project(S @ R.T + t, Ks)
        x, y = uv[:, 0], uv[:, 1]
        inb = (x >= 0) & (x < wid - 1) & (y >= 0) & (y < hgt - 1)
        x0 = np.clip(np.floor(x).astype(np.int64), 0, wid - 2)
        y0 = np.clip(np.floor(y).astype(np.int64), 0, hgt - 2)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        v = (dist_map[lay, y0[:, None], x0[:, None]] * (1 - fx) * (1 - fy)
             + dist_map[lay, y0[:, None], x0[:, None] + 1] * fx * (1 - fy)
             + dist_map[lay, y0[:, None] + 1, x0[:, None]] * (1 - fx) * fy
             + dist_map[lay, y0[:, None] + 1, x0[:, None] + 1] * fx * fy)
        dmin = np.minimum(v.min(axis=1), FUSION_PNP_EDGE_MAX_PX)
        out.append(np.where(inb, dmin, FUSION_PNP_EDGE_MAX_PX))
    return np.concatenate(out) if out else np.zeros(0)


def _pnp_solve(st, X, u, c_prior, iters=8, huber=None, R_init=None, t_init=None, edge_ctx=None):
    """回転＋並進を、再投影誤差（Huber）＋ F 重心の事前分布（＋ edge_ctx があれば見えている歯の輪郭）で解く。
    edge_ctx = (向き付きエッジ距離マップ, 歯群キーのリスト)。戻り値: (R, t, 残差 px, 回転 1σ [deg])。"""
    if huber is None:
        huber = _pnp_px(1.0)
    if FUSION_PNP_SCALE != QUALITY_SCALE:
        edge_ctx = None                       # 輪郭の距離マップは品質評価の縮尺なので、倍率が違うときは使わない
    Ks = st["Ks"]
    cF = ANCHOR_BY_KEY["F"]["centroid"].astype(np.float64)
    w_prior = 1.0 / FUSION_PNP_PRIOR_SIGMA_M      # 重心 1σ のずれ ≒ 再投影 1 px と同じ重み
    R = (st["R"] if R_init is None else R_init).copy()
    t = (st["t"] if t_init is None else t_init).copy()
    edge_sets, dist_map, w_edge = [], None, 0.0
    if edge_ctx is not None and FUSION_PNP_USE_EDGES and edge_ctx[0] is not None and edge_ctx[1]:
        dist_map = edge_ctx[0]
        edge_sets = [_pnp_edge_set(k) for k in edge_ctx[1] if k in ANCHOR_BY_KEY]
        n_edge = sum(len(s[2]) for s in edge_sets)
        # 輪郭の二乗和 ≈ 特徴点の二乗和 × FUSION_PNP_EDGE_REL になる重み（特徴点は 1 点 2 成分）
        w_edge = FUSION_PNP_EDGE_REL * np.sqrt(2.0 * max(len(X), 10) / max(n_edge, 1))
    layers = None

    def residual(Rm, tv):
        rp = (_pnp_project(X @ Rm.T + tv, Ks) - u).ravel()
        rc = ((Rm @ cF + tv) - c_prior) * w_prior if c_prior is not None else np.zeros(0)
        re = _pnp_edge_residual(dist_map, edge_sets, layers, Rm, tv, Ks) * w_edge if edge_sets else np.zeros(0)
        return rp, np.concatenate([rc, re])

    J = None
    axis = FUSION_PNP_AXIS                     # ⑭ 回転を模型座標の 1 軸まわりに制限（None なら 3 軸）
    n_rot = 1 if axis is not None else 3
    n_dof = n_rot + 3
    for _ in range(iters):
        if edge_sets:
            layers = _pnp_edge_layers(edge_sets, R, t, Ks)
        rp, rc = residual(R, t)
        r = np.concatenate([rp, rc])
        J = np.zeros((len(r), n_dof))
        eps = 1e-6
        for k in range(n_dof):
            d = np.zeros(n_dof)
            d[k] = eps
            Rk = (R @ cv2.Rodrigues(axis * d[0])[0]) if axis is not None else (cv2.Rodrigues(d[:3])[0] @ R)
            rpk, rck = residual(Rk, t + d[n_rot:])
            J[:, k] = (np.concatenate([rpk, rck]) - r) / eps
        a = np.abs(r[:len(rp)])
        wts = np.concatenate([np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-9)), np.ones(len(rc))])
        JW = J * wts[:, None]
        dx = -np.linalg.solve(JW.T @ J + 1e-12 * np.eye(n_dof), JW.T @ r)
        R = (R @ cv2.Rodrigues(axis * dx[0])[0]) if axis is not None else (cv2.Rodrigues(dx[:3])[0] @ R)
        t = t + dx[n_rot:]
        if np.linalg.norm(dx[:n_rot]) < 1e-7 and np.linalg.norm(dx[n_rot:]) < 1e-7:
            break
    res = np.linalg.norm(_pnp_project(X @ R.T + t, Ks) - u, axis=1)
    # 2 次元誤差の大きさ |e| の中央値は σ√(2 ln 2) = 1.177σ（レイリー分布）。以前は 1 次元用の 0.45 で割っていて
    # σ を約 1.75 倍に見積もり、正しい回転（11:35 run 78〜90 s、推定 3.6/8.0/19.5° vs 真値 3.0/8.9/19.3°）を捨てていた
    sigma2 = float(np.median(res) ** 2 / 1.386 + 1e-6)
    try:
        cov = sigma2 * np.linalg.inv(J.T @ J)
        std_deg = float(np.degrees(np.sqrt(max(np.max(np.linalg.eigvalsh(cov[:n_rot, :n_rot])), 0.0))))
    except np.linalg.LinAlgError:
        std_deg = float("inf")
    return R, t, res, std_deg


def pnp_output_pose(st, R_out):
    """回転 R_out のもとで、F 重心が PnP の F 重心に来る姿勢（統合中の出力の並進を PnP で置き換える）。"""
    cF = ANCHOR_BY_KEY["F"]["centroid"].astype(np.float64)
    R_out = np.asarray(R_out, np.float64)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = R_out
    out[:3, 3] = (st["R"] @ cF + st["t"]) - R_out @ cF
    return out


def pnp_track(st, gray, c_prior, bgr=None, edge_ctx=None, anchor_ok=None):
    """キーフレーム特徴点を前フレームから LK で追い（前後追跡チェック）、姿勢を解く。bgr があれば点の補充に使う。
    edge_ctx = (向き付きエッジ距離マップ, 見えている歯群のキー) を渡すと、その歯の輪郭も一緒に使って解く（⑧）。
    戻り値 dict: ok, still（静止判定）, R, n, std_deg, res_px, deg（キーフレームからの回転）, disp_px, added, edge_q。"""
    idx = np.where(st["alive"])[0]
    out = {"ok": False, "still": False, "n": int(len(idx)), "std_deg": float("nan"), "res_px": float("nan"),
           "deg": float("nan"), "disp_px": float("nan"), "R": None, "added": 0}
    if len(idx) < FUSION_PNP_MIN_POINTS:
        return out
    p0 = st["cur"][idx]
    win = (min(_pnp_odd(21), 31),) * 2
    lvl = 3 + int(FUSION_PNP_PX > 1.5)        # 元の解像度では動きの px も倍になるのでピラミッドを 1 段増やす
    p1, s1, _ = cv2.calcOpticalFlowPyrLK(st["gray"], gray, p0, None, winSize=win, maxLevel=lvl)
    pb, s2, _ = cv2.calcOpticalFlowPyrLK(gray, st["gray"], p1, None, winSize=win, maxLevel=lvl)
    good = (s1.ravel() == 1) & (s2.ravel() == 1) & (np.linalg.norm(pb - p0, axis=1) < _pnp_px(FUSION_PNP_FB_MAX_PX))
    st["alive"][idx[~good]] = False
    st["cur"][idx[good]] = p1[good]
    st["gray"] = gray
    idx = np.where(st["alive"])[0]
    # ③ 見た目の確認: キーフレームの小画像と似ていない点（手に覆われた点など）を捨てる
    if len(idx):
        ncc = np.sum(_pnp_patches(gray, st["cur"][idx]) * st["kf_patches"][idx], axis=1)
        out["ncc_p10"] = float(np.percentile(ncc, 10))
        out["ncc_drop"] = int(np.count_nonzero(ncc < FUSION_PNP_NCC_MIN))
        st["alive"][idx[ncc < FUSION_PNP_NCC_MIN]] = False
        idx = np.where(st["alive"])[0]
    out["n"] = int(len(idx))
    if len(idx) < FUSION_PNP_MIN_POINTS:
        return out
    # ⑲ アンカー: 隠れていると判定された歯群（適合度が低い歯群）の上の点は、このフレームの解から外す
    #   （点は捨てない。歯群が見えるようになればまた使う）。見えている歯群の点が少なすぎるときは全部の点で解く
    out["anchors"] = None
    if anchor_ok is not None and FUSION_PNP_ANCHORS and "anchor" in st and len(st["anchor"]) == len(st["X"]):
        lab = st["anchor"][idx]
        use = np.isin(lab, list(anchor_ok))
        if int(use.sum()) >= FUSION_PNP_ANCHOR_MIN_POINTS:
            idx = idx[use]
            out["anchors"] = sorted(set(lab[use].tolist()))
            out["n"] = int(len(idx))
        else:
            out["anchors_fallback"] = True
    # ① 静止判定（ZUPT）: 大多数の点がキーフレームから動いていなければ、模型は動いていない
    disp = np.linalg.norm(st["cur"][idx] - st["kf_pts"][idx], axis=1)
    out["disp_px"] = float(np.median(disp))
    if out["disp_px"] < _pnp_px(FUSION_PNP_STILL_PX):
        # ⑩(a) 模型が止まっているのに動いている点は、手などに引きずられた点
        moving = disp >= _pnp_px(FUSION_PNP_STILL_DROP_PX)
        if FUSION_PNP_STILL_DROP_PX > 0 and moving.any():
            st["alive"][idx[moving]] = False
            out["still_drop"] = int(moving.sum())
            out["n"] = int(np.count_nonzero(st["alive"]))
        st["R"], st["t"] = st["R0"].copy(), st["t0"].copy()
        out.update(ok=True, still=True, R=st["R0"].copy(), deg=0.0, std_deg=0.0, res_px=0.0)
        return out
    # ② 前回の解とキーフレーム姿勢の両方から解き、1 px 以内に乗る点が多い方を採る
    X_, u_ = st["X"][idx], st["cur"][idx].astype(np.float64)
    cands = [_pnp_solve(st, X_, u_, c_prior, edge_ctx=edge_ctx),
             _pnp_solve(st, X_, u_, c_prior, R_init=st["R0"], t_init=st["t0"], edge_ctx=edge_ctx)]
    R, t, res, std = max(cands, key=lambda c: (int(np.count_nonzero(c[2] < _pnp_px(1.0))), -float(np.median(c[2]))))
    inl = res < _pnp_px(FUSION_PNP_OUTLIER_PX)
    out["inlier_frac"] = float(inl.mean())
    if inl.mean() < FUSION_PNP_MIN_INLIER_FRAC:
        # ⑩(b) 一部の点にしか合わない解は、手などの少数派に合わせた解の疑い → 採らない（点も捨てない）
        out.update(std_deg=std, res_px=float(np.median(res)),
                   deg=float(np.degrees(np.linalg.norm(cv2.Rodrigues(R @ st["R0"].T)[0]))))
        return out
    if FUSION_PNP_MIN_POINTS <= inl.sum() < len(idx):
        st["alive"][idx[~inl]] = False                 # 剛体運動に合わない点（手・反射）は以後使わない
        idx = idx[inl]
        R, t, res, std = _pnp_solve(st, st["X"][idx], st["cur"][idx].astype(np.float64), c_prior, R_init=R, t_init=t,
                                    edge_ctx=edge_ctx)
    out.update(n=int(len(idx)), std_deg=std, res_px=float(np.median(res)),
               deg=float(np.degrees(np.linalg.norm(cv2.Rodrigues(R @ st["R0"].T)[0]))))
    # ⑧ その姿勢で、見えている歯群の輪郭が画像のエッジに乗っている割合（品質評価と同じ尺度）
    edge_q = None
    if edge_ctx is not None and edge_ctx[0] is not None and edge_ctx[1]:
        _P = np.eye(4, dtype=np.float32)
        _P[:3, :3], _P[:3, 3] = R, t
        edge_q = float(np.mean([estimate_anchor_quality(edge_ctx[0], _P, ANCHOR_BY_KEY[k])
                                for k in edge_ctx[1] if k in ANCHOR_BY_KEY]))
    out["edge_q"] = edge_q
    if len(idx) >= FUSION_PNP_MIN_POINTS and out["res_px"] <= _pnp_px(FUSION_PNP_MAX_RES_PX) and std <= FUSION_PNP_MAX_STD_DEG:
        st["R"], st["t"] = R, t
        out.update(ok=True, R=R)
        # ⑤ 点が減っていて解が十分よいときは、今の姿勢で歯の上から点を補充する
        #   （⑧ 見えている歯の輪郭が合っていない姿勢では補充しない。ずれた姿勢で点を結び付けるとずれが固まる）
        if (bgr is not None and len(idx) < FUSION_PNP_REFRESH_BELOW
                and out["res_px"] <= _pnp_px(FUSION_PNP_REFRESH_MAX_RES_PX)
                and (edge_q is None or edge_q >= FUSION_PNP_REFRESH_MIN_EDGE_Q)):
            small = cv2.resize(bgr, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
            out["added"] = _pnp_refresh(st, gray, compute_tooth_mask(small).astype(np.uint8) * 255)
    return out


def compute_tooth_mask(bgr):
    """歯らしい色（低彩度・明るい）の画素マスク。並進探索の内部点採点用。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] < ALIGN_WHITE_S_MAX * 255) & (hsv[..., 2] > ALIGN_WHITE_V_MIN * 255)


def _mask_hit_fraction(mask, verts, candidate_pose, want=True):
    """verts を投影し、mask が want と一致する画素に乗っている割合（画面外は不一致扱い）。"""
    pts2d, valid = project_vertices(verts, candidate_pose)
    xi = np.round(pts2d[:, 0]).astype(np.int32)
    yi = np.round(pts2d[:, 1]).astype(np.int32)
    inb = valid & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if not inb.any():
        return 0.0
    hit = mask[yi[inb], xi[inb]]
    return float(np.count_nonzero(hit if want else ~hit)) / len(xi)


def estimate_fill_quality(tooth_mask, candidate_pose):
    """F モデルの内部点が歯色マスクに乗っている割合。"""
    return _mask_hit_fraction(tooth_mask, ANCHORS[0]["fill_vertices"], candidate_pose, want=True)


def estimate_outside_quality(tooth_mask, candidate_pose):
    """咬合面のすぐ外の帯が歯色「でない」割合（白い台座・板の上の偽候補を落とす）。"""
    return _mask_hit_fraction(tooth_mask, ANCHORS[0]["outside_vertices"], candidate_pose, want=False)


def estimate_color_quality(tooth_mask, candidate_pose):
    return float(np.sqrt(estimate_fill_quality(tooth_mask, candidate_pose)
                         * estimate_outside_quality(tooth_mask, candidate_pose)))


def estimate_search_quality(dist_map, candidate_pose, max_dist, tooth_mask):
    """並進探索の採点: エッジ一致（画面外も外れ、距離に応じた連続スコア、max_dist は縮小画像 px）と
    内部点の歯色一致の幾何平均。片方だけ高い偽候補（白い台座・ぼかした ArUco 板・周期ずれ）を
    落とすために和ではなく積にしている。"""
    a = ANCHORS[0]
    edge_q = estimate_tracking_quality_for(
        dist_map, candidate_pose, a["draw_vertices"], a["edge_a_idx"], a["edge_b_idx"],
        max_dist=max_dist, count_out_of_bounds=True, soft=True)
    return float(np.sqrt(edge_q * estimate_color_quality(tooth_mask, candidate_pose)))


def estimate_rigid_quality(dist_map, candidate_pose, keys=None, top_k=None):
    """剛体整合スコア: 同じ候補姿勢を RIGID_QUALITY_KEYS の各アンカーのジオメトリで
    評価し、上位 RIGID_QUALITY_TOP_K の平均を返す。
    アンカー単体の最大値だと、3本歯のアンカーが歯列の別の歯に乗っただけで 1.0 に
    なり「4グループが同一剛体である」という制約を全く見ないため、ロスト/良好判定と
    自動姿勢選択の採点にはこちらを使う。"""
    keys = keys or RIGID_QUALITY_KEYS
    top_k = top_k or RIGID_QUALITY_TOP_K
    qs = sorted((estimate_anchor_quality(dist_map, candidate_pose, ANCHOR_BY_KEY[k]) for k in keys),
                reverse=True)
    top = qs[:max(1, min(top_k, len(qs)))]
    return float(np.mean(top))


def _predict_pose_by_velocity(pose_mat, vel_t, vel_rv):
    """① 出力速度で 1 フレーム先の姿勢を予測する（回転は歯列重心まわり、並進はカメラ座標）。"""
    out = pose_mat.copy().astype(np.float32)
    ang = float(np.linalg.norm(vel_rv))
    if ang > 1e-6:
        Rd, _ = cv2.Rodrigues(np.asarray(vel_rv, np.float64))
        out = _rotate_pose_about_point(out, Rd, _group_center_cam("F", out))
    out[:3, 3] += np.asarray(vel_t, np.float32)
    return out


def estimate_combined_quality(dist_map, candidate_pose):
    """自動姿勢選択の粗選抜スコア。以前は「どのアンカーか1つでも合えばよい」として
    max を取っていたが、それが誤姿勢を高得点にしていたため剛体整合スコアに変更。"""
    return estimate_rigid_quality(dist_map, candidate_pose)


def best_anchor_pose_and_quality(dist_map):
    """各アンカーモデルの「今の」内部姿勢(pose_cv)を読み、品質が最も高い
    アンカーの (key, pose, quality) を返す。これが Anchor Switching の中心。
    候補は AUTO_ANCHOR_KEYS（Unity から手動指定があればそのアンカーのみ）。"""
    keys = (manual_anchor_override,) if manual_anchor_override in ANCHOR_BY_KEY else AUTO_ANCHOR_KEYS
    best = None
    for k in keys:
        a = ANCHOR_BY_KEY[k]
        p = a["model"].pose_cv.copy().astype(np.float32)
        q = estimate_anchor_quality(dist_map, p, a)
        if best is None or q > best[2]:
            best = (a["key"], p, q)
    return best


def projected_center(candidate_pose, draw_verts=None):
    if draw_verts is None:
        draw_verts = ANCHORS[0]["draw_vertices"]
    center_model = np.mean(draw_verts, axis=0)
    center_cam = candidate_pose[:3, :3] @ center_model + candidate_pose[:3, 3]
    if center_cam[2] <= 0.05:
        return None
    x = K[0, 0] * center_cam[0] / center_cam[2] + K[0, 2]
    y = K[1, 1] * center_cam[1] / center_cam[2] + K[1, 2]
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return np.array([x, y], dtype=np.float32)


def rotation_angle_deg(pose_a, pose_b):
    r_delta = pose_a[:3, :3] @ pose_b[:3, :3].T
    cos_angle = np.clip((float(np.trace(r_delta)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))


def generate_candidate_poses():
    """保存済み12スロットそれぞれに、安価な回転摂動を加えて候補を密にする。"""
    candidates = []
    for slot in range(1, POSE_SLOT_COUNT + 1):
        base_pose, _ = load_pose_slot(slot)
        if base_pose is None:
            continue
        for yaw in COARSE_YAW_STEPS_DEG:
            for pitch in COARSE_PITCH_STEPS_DEG:
                candidate = base_pose.copy()
                if yaw != 0.0 or pitch != 0.0:
                    delta = rot_y(yaw) @ rot_x(pitch)
                    candidate[:3, :3] = delta @ base_pose[:3, :3]
                candidates.append((slot, candidate))
    return candidates


_ALIGN_OBJ_CENTER = None


def _align_object_center():
    """回転の中心に使うモデル座標系の重心（F アンカーの品質判定用頂点の平均）。"""
    global _ALIGN_OBJ_CENTER
    if _ALIGN_OBJ_CENTER is None:
        _ALIGN_OBJ_CENTER = np.mean(ANCHORS[0]["draw_vertices"], axis=0).astype(np.float32)
    return _ALIGN_OBJ_CENTER


def _perturb_pose(base_pose, dx, dy, depth_factor, yaw, pitch):
    """base_pose を「物体重心まわりに yaw/pitch 回転 → 同じ視線上で奥行きを depth_factor 倍
    → カメラ x/y に dx,dy [m] 平行移動」した候補を返す。
    回転をモデル原点ではなく重心まわりに掛けるのは、原点が物体から離れていると回転だけで
    画面上の位置が大きく飛び、格子の意味が無くなるため。"""
    R = base_pose[:3, :3]
    t = base_pose[:3, 3]
    c_model = _align_object_center()
    c_cam = R @ c_model + t
    if yaw != 0.0 or pitch != 0.0:
        R = (rot_y(yaw) @ rot_x(pitch)) @ R
    c_cam = c_cam * depth_factor
    c_cam = c_cam + np.array([dx, dy, 0.0], dtype=np.float32)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = R
    out[:3, 3] = c_cam - R @ c_model
    return out


def _align_base_poses():
    """並進探索の基準姿勢: 保存スロット + 現在姿勢 + 最後の良好姿勢（重複は除く）。"""
    bases = []
    for slot in range(1, POSE_SLOT_COUNT + 1):
        p, _ = load_pose_slot(slot)
        if p is not None:
            bases.append((slot, p.astype(np.float32)))
    bases.append((None, pose.copy().astype(np.float32)))
    if last_good_pose is not None:
        bases.append((None, last_good_pose.copy().astype(np.float32)))
    uniq = []
    for slot, p in bases:
        # 格子（±40 mm / ±20°）で十分カバーできる近い基準姿勢は 1 つにまとめる（基準が増えると探索時間が倍）
        dup = any(rotation_angle_deg(p, q) < 10.0 and np.linalg.norm(p[:3, 3] - q[:3, 3]) < 0.015
                  for _, q in uniq)
        if not dup:
            uniq.append((slot, p))
    return uniq


_align_base_centers = []


def _align_prior_distance(candidate_pose):
    """候補の物体中心と、最も近い基準姿勢の物体中心との距離 [m]。"""
    if not _align_base_centers:
        return 0.0
    c = candidate_pose[:3, :3] @ _align_object_center() + candidate_pose[:3, 3]
    return float(min(np.linalg.norm(c - b) for b in _align_base_centers))


def _score_grid(dist_map, tooth_mask, slot, base_pose, xy_offsets, depth_factors, yaws, pitches, max_dist,
                ref_area_norm=None):
    """格子の全候補を粗採点して [(q, slot, pose), ...] を返す（q は基準姿勢からの距離ペナルティ込み）。
    (奥行き, yaw, pitch) ごとに投影は 1 回だけ行い、画面平行の xy オフセットはカメラ座標の平行移動
    なので各点の画素シフト = fx·dx/Z で一括計算する（1 候補ずつ投影すると 1.5 ms × 5000 = 7.6 s
    かかっていた。2026-10-01 実機ログ）。"""
    a = ANCHORS[0]
    offs = np.asarray(xy_offsets, dtype=np.float32).reshape(-1, 2)
    n_off = len(offs)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    hm, wm = dist_map.shape[1:3]
    sc = QUALITY_SCALE
    c_model = _align_object_center()
    margin = ALIGN_CENTER_MARGIN_PX
    min_area = ref_area_norm[0] * ALIGN_MIN_AREA_RATIO if ref_area_norm else 0.0
    min_short = ref_area_norm[1] * ALIGN_MIN_SHORT_RATIO if ref_area_norm else 0.0
    ea, eb = a["edge_a_idx"], a["edge_b_idx"]
    T = EDGE_SAMPLE_T
    dx_px = (fx * offs[:, 0]).astype(np.float32)   # 画素シフト = dx_px / Z
    dy_px = (fy * offs[:, 1]).astype(np.float32)

    def _mask_frac(verts, R, t, want):
        pc = (R @ verts.T).T + t
        vz = pc[:, 2] > 0.05
        invz = 1.0 / np.maximum(pc[:, 2], 1e-6)
        vx = fx * pc[:, 0] * invz + cx
        vy = fy * pc[:, 1] * invz + cy
        xi = np.round(vx[None, :] + dx_px[:, None] * invz[None, :]).astype(np.int32)   # (n_off, M)
        yi = np.round(vy[None, :] + dy_px[:, None] * invz[None, :]).astype(np.int32)
        inb = vz[None, :] & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        hit = np.zeros(xi.shape, dtype=bool)
        hit[inb] = tooth_mask[yi[inb], xi[inb]]
        if not want:
            hit = inb & ~hit
        return hit.sum(axis=1) / float(verts.shape[0])

    results = []
    for depth_factor in depth_factors:
        for yaw in yaws:
            for pitch in pitches:
                p0 = _perturb_pose(base_pose, 0.0, 0.0, depth_factor, yaw, pitch)
                R, t = p0[:3, :3], p0[:3, 3]
                if not (0.03 < float(t[2]) < 2.0):
                    continue
                if min_area > 0.0:
                    area_n, short_n = _projected_extent_norm(p0)   # xy シフトでは変わらない
                    if area_n < min_area or short_n < min_short:
                        continue
                # 物体中心の画面位置（オフセットごと）
                c_cam = R @ c_model + t
                if c_cam[2] <= 0.05:
                    continue
                ccx = fx * (c_cam[0] + offs[:, 0]) / c_cam[2] + cx
                ccy = fy * (c_cam[1] + offs[:, 1]) / c_cam[2] + cy
                ok = (ccx >= margin) & (ccx <= w - margin) & (ccy >= margin) & (ccy <= h - margin)
                if not ok.any():
                    continue
                # エッジ一致（ソフトスコア）
                pc = (R @ a["draw_vertices"].T).T + t
                valid = pc[:, 2] > 0.05
                invz = 1.0 / np.maximum(pc[:, 2], 1e-6)
                x = fx * pc[:, 0] * invz + cx
                y = fy * pc[:, 1] * invz + cy
                sx = x[ea][:, None] * (1.0 - T) + x[eb][:, None] * T          # (E, nT)
                sy = y[ea][:, None] * (1.0 - T) + y[eb][:, None] * T
                sinv = invz[ea][:, None] * (1.0 - T) + invz[eb][:, None] * T
                ev = valid[ea] & valid[eb]
                d = np.stack([x[eb] - x[ea], y[eb] - y[ea]], axis=1)
                normal_ang = np.mod(np.arctan2(d[:, 1], d[:, 0]) + np.pi / 2.0, np.pi)
                nb = np.round(normal_ang / (np.pi / EDGE_ORIENT_BINS)).astype(np.int32) % EDGE_ORIENT_BINS
                xi = np.round((sx[None] + dx_px[:, None, None] * sinv[None]) * sc).astype(np.int32)   # (n_off, E, nT)
                yi = np.round((sy[None] + dy_px[:, None, None] * sinv[None]) * sc).astype(np.int32)
                inb = ev[None, :, None] & (xi >= 0) & (xi < wm) & (yi >= 0) & (yi < hm)
                xc = np.clip(xi, 0, wm - 1)
                yc = np.clip(yi, 0, hm - 1)
                dist = np.full(xi.shape, np.inf, dtype=np.float32)
                for off in (-1, 0, 1):
                    layer = (1 + ((nb + off) % EDGE_ORIENT_BINS))[None, :, None]
                    dist = np.minimum(dist, dist_map[layer, yc, xc])
                soft = np.where(inb, np.clip(1.0 - dist / max_dist, 0.0, 1.0), 0.0)
                total = float(np.count_nonzero(ev) * len(T))
                edge_q = soft.sum(axis=(1, 2)) / total if total > 0 else np.zeros(n_off)
                fill_q = _mask_frac(a["fill_vertices"], R, t, True)
                out_q = _mask_frac(a["outside_vertices"], R, t, False)
                q = np.sqrt(edge_q * np.sqrt(fill_q * out_q))
                # 基準姿勢からの距離ペナルティ
                centers = c_cam[None, :] + np.concatenate([offs, np.zeros((n_off, 1), np.float32)], axis=1)
                if _align_base_centers:
                    prior = np.min([np.linalg.norm(centers - b[None, :], axis=1) for b in _align_base_centers], axis=0)
                else:
                    prior = np.zeros(n_off, np.float32)
                q = q - ALIGN_PRIOR_PER_M * prior
                for k in np.nonzero(ok)[0]:
                    cand = p0.copy()
                    cand[0, 3] += offs[k, 0]
                    cand[1, 3] += offs[k, 1]
                    results.append((float(q[k]), slot, cand))
    return results


def _projected_extent_norm(candidate_pose):
    """内部点の投影 bbox の (面積 × 奥行き², 短辺 × 奥行き)。奥行きによる見かけの縮小を打ち消し、
    「向きによる潰れ」だけを見るための量。"""
    pts2d, valid = project_vertices(ANCHORS[0]["fill_vertices"], candidate_pose)
    if np.count_nonzero(valid) < 10:
        return 0.0, 0.0
    p = pts2d[valid]
    span = p.max(axis=0) - p.min(axis=0)
    z = float(candidate_pose[2, 3])
    return float(span[0] * span[1]) * z * z, float(min(span[0], span[1])) * z


def _projected_area_norm(candidate_pose):
    return _projected_extent_norm(candidate_pose)[0]


def _nms_poses(results, max_keep):
    """粗採点結果を降順に見て、既に採用した候補と近い（並進 ALIGN_NMS_TRANS_M・回転
    ALIGN_NMS_ROT_DEG 以内）ものは捨て、多様なセルを max_keep 個残す。"""
    kept = []
    for q, slot, cand in sorted(results, key=lambda r: r[0], reverse=True):
        near = any(np.linalg.norm(cand[:3, 3] - k[2][:3, 3]) < ALIGN_NMS_TRANS_M
                   and rotation_angle_deg(cand, k[2]) < ALIGN_NMS_ROT_DEG for k in kept)
        if not near:
            kept.append((q, slot, cand))
            if len(kept) >= max_keep:
                break
    return kept


def translation_search_candidates(dist_map, tooth_mask):
    """Quick Align 用: 並進 + 回転の格子を粗採点 → 上位セルを局所細分 → 多様な上位
    ALIGN_TOP_K 候補を [(coarse_q, slot, pose), ...] で返す。"""
    t0 = time.perf_counter()
    steps = np.arange(-ALIGN_XY_RANGE_M, ALIGN_XY_RANGE_M + 1e-6, ALIGN_XY_STEP_M, dtype=np.float32)
    xy_offsets = [(float(dx), float(dy)) for dx in steps for dy in steps]
    bases = _align_base_poses()
    global _align_base_centers
    _align_base_centers = [bp[:3, :3] @ _align_object_center() + bp[:3, 3] for _, bp in bases]
    ext = np.array([_projected_extent_norm(bp) for _, bp in bases])
    ref_area = (float(ext[:, 0].max()), float(ext[:, 1].max()))   # 最も正面向きの基準姿勢を基準にする
    coarse = []
    for slot, base_pose in bases:
        coarse += _score_grid(dist_map, tooth_mask, slot, base_pose, xy_offsets, ALIGN_DEPTH_FACTORS,
                              ALIGN_YAW_STEPS_DEG, ALIGN_PITCH_STEPS_DEG, ALIGN_COARSE_TOL_PX, ref_area)
    t1 = time.perf_counter()
    cells = _nms_poses(coarse, ALIGN_LOCAL_CELLS)
    local_xy = [(dx, dy) for dx in ALIGN_LOCAL_XY_M for dy in ALIGN_LOCAL_XY_M]
    fine = []
    for q, slot, cell_pose in cells:
        fine += _score_grid(dist_map, tooth_mask, slot, cell_pose, local_xy, ALIGN_LOCAL_DEPTH,
                            ALIGN_LOCAL_YAW_DEG, ALIGN_LOCAL_PITCH_DEG, ALIGN_FINE_TOL_PX, ref_area)
    t2 = time.perf_counter()
    shortlist = _nms_poses(fine, ALIGN_TOP_K)
    print(f"並進探索: 基準 {len(bases)} 姿勢 × 格子 = 粗 {len(coarse)} 候補 ({t1 - t0:.1f}s) → "
          f"上位 {len(cells)} セルを細分 {len(fine)} 候補 ({t2 - t1:.1f}s) → 精密化 {len(shortlist)} 候補")
    for q, slot, cand in shortlist:
        c = projected_center(cand)
        print(f"  粗候補 q={q:.2f} slot={slot} 画面({c[0]:.0f},{c[1]:.0f}) depth={cand[2, 3]:.3f}")
    return shortlist


def pose_auto_select(frame_bgr, dist_map, use_rotation_guard=True, reference_pose=None,
                     translation_search=False):
    """2段階の自動姿勢選択。
    Stage1（安価）: 候補全部を生の投影でCannyエッジと粗く採点（tracker.step()なし）。
    Stage2（高価）: 上位 COARSE_TOP_K だけ実際にトラッカーで精密化して最終採点。
    translation_search=True（Unity の Quick Align）: Stage1 を並進 + 回転の格子探索
    （translation_search_candidates）に置き換え、模型がスロット保存時と別の場所にあっても
    数 cm / 20〜30° の範囲で見つけられるようにする。数秒かかる。
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    original_pose = pose.copy().astype(np.float32)
    if reference_pose is None:
        reference_pose, _ = load_pose_slot(selected_pose_slot)
        if reference_pose is None:
            reference_pose = original_pose

    refine_steps = AUTO_SELECT_STEPS
    tooth_mask = None
    if translation_search and ALIGN_TRANS_SEARCH_ENABLED:
        tooth_mask = compute_tooth_mask(frame_bgr)
        shortlist = [(q, rotation_angle_deg(cand, reference_pose), slot, cand)
                     for q, slot, cand in translation_search_candidates(dist_map, tooth_mask)]
        use_rotation_guard = False
        refine_steps = ALIGN_REFINE_STEPS
        if not shortlist:
            reset_all_anchors(original_pose)
            print("自動姿勢選択失敗: 並進探索で候補がありません")
            return original_pose, None
    else:
        # 保存スロット由来の候補に加えて「今の pose」と「最後の良好 pose」も候補に入れる。
        # 誤検知でリカバリが走ったときに、遠いスロット近傍へ飛ばず現在位置へ戻れるようにする。
        candidates = list(generate_candidate_poses())
        candidates.append((None, original_pose.copy()))
        if last_good_pose is not None:
            candidates.append((None, last_good_pose.copy().astype(np.float32)))

        coarse_results = []
        for slot, candidate_pose in candidates:
            rotation_delta = rotation_angle_deg(candidate_pose, reference_pose)
            if use_rotation_guard and rotation_delta > ROTATION_REJECT_DEG:
                continue
            depth = float(candidate_pose[2, 3])
            if not (0.03 < depth < 2.0):
                continue
            if projected_center(candidate_pose) is None:
                continue
            coarse_quality = estimate_combined_quality(dist_map, candidate_pose)
            coarse_results.append((coarse_quality, rotation_delta, slot, candidate_pose))

        if not coarse_results:
            reset_all_anchors(original_pose)
            print("自動姿勢選択失敗: 粗選抜を通った候補がありません")
            return original_pose, None

        coarse_results.sort(key=lambda item: item[0], reverse=True)
        shortlist = coarse_results[:COARSE_TOP_K]
        print(f"自動姿勢選択: 候補 {len(coarse_results)} 個中、上位 {len(shortlist)} 個を精密化")

    best_slot, best_pose, best_score, best_quality, best_anchor_key = None, None, np.inf, 0.0, None
    for coarse_quality, rotation_delta, slot, candidate_pose in shortlist:
        before_center = projected_center(candidate_pose)
        reset_all_anchors(candidate_pose)
        for _ in range(refine_steps):
            tracker.step(frame_rgb)
        # 両アンカーとも同じ候補姿勢から精密化するが、それぞれ独立に収束するので
        # 結果として品質が高い方のアンカーの姿勢を、この候補の代表結果として使う
        # （Anchor Switchingの考え方を自動姿勢選択にもそのまま適用する）。
        anchor_key, refined_pose, _ = best_anchor_pose_and_quality(dist_map)
        # 候補の採点は単体アンカーの値ではなく剛体整合スコアで行う
        quality = estimate_rigid_quality(dist_map, refined_pose)
        if tooth_mask is not None:
            # 並進探索: M3T 精密化後もエッジだけだと周期ずれの偽姿勢が真値より高得点になるため、
            # 内部点の歯色一致を掛ける（粗採点と同じ考え方）。
            quality = float(np.sqrt(quality * estimate_color_quality(tooth_mask, refined_pose)))
        after_center = projected_center(refined_pose)
        if after_center is None:
            continue

        move_px = float(np.linalg.norm(after_center - before_center))
        depth = float(refined_pose[2, 3])
        center_penalty = float(np.linalg.norm(after_center - np.array([w * 0.5, h * 0.5], dtype=np.float32))) * 0.02
        depth_penalty = 0.0 if 0.03 < depth < 2.0 else 10000.0
        if use_rotation_guard and reference_pose is not None:
            trans_from_ref = float(np.linalg.norm(refined_pose[:3, 3] - reference_pose[:3, 3]))
            if trans_from_ref > RELOCK_MAX_TRANS_M:
                print(f"  候補(元スロット{slot}): 基準姿勢から {trans_from_ref*1000:.0f} mm 離れているため不採用 (q={quality:.2f})")
                continue
        rotation_penalty = rotation_delta * ROTATION_PENALTY_WEIGHT if use_rotation_guard else 0.0
        quality_penalty = (1.0 - quality) * QUALITY_SELECT_WEIGHT
        prior_penalty = (_align_prior_distance(refined_pose) * 1000.0 * ALIGN_PRIOR_FINAL_PER_MM
                         if tooth_mask is not None else 0.0)
        score = quality_penalty + move_px + center_penalty + depth_penalty + rotation_penalty + prior_penalty
        print(f"  候補(元スロット{slot}): score={score:.2f} coarse_q={coarse_quality:.2f} "
              f"refined_q={quality:.2f}(anchor {anchor_key}) move_px={move_px:.2f} "
              f"rot={rotation_delta:.1f}deg depth={depth:.3f}")

        if score < best_score:
            best_score, best_slot, best_pose, best_quality, best_anchor_key = score, slot, refined_pose, quality, anchor_key

    if best_pose is None:
        reset_all_anchors(original_pose)
        print("自動姿勢選択失敗: 精密化後に有効な候補がありません")
        return original_pose, None

    reset_all_anchors(best_pose)
    _auto_select_info["quality"] = float(best_quality)
    print(f"自動選択完了（元スロット{best_slot}近傍、アンカー{best_anchor_key}）: "
          f"score={best_score:.2f} quality={best_quality:.2f}")
    if translation_search and best_quality < ALIGN_MIN_QUALITY:
        print(f"警告: 並進探索の最良候補でも quality={best_quality:.2f} < {ALIGN_MIN_QUALITY}。"
              "模型が探索範囲（±4 cm / 奥行き ±25% / 30°）の外か、手や器具で隠れている可能性。"
              "模型を画面中央寄りに置き直して Quick Align をやり直してください")
    return best_pose, best_slot


def snap_to_current_frame(frame_bgr, steps=SNAP_START_STEPS):
    """steps回 tracker.step() を進めた後、その時点で品質が良い方のアンカーを
    選び (anchor_key, pose) を返す。"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    for _ in range(steps):
        tracker.step(frame_rgb)
    dist_map = compute_edge_distance_map(frame_bgr)
    best_key, best_pose, _ = best_anchor_pose_and_quality(dist_map)
    return best_key, best_pose


def start_tracking_from_current_pose(frame_bgr, light_snap_steps=5):
    """PnP や手動調整後に現在の pose をそのまま使って追跡を開始する。
    pose_auto_select を呼ばないため、固定カメラ環境で背景エッジに引き込まれない。
    light_snap_steps: 数値計算の安定化のみを目的とした最小限の tracker.step()。
    """
    global pose, snap_frames, tracking_active, active_anchor, prev_output_pose
    prev_output_pose = None   # 跳び量ゲートの基準を無効化
    global lost_streak, last_recovery_time, last_good_pose, last_quality, recovery_escalated
    global quality_smoothed, last_good_time, relock_reason, lost_start_time

    tracking_active = True
    reset_all_anchors(pose)

    if light_snap_steps > 0:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for _ in range(light_snap_steps):
            tracker.step(frame_rgb)
        dist_map = compute_edge_distance_map(frame_bgr)
        active_anchor, pose, _ = best_anchor_pose_and_quality(dist_map)

    # PnP後は既に位置合わせ済みなので SNAP ウォームアップは行わない
    snap_frames = 0
    relock_reason = "pnp"
    lost_streak = 0
    lost_start_time = None
    last_recovery_time = time.time()
    last_good_pose = pose.copy()
    last_quality = 1.0
    quality_smoothed = 1.0
    last_good_time = time.time()
    recovery_escalated = False
    pos_filter.x_prev = None
    rot_filter.q_prev = None
    print(f"追跡開始 / 現在姿勢（PnP/手動）から直接開始 / 軽量SNAP {light_snap_steps}ステップ / アクティブアンカー={active_anchor}")


# ─── ワイヤーフレーム描画 ─────────────────────────────────────────────────

def _draw_wireframe_into(out, pose_mat, viz_vertices, viz_edges, color, thickness):
    R = pose_mat[:3, :3]
    t = pose_mat[:3, 3]
    pts_cam = (R @ viz_vertices.T).T + t
    valid = pts_cam[:, 2] > 0.05
    if np.count_nonzero(valid) < 10:
        return False

    x = K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]
    y = K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]
    pts2d = np.stack([x, y], axis=1)
    finite = np.isfinite(pts2d).all(axis=1)

    for a, b in viz_edges:
        if not (valid[a] and valid[b] and finite[a] and finite[b]):
            continue
        p1, p2 = pts2d[a].astype(int), pts2d[b].astype(int)
        if (0 <= p1[0] < w and 0 <= p1[1] < h) or (0 <= p2[0] < w and 0 <= p2[1] < h):
            cv2.line(out, tuple(p1), tuple(p2), color, thickness)
    return True


def draw_obj_wireframe(bgr, pose_mat, viz_vertices=None, viz_edges=None, color=(255, 0, 255)):
    """単一アンカー用（ADJUST モードなど、姿勢が1つしかない場合に使う）。"""
    if viz_vertices is None:
        viz_vertices = ANCHORS[0]["viz_draw_vertices"]
        viz_edges = ANCHORS[0]["viz_draw_edges"]
    out = bgr.copy()
    ok = _draw_wireframe_into(out, pose_mat, viz_vertices, viz_edges, color, 1)
    if not ok:
        cv2.putText(out, "Model is behind camera or too close", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


def draw_dual_wireframe(bgr, poses_by_key, active_key):
    """2アンカーを同時に描く。アクティブ側は太く/固有色、非アクティブ側は
    グレーで薄く描いて、Anchor Switching の「今どちらを採用しているか」を
    一目で分かるようにする（論文の可視化方針を踏襲）。"""
    out = bgr.copy()
    # 非アクティブを先に描き、アクティブを後から上書きして見やすくする。
    # 階層アンカーでは F（歯列全体）が A〜D を含むので、非アクティブの歯群は描かない
    # （5 体 × 2500 本の cv2.line で 55 ms かかっていた）。F 非アクティブ時は F だけ薄く描く。
    for a in ANCHORS:
        if a["key"] == active_key:
            continue
        if HIERARCHICAL_ANCHORS and a["key"] != "F":
            continue
        _draw_wireframe_into(out, poses_by_key[a["key"]], a["viz_draw_vertices"], a["viz_draw_edges"],
                              INACTIVE_ANCHOR_COLOR, 1)
    active = ANCHOR_BY_KEY[active_key]
    ok = _draw_wireframe_into(out, poses_by_key[active_key], active["viz_draw_vertices"], active["viz_draw_edges"],
                               active["active_color"], 2)
    if not ok:
        cv2.putText(out, "Model is behind camera or too close", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


# ─── 追跡開始・再ロック ───────────────────────────────────────────────────

lost_streak = 0
lost_start_time = None     # 品質が LOSS を割り始めた時刻（HOLD → 自動再ロックの切替判定に使う）
last_recovery_time = 0.0
last_good_pose = None
last_good_time = 0.0       # last_good_pose を更新した時刻（古い pose へ引き戻さないための判定用）
last_quality = 0.0         # 判定に使う平滑化済み剛体整合スコア
quality_raw = 0.0          # 平滑化前の剛体整合スコア（HUD/CSV 用）
quality_smoothed = 0.0
relock_reason = ""         # 直近の SNAP を起こした経路（auto/manual/start/pnp）。HUD と CSV の mode に付く
recovery_escalated = False
good_streak = 0
last_auto_capture_time = 0.0
auto_capture_cursor = 0
active_anchor = ARCH_ANCHOR_KEY  # 現在 Unity へ出力している姿勢の出どころ（F / A〜D）
manual_anchor_override = ""  # Unity トグルUIからの手動指定。空文字 = 自動切替
unity_display_anchor = ""    # （旧）単体ボディ表示専用の指定。階層アンカーでは display_anchor を使う
display_anchor = ""          # Unity で強調表示する歯群（適合度が最も高い歯群、または切替先）。空文字 = 全体
anchor_switch_time = 0.0     # 直近のアンカー切替時刻（最低滞在時間の判定用）
anchor_switching_enabled = ANCHOR_SWITCHING_DEFAULT   # False = 比較用ベースライン（F のみ）
anchor_fusion_enabled = ANCHOR_FUSION_DEFAULT         # True = 見えている歯群すべてを統合（False = 最良 1 歯群を選択）
fusion_active = False        # いま統合姿勢を出しているか
fusion_keys = []             # 統合に使っている歯群
fusion_off_f = False         # 統合モードで F を離れている（歯群が使えなくても F の生姿勢には戻らず保持する）
fusion_members = {}          # 統合に採用中の歯群 → 採用時刻（ヒステリシス用）
fusion_enter_streak = {}     # 歯群ごとの ENTER 閾値連続フレーム数
fusion_hold_pose = None      # 統合中の直前出力姿勢（使える歯群がない間の保持、並進平滑化、採用判定の基準）
fusion_f_ok_since = None     # F のスコアが回復閾値以上になり続けている開始時刻
fusion_no_member_since = None  # 使える歯群が無くなった時刻
anchor_switch_pose = None    # 歯群へ切り替えた時点の姿勢（逸脱監視の基準）
group_block_until = 0.0      # 逸脱で F へ戻した後、この時刻まで歯群への再切替を禁止
prev_output_pose = None      # 前フレームの出力姿勢（跳び量ゲートの基準）。再ロック直後は無効
jump_target_pose = None      # ゲートが拒否した跳び先（連続性の判定用）
jump_target_streak = 0
gate_reject_streak = 0       # ゲートが連続して拒否したフレーム数（固着検出）
force_relock = False         # 固着検出で自動再ロックを強制するフラグ
rotation_uncertain_until = 0.0         # ② F 復帰後、この時刻までは「回転不確か」警告を維持
rotation_uncertain_after_return = ""   # ② 復帰時の警告文
rotation_uncertain_reason = ""         # ② 今フレームの警告理由（空 = 警告なし）。Unity へ送る
frame_event = ""                       # CSV の event 列（復帰時回転再探索の結果など、そのフレームの特記事項）
group_poses_raw = None                 # ⑤ M3T ステップ直後の各歯群ボディの姿勢（統合で上書きされる前のもの）
return_search_job = None               # ① 分割実行中の復帰時回転再探索（None = なし）。return_search_begin/advance 参照
fusion_rot_ref = None                  # 統合中の基準姿勢（回転 = 現在の出力回転、F 重心は離脱時の位置）。_fuse_groups の ref
fusion_rot_src = ""                    # いま回転を追従している歯群（"" = 固定）
fusion_rot_total_deg = 0.0             # 離脱時の回転からの累計回転量
fusion_rot_motion_ref = None           # 回転追従の動き検出の基準（motion_ref_begin の戻り値）
fusion_rot_motion_until = 0.0          # この時刻までは「模型が動いた」とみなして回転追従を許す
fusion_rot_motion_px = None            # 今フレームの基準からの移動量 [px, 縮小画像]（CSV 用）
pnp_last_good = None                   # 直近の F 良好フレーム (縮小グレー, F 姿勢, 時刻)。遮蔽が始まったらキーフレームにする
pnp_f_since = None                     # F で出力し始めた時刻（統合中は None）。戻った直後の F はキーフレームにしない
f_guard = None                         # ⑥ F 追跡中の見張り用キーフレーム特徴点 PnP（pnp_keyframe_begin の戻り値）
f_guard_time = 0.0                     # その作成時刻
f_guard_streak = 0                     # F が PnP から離れている連続フレーム数
f_guard_after_return = False           # ⑯ 統合から戻った後、まだ見張りのキーフレームを作っていない
fusion_pnp_bad_since = None            # ⑦ 統合中に PnP が使えなくなった時刻（使えていれば None）
fusion_stuck_relock_time = 0.0         # ⑦ 最後に抜け出せない状態から再ロックした時刻
fusion_pnp = None                      # 統合中のキーフレーム特徴点 PnP の状態（pnp_keyframe_begin の戻り値）
fusion_pnp_info = None                 # 今フレームの PnP 結果（CSV 用）
fusion_rot_streak = {}                 # 歯群ごとの「回転源の条件を満たした連続フレーム数」
fusion_rot_prev_R = {}                 # 歯群ごとの前フレームの回転（安定判定用）
fusion_rot_bias = {}                   # 歯群ごとの「歯群ボディの回転 → 出力回転」相対回転（モデル座標 3x3、F 追跡中に EMA 学習）
marker_stale_since = None              # マーカー基準が古い疑い（F 良好なのに誤差大）が続き始めた時刻
marker_stale_warned = False
pose_vel_t = np.zeros(3)               # ① 出力姿勢の並進速度 [m/フレーム]（EMA）
pose_vel_rv = np.zeros(3)              # ① 出力姿勢の回転速度（回転ベクトル [rad/フレーム]、EMA）
prev_pose_src = None                   # 前フレームの姿勢源（"F" / "A+B" / "-" など。F への切替で速度を 0 にする）
coast_until = 0.0                      # ① この時刻までは HOLD で凍結せず惰性で追う
coast_decided = False                  # ① 今回の HOLD で惰性にするか判定済みか（HOLD に入った瞬間に 1 回だけ決める）
coasting = False                       # ① 今フレーム惰性追跡中か（統合の「使える歯群なし」分岐が参照）
fusion_solo_blocked = ""               # 案 A: 単独のため統合から外している歯群（ログ用）
nontooth_fill_ema = None               # 歯色マスクに乗っている F 内部点の割合（EMA）。F 追跡中だけ更新
nontooth_since = None                  # 割合が NONTOOTH_FILL_MIN を割り続け始めた時刻
nontooth_reason = ""                   # 今フレームの警告文（Unity バナー用）
nontooth_low_sec = 0.0                 # 割合が低かった累積秒数（統合中は加算せず、F に戻ったら続きから）
nontooth_last_t = None
session_depth_ref = None               # 追跡開始後に最初に良好になった姿勢の奥行き [m]（奥行き窓の基準）
depth_window_hit = False
pose_jump_rejects = 0        # ゲートで捨てた回数（ログ用）
relock_fail_count = 0        # 自動再ロックの連続失敗回数（エスカレーション用）
_auto_select_info = {"quality": 0.0}   # pose_auto_select が最後に選んだ候補の品質


def start_tracking_from_frame(frame_bgr, use_rotation_guard=True):
    global pose, selected_pose_slot, snap_frames, tracking_active, active_anchor, prev_output_pose
    prev_output_pose = None   # 跳び量ゲートの基準を無効化（再ロック先を「跳び」と誤判定しないため）
    global lost_streak, last_recovery_time, last_good_pose, last_quality, recovery_escalated
    global quality_smoothed, last_good_time, relock_reason, lost_start_time

    tracking_active = True
    dist_map = compute_edge_distance_map(frame_bgr)
    auto_pose, auto_slot = pose_auto_select(frame_bgr, dist_map, use_rotation_guard=use_rotation_guard)
    pose = auto_pose
    if auto_slot is not None:
        selected_pose_slot = auto_slot

    active_anchor, pose = snap_to_current_frame(frame_bgr)
    snap_frames = SNAP_FRAME_COUNT

    lost_streak = 0
    lost_start_time = None
    last_recovery_time = time.time()
    last_good_pose = pose.copy()
    # SNAP のウォームアップ中は品質判定をスキップするため、ここで暫定的に
    # 「良好」扱いにしておく（そうしないと古い last_quality が低いままだと
    # Unity 送信のフリーズゲートがこの新しい姿勢を止めてしまう）。
    last_quality = 1.0
    quality_smoothed = 1.0
    last_good_time = time.time()
    recovery_escalated = False
    pos_filter.x_prev = None
    rot_filter.q_prev = None

    mode = "SAFE" if use_rotation_guard else "GLOBAL"
    relock_reason = "start"
    print(f"追跡開始 / {mode} 自動姿勢選択 / SNAPブースト / アクティブアンカー={active_anchor}")


def relock_from_frame(frame_bgr, reference_pose=None, use_rotation_guard=False, reason="manual"):
    """追跡中に 'r' やスロットキーで手動再ロックする。
    'r'（reference_pose=None, ガード無し）: Space/g と同じ GLOBAL 全探索。実物が大きく
        ずれていて生のスロット姿勢＋snap では収束しないときのため。
    スロットキー（reference_pose=そのスロットの姿勢, ガード有り）: そのスロット姿勢を基準に
        近傍だけを探索する。2026-09-09 の録画で、合っている状態でスロット1を押したら
        GLOBAL 探索が 100mm 先の候補を選んで外れたため、スロットキーは「その姿勢へ戻す」
        という操作の意図どおりに動くようにした。
    """
    global pose, selected_pose_slot, snap_frames, active_anchor, prev_output_pose
    prev_output_pose = None   # 跳び量ゲートの基準を無効化
    global lost_streak, last_recovery_time, last_good_pose, last_quality, recovery_escalated
    global quality_smoothed, last_good_time, relock_reason, lost_start_time

    dist_map = compute_edge_distance_map(frame_bgr)
    auto_pose, auto_slot = pose_auto_select(frame_bgr, dist_map, use_rotation_guard=use_rotation_guard,
                                            reference_pose=reference_pose)
    pose = auto_pose
    if auto_slot is not None:
        selected_pose_slot = auto_slot

    active_anchor, pose = snap_to_current_frame(frame_bgr)
    snap_frames = SNAP_FRAME_COUNT

    lost_streak = 0
    lost_start_time = None
    last_recovery_time = time.time()
    last_good_pose = pose.copy()
    last_quality = 1.0
    quality_smoothed = 1.0
    last_good_time = time.time()
    recovery_escalated = False
    pos_filter.x_prev = None
    rot_filter.q_prev = None
    relock_reason = reason
    mode = "GLOBAL" if not use_rotation_guard else "SAFE(基準姿勢の近傍)"
    print(f"手動再ロック({reason}) / {mode} 自動姿勢選択 / SNAPブースト / アクティブアンカー={active_anchor}")


# ─── Unity 通信 ──────────────────────────────────────────────────────────

unity_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if UNITY_UDP_ENABLED else None


last_anchor_quality = {}   # 直近フレームの各アンカー単体適合度（Unity 送信・HUD 用）


def _marker_payload():
    err = globals().get("marker_err")
    registered = marker_ref is not None and bool(marker_ref.T_ref_model)
    n_samples = 0
    if marker_ref is not None and marker_ref.last is not None:
        n_samples = len(marker_ref.samples.get(marker_ref.last["key"], []))
    elif marker_ref is not None:
        n_samples = sum(len(v) for v in marker_ref.samples.values())
    return {
        "marker_registered": registered,
        "marker_visible": marker_ref is not None and marker_ref.last is not None,
        "marker_valid": err is not None,
        "marker_err_mm": float(err["mm"]) if err is not None else 0.0,
        "marker_err_deg": float(err["deg"]) if err is not None else 0.0,
        "marker_samples": int(n_samples),
        "marker_msg": globals().get("marker_status_text", ""),
        # ② 回転不確か警告（統合中の回転固定が長い／大きく動いた／HOLD 中／F 復帰直後）
        "rotation_uncertain": bool(globals().get("rotation_uncertain_reason", "")),
        "rotation_uncertain_reason": globals().get("rotation_uncertain_reason", ""),
    }


def send_pose_to_unity(pose_mat, mode_text):
    if unity_udp_sock is None:
        return
    # active_anchor（Unity が強調表示する歯群）: 階層アンカーでは display_anchor（適合度最良の歯群、
    # または切替先）。pose_anchor は実際に姿勢を出しているアンカー（F = 歯列全体）。
    if SINGLE_BODY_MODE:
        if HIERARCHICAL_ANCHORS:
            unity_anchor = display_anchor if (display_anchor in ("A", "B", "C", "D") or "+" in display_anchor) else "ALL"
        else:
            unity_anchor = unity_display_anchor if unity_display_anchor in ("A", "B", "C", "D") else "ALL"
    else:
        unity_anchor = active_anchor
    payload = json.dumps({
        "name": ANCHOR_BY_KEY[active_anchor]["name"],
        "active_anchor": unity_anchor,
        # 統合で F を離れたが使える歯群がない（保持/予測）間は "hold"（以前は "F" と出ていた）
        "pose_anchor": ("+".join(fusion_keys) if fusion_active and fusion_keys
                        else ("hold" if fusion_off_f else active_anchor)),
        "quality": [round(float(last_anchor_quality.get(k, 0.0)), 3) for k in ("F", "A", "B", "C", "D")],
        "timestamp": time.time(),
        "mode": mode_text,
        "pose_cv": pose_mat.astype(float).reshape(16).tolist(),
        # M3T に渡したのと同じ（歪み補正後の）内部パラメータ。Unity の M3TCameraIntrinsics がこれで投影を
        # 合わせる。以前は Unity 側に手書きの K があり、再キャリブ後に fx 1001.6 vs 1146.6 とずれて
        # 重畳が画像端で 30〜77 px 外れていた（2026-10-09）。
        "K": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        "image_size": [int(w), int(h)],
        # ArUco 基準との実測誤差（fit quality は代理指標なので、精度はこちらで見る）。
        # marker_err はメインループでこの送信の直前に計算される。
        **_marker_payload(),
    }, separators=(",", ":")).encode("utf-8")
    unity_udp_sock.sendto(payload, (UNITY_UDP_HOST, UNITY_UDP_PORT))


command_socket = None
if UNITY_UDP_ENABLED:
    try:
        command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        command_socket.bind((UNITY_UDP_HOST, UNITY_COMMAND_PORT))
        command_socket.setblocking(False)
        print(f"Unity コマンド UDP 待受中: {UNITY_UDP_HOST}:{UNITY_COMMAND_PORT}")
    except OSError as e:
        command_socket = None
        print(f"Unity コマンド UDP 無効: {e}")


def stop_tracking_from_unity():
    global pose, tracking_active
    if tracking_active:
        pose = ANCHOR_BY_KEY[active_anchor]["model"].pose_cv.copy().astype(np.float32)
    tracking_active = False
    print("ADJUST モード / Unity から停止")


def _rot_axis_matrix(axis, deg):
    v = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis]
    R, _ = cv2.Rodrigues(np.array(v, np.float64) * np.deg2rad(deg))
    return R.astype(np.float32)


marker_status_text = ""   # 直近のマーカー登録操作の結果（Unity パネルに表示する）


def register_marker_sample():
    """'m' キー / Unity の marker_register。今の pose を正としてマーカー→モデルの変換を 1 サンプル登録する。
    誤登録を防ぐ安全チェックは従来の 'm' キーのものと同じ。"""
    global marker_status_text
    if marker_ref is None:
        msg = "マーカー基準は無効です（MARKER_REF_ENABLED=False）"
    elif marker_ref.last is None:
        msg = "マーカーが検出されていないため登録できません"
    elif tracking_active and (time.time() - last_recovery_time < 3.0 or lost_streak > 0):
        # 再ロック直後は quality_smoothed が 1.0 に初期化されていて低品質判定をすり抜ける。
        # 2026-09-21 19:02 に HOLD/SNAP ループ中の姿勢が 2 件登録された事故があったため拒否する。
        msg = "再ロック直後 / 品質低下中は登録できません。数秒安定してから押してください"
    elif tracking_active and snap_frames > 0:
        # 2026-09-09: 自動リカバリ中（SNAP）にズレた姿勢を登録してしまった事故があったため拒否する
        msg = "SNAP（再ロック）中は登録できません。追跡が落ち着いて合っているのを確認してから押してください"
    elif tracking_active and last_quality < QUALITY_LOSS_THRESHOLD:
        msg = (f"品質が低い（{last_quality:.2f} < {QUALITY_LOSS_THRESHOLD}）ため登録しません。"
               "PnP クリック初期化の直後（ADJUST のまま）に押すのが確実です")
    else:
        k = marker_ref.register(pose)
        s = marker_ref.summary(k)
        msg = (f"マーカー基準にサンプル追加: {k}（{marker_ref.last['n_markers']} 枚, 再投影 {marker_ref.last['rms_px']:.2f}px）"
               f" → 登録 {s['n']} 件中 {s['n_used']} 件（外れ値除外後）の平均を基準に使用。"
               f" ばらつき: RMS {s['rms_mm']:.1f} mm / {s['rms_deg']:.1f} deg, 最大 {s['max_mm']:.1f} mm / {s['max_deg']:.1f} deg")
        if s["n"] < 3:
            msg += "  → 視点・距離を変えてあと数回登録すると偶発誤差が平均化されます（取り消しは直近 1 件ずつ）"
    print(msg)
    marker_status_text = msg


def undo_marker_sample():
    """Backspace / Unity の marker_undo。直近の登録サンプルを 1 件取り消す。"""
    global marker_status_text
    if marker_ref is None:
        return
    k = marker_ref.undo_last_sample()
    s = marker_ref.summary(k) if k else None
    msg = f"マーカー基準サンプルを 1 件取り消し: {k} → 残り {s['n'] if s else 0} 件" if k else "取り消せるサンプルがありません"
    print(msg)
    marker_status_text = msg


# ─── 録画（オフライン調整用，2026-10-21） ───────────────────────────────────────
# experiment/record.py が UDP（コマンドポート）で record_start:<名前> / event:<文字列> / record_stop を送る。
# 追跡に渡す前の生フレーム（歪み補正前）を experiment/sessions/<名前>/video.avi（MJPG）に、各フレームの時刻を frames.csv に、
# 録画開始時の出力姿勢を start_pose.npy に保存する。再生（M3T_REPLAY_CLIP）はこの姿勢から始めるので、
# 録画は F で安定して追跡しているときだけ始める（遮蔽前に 2 s 以上 F で追える区間を入れること: PnP のキーフレームと
# F の見張りは F 追跡 0.3 s 後から作られる）。
EXPERIMENT_DIR = os.path.join(SCRIPT_DIR, "experiment")
RECORD_FPS_NOMINAL = 20.0     # AVI ヘッダ用の名目値。再生は frames.csv の実時刻で進むので結果には影響しない


class SessionRecorder:
    def __init__(self):
        self.q = _queue.Queue(maxsize=600)
        self.active = False
        self.clip_dir = None
        self.n = 0
        self.dropped = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def start(self, clip_dir, meta):
        if self.active:
            self.stop()
        os.makedirs(clip_dir, exist_ok=True)
        self.clip_dir = clip_dir
        self.n = 0
        self.dropped = 0
        self.active = True
        self.q.put(("start", clip_dir, meta))

    def push(self, raw, ts, start_pose):
        if not self.active:
            return
        if self.n == 0:
            self.q.put(("start_pose", np.asarray(start_pose, np.float32).copy()))
        try:
            self.q.put_nowait(("frame", raw, ts, self.n))
            self.n += 1
        except _queue.Full:
            self.dropped += 1   # 書き出しが追いつかないとき（その分は frames.csv にも載らない）

    def event(self, text, ts):
        if self.active:
            self.q.put(("event", self.n, ts, text))

    def stop(self):
        if self.active:
            self.active = False
            self.q.put(("stop", self.n, self.dropped))

    def _loop(self):
        writer = frames_f = events_f = frames_w = events_w = None
        clip_dir = None
        while True:
            item = self.q.get()
            kind = item[0]
            try:
                if kind == "start":
                    clip_dir, meta = item[1], item[2]
                    with open(os.path.join(clip_dir, "meta.json"), "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=1)
                    frames_f = open(os.path.join(clip_dir, "frames.csv"), "w", newline="", encoding="utf-8")
                    frames_w = csv.writer(frames_f)
                    frames_w.writerow(["index", "timestamp"])
                    events_f = open(os.path.join(clip_dir, "events.csv"), "w", newline="", encoding="utf-8")
                    events_w = csv.writer(events_f)
                    events_w.writerow(["frame_index", "timestamp", "event"])
                    writer = None
                elif kind == "start_pose" and clip_dir:
                    np.save(os.path.join(clip_dir, "start_pose.npy"), item[1])
                elif kind == "frame" and frames_w is not None:
                    raw, ts, idx = item[1], item[2], item[3]
                    if writer is None:
                        hh, ww = raw.shape[:2]
                        writer = cv2.VideoWriter(os.path.join(clip_dir, "video.avi"),
                                                 cv2.VideoWriter_fourcc(*"MJPG"), RECORD_FPS_NOMINAL, (ww, hh))
                        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
                    writer.write(raw)
                    frames_w.writerow([idx, f"{ts:.6f}"])
                elif kind == "event" and events_w is not None:
                    events_w.writerow([item[1], f"{item[2]:.6f}", item[3]])
                    events_f.flush()
                elif kind == "stop":
                    if writer is not None:
                        writer.release()
                    for f in (frames_f, events_f):
                        if f is not None:
                            f.close()
                    print(f"[record] 保存: {clip_dir}（{item[1]} フレーム、取りこぼし {item[2]}）")
                    writer = frames_f = events_f = frames_w = events_w = None
                    clip_dir = None
            except Exception as e:
                print(f"[record] 書き出しエラー: {e}")


recorder = None if REPLAY else SessionRecorder()


def handle_record_command(command, reply):
    """録画の UDP コマンド。reply(text) で送信元に結果を返す（experiment/record.py が確認に使う）。"""
    if command.startswith("record_start:"):
        rel = command[len("record_start:"):].strip().replace("..", "_")
        if not tracking_active:
            reply("error:not_tracking")
            print("[record] 追跡中でないため録画を開始しません（位置合わせして追跡を開始してから）")
            return True
        # 再生は録画開始時の姿勢から始まるので、F で安定して追えているときだけ始める
        if snap_frames > 0 or fusion_off_f or quality_smoothed < QUALITY_GOOD_THRESHOLD:
            reply(f"error:not_stable:q={quality_smoothed:.2f}:fusion={int(fusion_off_f)}:snap={int(snap_frames > 0)}")
            print(f"[record] 追跡が安定していないため録画を開始しません（q={quality_smoothed:.2f}, "
                  f"統合={int(fusion_off_f)}, SNAP={int(snap_frames > 0)}）")
            return True
        clip_dir = os.path.join(EXPERIMENT_DIR, "sessions", rel)
        recorder.start(clip_dir, {
            "clip": rel,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_size": [int(w), int(h)],
            "calibration": calibration_path,
            "marker_registration": os.path.join(pose_slots_dir, "marker_registration.json"),
            "commit_note": "tracker_four_group.py (main + ⑪⑫)",
        })
        try:   # 再生は録画時点のマーカー登録で評価する（後で登録し直しても比べられるように）
            import shutil
            shutil.copy(os.path.join(pose_slots_dir, "marker_registration.json"), clip_dir)
        except OSError:
            pass
        reply("ok:record_start")
        print(f"[record] 開始: {clip_dir}")
        return True
    if command == "record_stop":
        recorder.stop()
        reply("ok:record_stop")
        return True
    if command.startswith("event:"):
        recorder.event(command[len("event:"):].strip(), time.time())
        reply("ok:event")
        return True
    if command == "status":
        reply(f"status:tracking={int(tracking_active)}:recording={int(recorder.active)}"
              f":quality={quality_smoothed:.2f}:fusion={int(fusion_off_f)}:snap={int(snap_frames > 0)}")
        return True
    return False


def poll_unity_commands(frame_bgr):
    """Unity からの UDP コマンド。デモは Unity の UI だけで初期位置合わせ〜追跡開始ができるように、
    OpenCV ウィンドウのキー操作に対応するコマンドを一通り受ける:
      start / global_start / stop / quit
      auto_align        : ADJUST 中に保存スロットから全探索して姿勢を置く（追跡は開始しない）
      start_from_pose   : 今の姿勢から追跡開始（'f' キー相当）
      load_pose:N / save_pose:N
      adjust_t:x|y|z:メートル / adjust_r:x|y|z:度   （ADJUST 中の手動微調整）
      anchor_override:A〜D / 空 = Auto
    """
    global pose, selected_pose_slot
    if command_socket is None:
        return
    while True:
        try:
            data, sender = command_socket.recvfrom(1024)
        except BlockingIOError:
            break
        except ConnectionResetError:
            continue   # Windows: 直前に返信した相手がもういないと次の recv がこれを出す
        except OSError as e:
            print(f"Unity コマンド UDP エラー: {e}")
            break

        command = data.decode("utf-8", errors="ignore").strip()
        cmd_lower = command.lower()

        def _reply(text, _to=sender):
            try:
                command_socket.sendto(text.encode("utf-8"), _to)
            except OSError:
                pass

        if recorder is not None and handle_record_command(command, _reply):
            continue
        if cmd_lower == "start":
            if not tracking_active:
                start_tracking_from_frame(frame_bgr, use_rotation_guard=True)
        elif cmd_lower == "global_start":
            if not tracking_active:
                start_tracking_from_frame(frame_bgr, use_rotation_guard=False)
        elif cmd_lower == "stop":
            stop_tracking_from_unity()
        elif cmd_lower == "quit":
            raise KeyboardInterrupt
        elif cmd_lower == "auto_align":
            if tracking_active:
                print("auto_align は ADJUST 中のみ（追跡中はスロットキー相当の再ロックを使う）")
            else:
                t_align = time.perf_counter()
                dm = compute_edge_distance_map(frame_bgr)
                p, slot = pose_auto_select(frame_bgr, dm, use_rotation_guard=False,
                                           translation_search=True)
                pose = p
                reset_all_anchors(pose)
                if slot is not None:
                    selected_pose_slot = slot
                print(f"Unity: auto_align 完了（スロット {slot} 近傍、{time.perf_counter() - t_align:.1f}s）。"
                      "位置を確認して start_from_pose へ")
        elif cmd_lower == "start_from_pose":
            if not tracking_active:
                start_tracking_from_current_pose(frame_bgr)
        elif command.startswith("load_pose:"):
            try:
                slot = int(command.split(":", 1)[1])
            except ValueError:
                slot = selected_pose_slot
            loaded, path = load_pose_slot(slot)
            if loaded is not None:
                pose = loaded
                selected_pose_slot = slot
                reset_all_anchors(pose)
                if tracking_active:
                    relock_from_frame(frame_bgr, reference_pose=loaded, use_rotation_guard=True, reason="slot")
                print(f"Unity: スロット {slot} を読み込み")
            else:
                print(f"Unity: スロット {slot} は未保存")
        elif command.startswith("save_pose:"):
            try:
                slot = int(command.split(":", 1)[1])
            except ValueError:
                slot = selected_pose_slot
            save_pose_slot(slot, pose)
            print(f"Unity: 現在姿勢をスロット {slot} に保存")
        elif cmd_lower == "marker_register":   # 'm' キー相当（安全チェック込み）
            register_marker_sample()
        elif cmd_lower == "marker_undo":       # Backspace 相当。全消去は誤操作防止のため 'x' キーのみ
            undo_marker_sample()
        elif command.startswith("adjust_t:") or command.startswith("adjust_r:"):
            parts = command.split(":")
            if len(parts) == 3 and parts[1].lower() in ("x", "y", "z"):
                axis = parts[1].lower()
                try:
                    amount = float(parts[2])
                except ValueError:
                    amount = 0.0
                if tracking_active:
                    print("adjust_* は ADJUST 中のみ有効")
                elif parts[0] == "adjust_t":
                    pose["xyz".index(axis), 3] += amount
                    reset_all_anchors(pose)
                else:
                    pose[:3, :3] = _rot_axis_matrix(axis, amount) @ pose[:3, :3]
                    reset_all_anchors(pose)
        elif command.startswith("anchor_switching:"):
            global anchor_switching_enabled, active_anchor, anchor_switch_pose
            global anchor_fusion_enabled, fusion_active, fusion_keys
            global fusion_off_f, fusion_members, fusion_enter_streak, fusion_hold_pose
            global fusion_f_ok_since, fusion_no_member_since
            val = command[len("anchor_switching:"):].strip().lower()
            anchor_switching_enabled = val not in ("off", "0", "false")
            anchor_fusion_enabled = val in ("fusion", "fuse", "all")
            if not anchor_switching_enabled or anchor_fusion_enabled:
                active_anchor = "F"
                anchor_switch_pose = None
            fusion_active = False
            fusion_keys = []
            fusion_off_f = False
            fusion_members = {}
            fusion_enter_streak = {}
            fusion_hold_pose = None
            print("Anchor Switching: " + ("OFF（比較用ベースライン: F のみ）" if not anchor_switching_enabled
                                          else ("FUSION（見えている歯群を統合）" if anchor_fusion_enabled else "ON（最良 1 歯群を選択）")))
        elif command.startswith("anchor_override:"):
            global manual_anchor_override, unity_display_anchor
            key = command[len("anchor_override:"):].strip().upper()
            if SINGLE_BODY_MODE and not HIERARCHICAL_ANCHORS:
                # 追跡は 1 ボディなので切替対象は無い。Unity の強調表示用にキーだけ記録する
                # （manual_anchor_override に "B" 等を入れると ANCHORS に無いキーで落ちるので別変数）。
                unity_display_anchor = key if key in ("A", "B", "C", "D") else ""
                print(f"Unity アンカー表示指定: {unity_display_anchor or 'ALL'}（単体ボディモードでは表示のみ）")
                continue
            if key in ANCHOR_BY_KEY:
                manual_anchor_override = key
                print(f"Unity アンカーオーバーライド: {key}")
            else:
                manual_anchor_override = ""
                print(f"Unity アンカーオーバーライド解除（Auto）")
        elif command:
            print(f"未知の Unity コマンド: {command}")


video_clients_lock = threading.Lock()
video_clients: list[socket.socket] = []
last_video_send_time = 0.0


# JPEG エンコードと TCP 送信は別スレッドで行う（720p の JPEG 化は 15〜25 ms あり、
# メインループで毎回やると fps とラグに直結する）。メインループは最新フレームを置くだけ。
_video_pending = {"frame": None}
_video_cond = threading.Condition()


def send_video_frame_to_unity(frame_bgr):
    global last_video_send_time
    if not UNITY_VIDEO_ENABLED:
        return
    now = time.time()
    if now - last_video_send_time < 1.0 / UNITY_VIDEO_FPS:
        return
    with video_clients_lock:
        if not video_clients:
            return
    last_video_send_time = now
    with _video_cond:
        _video_pending["frame"] = frame_bgr  # 古いフレームが未処理なら捨てて最新に置き換える
        _video_cond.notify()


def unity_video_encoder_loop():
    while True:
        with _video_cond:
            while _video_pending["frame"] is None:
                _video_cond.wait()
            frame_bgr = _video_pending["frame"]
            _video_pending["frame"] = None
        video_frame = frame_bgr
        if UNITY_VIDEO_WIDTH > 0 and frame_bgr.shape[1] > UNITY_VIDEO_WIDTH:
            scale = UNITY_VIDEO_WIDTH / frame_bgr.shape[1]
            video_frame = cv2.resize(frame_bgr, (UNITY_VIDEO_WIDTH, int(frame_bgr.shape[0] * scale)),
                                      interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", video_frame, [int(cv2.IMWRITE_JPEG_QUALITY), UNITY_VIDEO_JPEG_QUALITY])
        if not ok:
            continue
        payload = encoded.tobytes()
        header = struct.pack(">I", len(payload))
        dead = []
        with video_clients_lock:
            for client in video_clients:
                try:
                    client.sendall(header)
                    client.sendall(payload)
                except OSError:
                    dead.append(client)
            for client in dead:
                try:
                    client.close()
                except OSError:
                    pass
                video_clients.remove(client)


def unity_video_server_loop():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((UNITY_UDP_HOST, UNITY_VIDEO_PORT))
        server.listen(2)
        server.settimeout(0.5)
        print(f"Unity 動画 TCP 待受中: {UNITY_UDP_HOST}:{UNITY_VIDEO_PORT}")
        while True:
            try:
                client, _ = server.accept()
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with video_clients_lock:
                    video_clients.append(client)
                print("Unity 動画クライアント接続")
            except socket.timeout:
                continue
            except OSError as e:
                print(f"Unity 動画 TCP 停止: {e}")
                break
    except OSError as e:
        print(f"Unity 動画 TCP 無効: {e}")
    finally:
        try:
            server.close()
        except OSError:
            pass


if UNITY_VIDEO_ENABLED:
    threading.Thread(target=unity_video_server_loop, daemon=True).start()
    threading.Thread(target=unity_video_encoder_loop, daemon=True).start()


# ─── CSV ロガー ──────────────────────────────────────────────────────────

os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"tracking_{time.strftime('%Y%m%d_%H%M%S')}.csv")
if REPLAY:
    log_path = os.environ.get("M3T_REPLAY_LOG") or os.path.join(REPLAY_CLIP_DIR, "replay.csv")
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
log_file = open(log_path, "w", newline="", encoding="utf-8")
log_writer = csv.writer(log_file)
log_writer.writerow([
    "frame", "timestamp", "mode", "quality",
    "tx", "ty", "tz",
    "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22",
    "qA", "qB", "qC", "qD", "qF",   # 各アンカー単体の品質（quality 列は EMA 平滑化した剛体整合スコア）。F = 歯列全体
    "quality_raw",            # 平滑化前の剛体整合スコア
    # マーカー基準（登録済みのときだけ値が入る）: 追跡姿勢との差と、基準の姿勢そのもの
    "ref_key", "err_mm", "err_deg",
    "ref_tx", "ref_ty", "ref_tz", "ref_rx", "ref_ry", "ref_rz",
    "ref_n_samples",          # 基準（マーカー→モデル変換）の登録サンプル数（平均化に使った件数）
    "anchor_switching",       # 2 = 統合（見えている歯群すべて）、1 = 最良 1 歯群を選択、0 = 比較用ベースライン（F のみ）
    # ⑤ 歯群ごとの回転精度（マーカー登録済みのときだけ）: 各歯群ボディ自身の姿勢とマーカー予測姿勢の差。
    #   g?_rot_deg = 回転差、g?_cen_mm = 歯群重心の 3D 差、g?_cen_depth_mm = そのうちカメラ光線方向の成分
    "gA_rot_deg", "gA_cen_mm", "gA_cen_depth_mm",
    "gB_rot_deg", "gB_cen_mm", "gB_cen_depth_mm",
    "gD_rot_deg", "gD_cen_mm", "gD_cen_depth_mm",
    "gF_rot_deg",             # F ボディ（統合中は統合姿勢＝固定回転）の回転差
    "event",                  # 特記事項（復帰時回転再探索の結果など）
    "rot_uncertain",          # ② 「回転不確か」警告が出ていたフレームは 1
    # 歯群ボディ自身の回転と出力回転の差（マーカーなしでも取れる）。F 追跡中は「歯群の回転が F とどれだけ合うか」、
    # 統合中は「追従先の歯群と出力の残差」になる
    "gA_vs_out_deg", "gB_vs_out_deg", "gD_vs_out_deg",
    "fus_rot_src",            # 統合中に回転を追従している歯群（空 = 固定 / 統合でない）。"(cap)" 付きは累計上限で停止
    "fus_rot_deg",            # 統合中の、離脱時の回転からの累計回転量 [deg]
    "gF_cen_mm",              # 案 E 用: F ボディ自身（統合中も追跡している生の姿勢）の歯列重心の 3D 誤差 [mm]。qF と組で見る
    "fill_q",                 # 歯色マスクに乗っている F 内部点の割合（EMA、F 追跡中のみ）。誤ロック検知のしきい値調整用
    "fus_motion_px",          # 統合中、回転源の投影範囲の基準フレームからの特徴点移動量 [px, 縮小画像]（回転追従ゲート用）
    "fus_pnp_n",              # 統合中、キーフレーム特徴点 PnP に使えた点数
    "fus_pnp_std_deg",        # その回転の 1σ（最大固有値）[deg]
    "fus_pnp_deg",            # キーフレームからの回転量 [deg]（ZUPT しきい値と比べる）
    "fus_pnp_disp_px",        # 生存点のキーフレームからの移動量の中央値 [px, 縮小画像]（静止判定）
])

diag_frame_dir = None
if DIAG_FRAME_DUMP_EVERY > 0:
    diag_frame_dir = log_path[:-4] + "_frames"
    os.makedirs(diag_frame_dir, exist_ok=True)


def log_frame(frame_idx, mode_text, quality, pose_mat, quality_by_key=None, quality_raw_value=float("nan"),
              marker_err=None, group_poses=None, event="", rot_uncertain=0, fus_rot_src="", fus_rot_deg=None,
              fill_q=None, fus_motion_px=None, fus_pnp=None):
    R, t = pose_mat[:3, :3], pose_mat[:3, 3]
    qk = quality_by_key or {}
    gcols = [""] * 10
    vs_out = [""] * 3
    gF_cen = ""
    if group_poses:
        vs_out = [f"{rotation_angle_deg(group_poses[k], pose_mat):.3f}" if group_poses.get(k) is not None else ""
                  for k in ("A", "B", "D")]
        if marker_err is not None and group_poses.get("F") is not None:
            _P = marker_err["predicted"]
            gF_cen = f"{float(np.linalg.norm(_group_center_cam('F', group_poses['F']) - _group_center_cam('F', _P))) * 1000.0:.3f}"
    if marker_err is not None:
        P = marker_err["predicted"]
        rv, _ = cv2.Rodrigues(P[:3, :3].astype(np.float64))
        ref_cols = [marker_err["key"], f"{marker_err['mm']:.3f}", f"{marker_err['deg']:.3f}",
                    f"{P[0,3]:.6f}", f"{P[1,3]:.6f}", f"{P[2,3]:.6f}",
                    f"{rv[0,0]:.6f}", f"{rv[1,0]:.6f}", f"{rv[2,0]:.6f}",
                    str(marker_err.get("n_samples", ""))]
        if group_poses:
            # ⑤ 歯群ボディ自身の姿勢（統合では重心位置だけを使い回転は捨てている）がどれだけ正しいかを記録する。
            gcols = []
            for k in ("A", "B", "D"):
                pk = group_poses.get(k)
                if pk is None:
                    gcols += ["", "", ""]
                    continue
                c = ANCHOR_BY_KEY[k]["centroid"]
                c_trk = pk[:3, :3] @ c + pk[:3, 3]
                c_prd = P[:3, :3] @ c + P[:3, 3]
                d = c_trk - c_prd
                ray = c_prd / max(float(np.linalg.norm(c_prd)), 1e-9)
                gcols += [f"{rotation_angle_deg(pk, P):.3f}",
                          f"{float(np.linalg.norm(d)) * 1000.0:.3f}",
                          f"{float(np.dot(d, ray)) * 1000.0:.3f}"]
            pf = group_poses.get("F")
            gcols.append(f"{rotation_angle_deg(pf, P):.3f}" if pf is not None else "")
    else:
        ref_cols = ["", "", "", "", "", "", "", "", "", ""]
    log_writer.writerow([
        frame_idx, f"{time.time():.6f}", mode_text, f"{quality:.4f}",
        f"{t[0]:.6f}", f"{t[1]:.6f}", f"{t[2]:.6f}",
        f"{R[0,0]:.6f}", f"{R[0,1]:.6f}", f"{R[0,2]:.6f}",
        f"{R[1,0]:.6f}", f"{R[1,1]:.6f}", f"{R[1,2]:.6f}",
        f"{R[2,0]:.6f}", f"{R[2,1]:.6f}", f"{R[2,2]:.6f}",
        *(f"{qk.get(k, float('nan')):.4f}" for k in ("A", "B", "C", "D", "F")),
        f"{quality_raw_value:.4f}",
        *ref_cols,
        ("2" if anchor_fusion_enabled else "1") if anchor_switching_enabled else "0",
        *gcols,
        event,
        rot_uncertain,
        *vs_out,
        fus_rot_src,
        "" if fus_rot_deg is None else f"{fus_rot_deg:.3f}",
        gF_cen,
        "" if fill_q is None else f"{fill_q:.3f}",
        "" if fus_motion_px is None else f"{fus_motion_px:.3f}",
        "" if fus_pnp is None else fus_pnp["n"],
        "" if fus_pnp is None else f"{fus_pnp['std_deg']:.3f}",
        "" if fus_pnp is None else f"{fus_pnp['deg']:.3f}",
        "" if fus_pnp is None else f"{fus_pnp['disp_px']:.3f}",
    ])


# ─── メインループ ─────────────────────────────────────────────────────────

tracking_active = False
view_mode = 0
snap_frames = 0
frame_idx = 0
frame_read_failures = 0

click_init_active = False
click_points = []


def on_mouse(event, x, y, flags, param):
    global click_points
    if click_init_active and event == cv2.EVENT_LBUTTONDOWN:
        if len(click_points) < len(LANDMARK_POINTS_MODEL):
            click_points.append((float(x), float(y)))


def solve_pnp_from_clicks():
    """クリックされた画像点とランドマーク点からPnPで姿勢を求める。
    6点以上ある場合はsolvePnPRansacを使い、クリックミスした点があっても
    外れ値として自動的に除外できるようにする（OpenCV公式tutorial_py_poseの
    構成を参考）。4〜5点しかない場合はRANSACには不足なのでP3P/AP3P+depth
    フィルタにフォールバックする。"""
    obj_pts = LANDMARK_POINTS_MODEL.astype(np.float64)
    img_pts = np.array(click_points, dtype=np.float64)

    if len(click_points) >= 6:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, K.astype(np.float64), None,
            reprojectionError=8.0, confidence=0.99, iterationsCount=200,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None
        n_in = 0 if inliers is None else len(inliers)
        print(f"solvePnPRansac: {n_in}/{len(click_points)} 点を採用（外れ値を自動除外）")
    else:
        flags = cv2.SOLVEPNP_AP3P if len(click_points) == 4 else cv2.SOLVEPNP_ITERATIVE
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj_pts.reshape(-1, 1, 3), img_pts.reshape(-1, 1, 2),
            K.astype(np.float64), None, flags=flags)
        if not ok:
            return None, None
        best = None
        for rv, tv in zip(rvecs, tvecs):
            z = float(tv[2, 0])
            if 0.05 <= z <= 1.0:
                best = (rv, tv)
                break
        if best is None:
            best = (rvecs[0], tvecs[0])
        rvec, tvec = best

    R_mat, _ = cv2.Rodrigues(rvec)
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = R_mat
    result[:3, 3] = tvec.reshape(3)

    obj_reproj = (R_mat @ LANDMARK_POINTS_MODEL.T).T + tvec.reshape(3)
    xs = K[0, 0] * obj_reproj[:, 0] / obj_reproj[:, 2] + K[0, 2]
    ys = K[1, 1] * obj_reproj[:, 1] / obj_reproj[:, 2] + K[1, 2]
    reproj = np.stack([xs, ys], axis=1)
    err = float(np.mean(np.linalg.norm(reproj - np.array(click_points), axis=1)))
    return result, err


print(f"ログ: {log_path}")
print("Anchor Switching: " + ("OFF（比較用ベースライン: F のみ。M3T_ANCHOR_SWITCHING=0）" if not ANCHOR_SWITCHING_DEFAULT
                               else ("FUSION（見えている歯群を統合。M3T_ANCHOR_FUSION=0 で単一選択）" if ANCHOR_FUSION_DEFAULT else "ON（最良 1 歯群を選択）")))
print("キー: Space=通常追跡開始(保護あり)/停止  g=全探索で追跡開始  ESC=終了")
print("      1〜9,0,-,= : 姿勢スロット読込  p=保存  r=選択スロットで再ロック  v=表示切替  t=自動再ロックON/OFF")
print("      a/d=X移動  q/e=Y移動  w/s=Z(奥行き)移動  i/k,j/l,u/o=回転（追跡停止中のみ）")
print(f"      c=クリック位置合わせ開始（{len(LANDMARK_POINTS_MODEL)}点をクリックするとsolvePnPで自動計算、追跡停止中のみ）")
print(f"      n=良姿勢の自動登録ON/OFF（デフォルトON。良品質が{AUTO_CAPTURE_MIN_GOOD_STREAK}フレーム続くとスロット"
      f"{AUTO_CAPTURE_SLOTS[0]}〜{AUTO_CAPTURE_SLOTS[-1]}に自動保存、スロット1は保護）")

# Unity から起動されたとき（環境変数 M3T_HEADLESS=1）は OpenCV ウィンドウを出さない。
# 2026-10-01 計測: cv2.waitKey(1) だけで 32 ms、HUD 描画 13 ms、レンダ 5 ms = 1 フレーム 85 ms のうち 50 ms が
# 表示系だった（Unity 側に映像と品質バーがあるので不要）。キー操作は Unity のコマンドで代替する。
HEADLESS = os.environ.get("M3T_HEADLESS", "0") == "1"
_HEADLESS_SHOW = np.zeros((4, 4, 3), dtype=np.uint8)   # HUD の putText を素通しさせるためのダミー画像
if HEADLESS:
    print("ヘッドレスモード: OpenCV ウィンドウなし（M3T_HEADLESS=1）")
else:
    cv2.namedWindow("few_teeth_tracking", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("few_teeth_tracking", on_mouse)
    cv2.namedWindow("reference", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("reference", on_mouse_reference)
    cv2.imshow("reference", _render_reference())


# ─── 簡易プロファイラ ───────────────────────────────────────────────────────
# 各段階の所要時間を集計し、PROFILE_EVERY_N フレームごとに平均を、1 フレームが
# PROFILE_SLOW_FRAME_MS を超えたときはその内訳を表示する（ラグ・固まりの切り分け用）。
PROFILE_ENABLED = True
PROFILE_EVERY_N = 100
PROFILE_SLOW_FRAME_MS = 250.0
_pf_acc = {}
_pf_cur = {}
_pf_last = 0.0
_pf_frame_start = 0.0
_pf_n = 0


def _pf_begin():
    global _pf_last, _pf_frame_start
    _pf_last = _pf_frame_start = time.perf_counter()
    _pf_cur.clear()


def _pf_mark(name):
    global _pf_last
    now = time.perf_counter()
    _pf_cur[name] = _pf_cur.get(name, 0.0) + (now - _pf_last) * 1000.0
    _pf_last = now


def _pf_end():
    global _pf_n
    if not PROFILE_ENABLED:
        return
    _pf_mark("other")
    total = (time.perf_counter() - _pf_frame_start) * 1000.0
    for k, v in _pf_cur.items():
        _pf_acc[k] = _pf_acc.get(k, 0.0) + v
    _pf_acc["_total"] = _pf_acc.get("_total", 0.0) + total
    _pf_n += 1
    if total > PROFILE_SLOW_FRAME_MS:
        print(f"[profile] 遅いフレーム {total:.0f} ms: " + "  ".join(f"{k}={v:.0f}" for k, v in _pf_cur.items()))
    if _pf_n >= PROFILE_EVERY_N:
        avg_total = _pf_acc["_total"] / _pf_n
        parts = "  ".join(f"{k}={v/_pf_n:.1f}" for k, v in _pf_acc.items() if k != "_total")
        print(f"[profile] 平均 {avg_total:.1f} ms/frame ({1000.0/avg_total:.1f} fps): {parts}")
        _pf_acc.clear()
        _pf_n = 0


while True:
    _pf_begin()
    ret, raw, _frame_seq = _grabber.read(_frame_seq)   # 最新フレームのみ（別スレッド読み取り）
    if not ret and REPLAY:
        print("[replay] クリップの再生が終わりました")
        break
    if ret and recorder is not None and recorder.active:
        recorder.push(raw, time.time(), pose)   # pose = 直前フレームの出力（再生はここから始める）
    if not ret:
        frame_read_failures += 1
        if frame_read_failures == 1:
            print("カメラ読み取り失敗。再試行中...")
        if frame_read_failures >= FRAME_READ_RETRY_LIMIT:
            print("カメラ読み取りが繰り返し失敗したため終了します")
            break
        time.sleep(0.05)
        continue
    if frame_read_failures > 0:
        print("カメラ読み取り復帰")
        frame_read_failures = 0

    frame_img = prepare_frame(raw)
    _pf_mark("capture")
    frame_raw = frame_img          # 保存・マーカー検出用の未加工フレーム
    marker_err = None
    frame_event = ""
    if marker_ref is not None:
        marker_ref.detect(frame_raw)
        if MARKER_MASK_FOR_TRACKER and marker_ref.last is not None:
            # 以降の追跡・品質評価・表示はボードを消した画像で行う（基準が追跡を汚さないように）
            frame_img = marker_ref.mask_out(frame_raw, pad_px=MARKER_MASK_PAD_PX, method=MARKER_MASK_METHOD)
    _pf_mark("marker")
    if REPLAY and frame_idx == 0:
        # 実機では録画開始時点で F の追跡が続いていたので、SNAP（再探索）はせずその姿勢から続ける
        pose = np.load(os.path.join(REPLAY_CLIP_DIR, "start_pose.npy")).astype(np.float32)
        start_tracking_from_current_pose(frame_img, light_snap_steps=0)
    frame_idx += 1

    # 映像は姿勢と同時に送る（ループの最後、send_pose_to_unity の直前）。以前はここで送っていて、Unity には
    # 新しい映像が先に届き、その映像の姿勢は追跡・品質評価の後（約 50 ms 後）に届くので、動かすとモデルが映像より
    # 1 フレーム遅れて見えた（2026-10-20 「動かすと追従が遅れる」）。
    poll_unity_commands(frame_img)
    _pf_mark("poll")

    key = 255 if HEADLESS else (cv2.waitKey(1) & 0xFF)
    _pf_mark("waitkey")
    step_t = 0.005
    step_r = 2.0

    if key == 27:  # ESC
        break
    elif key == ord(" "):
        if tracking_active:
            stop_tracking_from_unity()
        else:
            start_tracking_from_frame(frame_img, use_rotation_guard=True)
    elif key == ord("g"):
        start_tracking_from_frame(frame_img, use_rotation_guard=False)
    elif key == ord("v"):
        view_mode = (view_mode + 1) % len(VIEW_MODES)
    elif key == ord("p"):
        saved_path = save_pose_slot(selected_pose_slot, pose)
        print(f"姿勢スロット {selected_pose_slot} を保存: {saved_path}")
    elif key == ord("t"):
        QUALITY_CHECK_ENABLED = not QUALITY_CHECK_ENABLED
        lost_streak = 0
        print(f"自動再ロック: {'ON' if QUALITY_CHECK_ENABLED else 'OFF'}")
    elif key == ord("n"):
        AUTO_POSE_CAPTURE_ENABLED = not AUTO_POSE_CAPTURE_ENABLED
        good_streak = 0
        print(f"良姿勢の自動登録: {'ON' if AUTO_POSE_CAPTURE_ENABLED else 'OFF'}"
              f"（スロット{AUTO_CAPTURE_SLOTS[0]}〜{AUTO_CAPTURE_SLOTS[-1]}をラウンドロビンで使用）")
    elif not tracking_active and key == ord("c"):
        if len(LANDMARK_POINTS_MODEL) < 4:
            print("ランドマークが4点未満のため、クリック位置合わせは使えません（手動調整を使ってください）")
        else:
            click_init_active = not click_init_active
            click_points = []
            print(f"クリック位置合わせ: {'開始' if click_init_active else '中止'}"
                  + (f"（'reference'ウィンドウの番号1〜{len(LANDMARK_POINTS_MODEL)}を見て、"
                     "対応する実物の点をカメラ映像側で順にクリック）" if click_init_active else ""))

    if click_init_active and len(click_points) == len(LANDMARK_POINTS_MODEL):
        new_pose, reproj_err = solve_pnp_from_clicks()
        click_init_active = False
        click_points = []
        if new_pose is not None:
            pose = new_pose
            print(f"solvePnP完了: 再投影誤差 {reproj_err:.1f}px"
                  " → 'f'=この姿勢から追跡開始（固定カメラ推奨）"
                  " | wasd/ijkl=手動微調整後にfキー | p=保存")
        else:
            print("solvePnP失敗。クリック位置をやり直してください（cキー）")
    elif key == ord("["):
        current_exposure -= 1
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
        print(f"露出を下げました: 要求={current_exposure} 実際={cap.get(cv2.CAP_PROP_EXPOSURE)}")
    elif key == ord("]"):
        current_exposure += 1
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
        print(f"露出を上げました: 要求={current_exposure} 実際={cap.get(cv2.CAP_PROP_EXPOSURE)}")
    elif not tracking_active and key == ord("f"):
        # 現在の pose（PnP結果や手動微調整後）から直接追跡開始。
        # Space と違い pose_auto_select を呼ばないため固定カメラ環境に適する。
        start_tracking_from_current_pose(frame_img, light_snap_steps=5)
    # 手動ナッジ中は reset_pose_cv を呼ばない。毎回モデル側に反映しようと
    # するとレンダラの内部状態と衝突してクラッシュすることがある。ローカル
    # の pose 変数だけ更新し、表示用ワイヤーフレームはそれを直接使う。
    elif not tracking_active and key == ord("a"):
        pose[0, 3] -= step_t
    elif not tracking_active and key == ord("d"):
        pose[0, 3] += step_t
    elif not tracking_active and key == ord("q"):
        pose[1, 3] -= step_t
    elif not tracking_active and key == ord("e"):
        pose[1, 3] += step_t
    elif not tracking_active and key == ord("w"):
        pose[2, 3] -= step_t
    elif not tracking_active and key == ord("s"):
        pose[2, 3] += step_t
    elif not tracking_active and key == ord("i"):
        pose[:3, :3] = rot_x(step_r) @ pose[:3, :3]
    elif not tracking_active and key == ord("k"):
        pose[:3, :3] = rot_x(-step_r) @ pose[:3, :3]
    elif not tracking_active and key == ord("j"):
        pose[:3, :3] = rot_y(step_r) @ pose[:3, :3]
    elif not tracking_active and key == ord("l"):
        pose[:3, :3] = rot_y(-step_r) @ pose[:3, :3]
    elif not tracking_active and key == ord("u"):
        pose[:3, :3] = rot_z(step_r) @ pose[:3, :3]
    elif not tracking_active and key == ord("o"):
        pose[:3, :3] = rot_z(-step_r) @ pose[:3, :3]
    elif key == ord("m"):
        # 追跡が合っていると判断した瞬間に押す。今の pose を正としてマーカー→モデルの変換を登録
        register_marker_sample()
    elif key == 8:  # Backspace: 直近の登録サンプルを 1 件取り消す
        undo_marker_sample()
    elif key in (ord("x"), ord("M")):
        # 'x' = 登録を全消去（Shift 不要。'M' は環境によって 'm' として届くことがあった）
        if marker_ref is not None:
            n_before = sum(len(v) for v in marker_ref.samples.values())
            bak = marker_ref.clear()
            print(f"マーカー基準の登録を消去（{n_before} 件）" + (f"。バックアップ: {bak}" if bak else ""))
    elif key == ord("r"):
        if tracking_active:
            relock_from_frame(frame_img)
        else:
            loaded_pose, loaded_path = load_pose_slot(selected_pose_slot)
            if loaded_pose is not None:
                pose = loaded_pose
                reset_all_anchors(pose)
                print(f"姿勢スロット {selected_pose_slot} を読み込み: {loaded_path}")
            else:
                print(f"姿勢スロット {selected_pose_slot} は未保存")
    elif key in POSE_SLOT_KEYS:
        selected_pose_slot = POSE_SLOT_KEYS[key]
        loaded_pose, loaded_path = load_pose_slot(selected_pose_slot)
        if loaded_pose is not None:
            pose = loaded_pose
            reset_all_anchors(pose)
            if tracking_active:
                # スロットの姿勢を基準に近傍だけ探索（GLOBAL 全探索だと無関係な候補へ飛ぶ）
                relock_from_frame(frame_img, reference_pose=loaded_pose,
                                  use_rotation_guard=True, reason="slot")
            print(f"姿勢スロット {selected_pose_slot} を読み込み: {loaded_path}")
        else:
            print(f"姿勢スロット {selected_pose_slot} は未保存")

    if tracking_active:
        frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

        # ① 分割実行中の復帰時回転再探索: ライブの step の前に、控えた画像で候補を数個だけ評価する
        return_search_result = None
        return_search_result_text = ""
        if return_search_job is not None:
            if snap_frames > 0:
                return_search_job = None   # 再ロック中は意味がない
            else:
                return_search_result = return_search_advance(return_search_job)
                if return_search_result is not None:
                    return_search_result_text = return_search_job["occl_text"]
                    return_search_job = None
        _pf_mark("return_search")

        if snap_frames > 0:
            for _ in range(SNAP_FRAME_STEPS):
                tracker.step(frame_rgb)
            snap_frames -= 1
            # 何が SNAP を起こしたか（auto=品質低下の自動再ロック / manual=キー操作 / start=追跡開始）
            # を残しておく。ログ解析で「誤リカバリ」と「手動操作」を区別するため。
            mode_text = f"SNAP {snap_frames} {relock_reason}".rstrip()
        else:
            for _ in range(TRACKING_STEPS_PER_FRAME):
                tracker.step(frame_rgb)
            mode_text = "TRACKING"

        # tracker.step() は登録済みの全アンカーを毎フレーム独立に更新する。
        # quality 評価のために diverge したポーズを読み取るが、最終的には
        # active_anchor の pose に全アンカーを同期する（後段で reset_all_anchors）。
        _pf_mark("m3t_step")
        poses_by_key = {a["key"]: a["model"].pose_cv.copy().astype(np.float32) for a in ANCHORS}
        if return_search_result is not None:
            # ① 再探索の全候補が終わった: 採用候補の相対回転（モデル座標）を今の F 姿勢に掛けて適用する
            _rel, _rdeg, _q0, _q1, _rel_sec, _busy_sec = return_search_result
            if _rel is not None:
                _rp = (poses_by_key["F"].astype(np.float64) @ _rel.astype(np.float64)).astype(np.float32)
                reset_all_anchors(_rp)
                poses_by_key = {k: _rp.copy() for k in poses_by_key}
                prev_output_pose = None   # 再探索先を跳び量ゲートが「跳び」として戻さないように
                if _rdeg >= 3.0:
                    rotation_uncertain_until = time.time() + ROT_UNCERTAIN_AFTER_RETURN_SEC
                    rotation_uncertain_after_return = f"rotation corrected by {_rdeg:.0f} deg on return - verify alignment"
            frame_event = (f"return_search:{_rdeg:.1f}deg q{_q0:.2f}->{_q1:.2f} "
                           f"{_rel_sec*1000:.0f}ms(busy {_busy_sec*1000:.0f}ms) {return_search_result_text}")
        # ⑤ 各歯群ボディ自身の推定（回転込み）。この後の統合／同期で poses_by_key は出力姿勢に上書きされるので、
        # ログ用にここで控えておく（14:34 run で A/B/D の回転誤差が全部 F と同じ値になっていた原因）。
        group_poses_raw = {k: p.copy() for k, p in poses_by_key.items()}
        pose = poses_by_key[active_anchor]
        quality_by_key = {}  # SNAPウォームアップ中は未評価（HUD表示は0.0のまま）
        if snap_frames > 0:
            if relock_reason != "auto":
                session_depth_ref = None   # 手動の開始／スロット再ロックは配置が変わった可能性があるので奥行き基準を取り直す
            fusion_active = False
            fusion_keys = []
            fusion_off_f = False
            fusion_members = {}
            fusion_enter_streak = {}
            fusion_hold_pose = None
            fusion_f_ok_since = None
            fusion_no_member_since = None

        if snap_frames == 0:
            # 距離マップ（方向付き 9 枚）は 1 フレーム 25 ms かかるので QUALITY_EVAL_EVERY フレームに
            # 1 回だけ再計算し、間のフレームは前回のマップを使う。q は EMA で平滑化しているので影響は小さい。
            if (frame_idx % QUALITY_EVAL_EVERY == 0) or dist_map is None:
                dist_map = compute_edge_distance_map(frame_img)
            quality_by_key = {a["key"]: estimate_anchor_quality(dist_map, poses_by_key[a["key"]], a)
                               for a in ANCHORS}
            _pf_mark("quality")
            # 回転源 PnP: F で出力している良好フレームを控えておき、遮蔽が始まったらそれをキーフレームにする
            # （F に戻ってから FUSION_PNP_KEYFRAME_MIN_F_SEC 経つまでは控えない）
            if fusion_off_f:
                pnp_f_since = None
            elif pnp_f_since is None:
                pnp_f_since = time.time()

            # ⑥ F 追跡中の見張り: 特徴点 PnP と F が大きく食い違い続けたら F を PnP の姿勢へ戻す
            f_guard_diff = None      # このフレームの見張り PnP と F の差 (mm, deg)。⑪ キーフレームの検算に使う
            f_guard_pose_now = None  # このフレームで見張り PnP が解けたときの姿勢（⑰ 統合に入るときの姿勢）
            f_guard_reset_now = False
            if F_GUARD_ENABLED and not fusion_off_f:
                _now_g = time.time()
                _gray_g = _pnp_gray(frame_img)
                _fpose = poses_by_key["F"]
                _qf_g = quality_by_key.get("F", 0.0)
                _f_long = pnp_f_since is not None and _now_g - pnp_f_since >= (
                    F_GUARD_AFTER_RETURN_SEC if f_guard_after_return else F_GUARD_MIN_F_SEC)
                _rekey = f_guard is None
                if f_guard is not None:
                    _gi = pnp_track(f_guard, _gray_g, None, bgr=frame_img)
                    if _gi["ok"] and _gi["n"] >= F_GUARD_MIN_POINTS and _gi["res_px"] <= _pnp_px(F_GUARD_MAX_RES_PX):
                        _P = pnp_output_pose(f_guard, f_guard["R"])
                        _d_mm = float(np.linalg.norm(_group_center_cam("F", _fpose) - _group_center_cam("F", _P))) * 1000.0
                        _d_deg = rotation_angle_deg(_fpose, _P)
                        f_guard_diff = (_d_mm, _d_deg)
                        f_guard_pose_now = _P
                        if _d_mm > F_GUARD_DIVERGE_MM or _d_deg > F_GUARD_DIVERGE_DEG:
                            f_guard_streak += 1
                        else:
                            f_guard_streak = 0
                        if f_guard_streak >= F_GUARD_DIVERGE_FRAMES:
                            print(f"F の見張り: F が特徴点 PnP から {_d_mm:.0f} mm / {_d_deg:.0f}° 離れた"
                                  f"（{f_guard_streak} フレーム、qF={_qf_g:.2f}）→ PnP の姿勢へ戻す")
                            reset_all_anchors(_P)
                            poses_by_key = {k: _P.copy() for k in poses_by_key}
                            pose = poses_by_key[active_anchor]
                            quality_by_key = {a["key"]: estimate_anchor_quality(dist_map, poses_by_key[a["key"]], a)
                                              for a in ANCHORS}
                            prev_output_pose = None          # 戻した姿勢を跳び量ゲートが「跳び」として戻さないように
                            pose_vel_t = np.zeros(3)
                            pose_vel_rv = np.zeros(3)
                            frame_event = f"f_guard_reset {_d_mm:.0f}mm/{_d_deg:.0f}deg"
                            f_guard_streak = 0
                            f_guard_reset_now = True
                            # ⑪(b) 戻す前の F（ずれていた）を遮蔽中の PnP のキーフレームに残さない
                            if FUSION_ROT_FROM_GROUP and FUSION_ROT_SOURCE == "pnp":
                                pnp_last_good = (_gray_g, _P.copy(), _now_g)
                        elif (_d_mm <= F_GUARD_AGREE_MM and _d_deg <= F_GUARD_AGREE_DEG and _qf_g >= F_GUARD_MIN_QF
                              and _now_g - f_guard_time >= F_GUARD_REKEY_SEC):
                            _rekey = True
                    else:
                        f_guard_streak = 0
                        # 見張りが使えない（点不足・残差大）まま F が良好なら取り直す
                        _rekey = _qf_g >= F_GUARD_MIN_QF and _now_g - f_guard_time >= F_GUARD_REKEY_SEC
                if _rekey and _f_long and _qf_g >= F_GUARD_MIN_QF:
                    f_guard = pnp_keyframe_begin(_gray_g, poses_by_key["F"], with_anchors=False)   # 見張りは歯群の振り分け不要
                    f_guard_time = _now_g
                    f_guard_streak = 0
                    f_guard_after_return = False
            elif fusion_off_f:
                f_guard = None
                f_guard_streak = 0
            _pf_mark("f_guard")

            # 回転源 PnP: F で出力している良好フレームを控えておき、遮蔽が始まったらそれをキーフレームにする
            # （F に戻ってから FUSION_PNP_KEYFRAME_MIN_F_SEC 経つまでは控えない。⑪(a) 見張りと食い違う F は控えない）
            _key_guard_ok = (FUSION_PNP_KEYFRAME_GUARD_DEG <= 0 or f_guard_diff is None
                             or (f_guard_diff[0] <= FUSION_PNP_KEYFRAME_GUARD_MM
                                 and f_guard_diff[1] <= FUSION_PNP_KEYFRAME_GUARD_DEG))
            if (FUSION_ROT_FROM_GROUP and FUSION_ROT_SOURCE == "pnp" and not fusion_off_f and not f_guard_reset_now
                    and time.time() - pnp_f_since >= FUSION_PNP_KEYFRAME_MIN_F_SEC
                    and quality_by_key.get("F", 0.0) >= FUSION_PNP_KEYFRAME_MIN_QF and _key_guard_ok):
                pnp_last_good = (_pnp_gray(frame_img), poses_by_key["F"].copy(), time.time())

            last_anchor_quality = dict(quality_by_key)

            if HIERARCHICAL_ANCHORS:
                # ── Anchor Switching（階層版）────────────────────────────────────
                # 主アンカー F（歯列全体）が十分合っていれば F。F が遮蔽などで落ちたときだけ、
                # 適合度が高く、かつ直前の良好姿勢から大きく離れていない歯群へ切り替える。
                # 比較用ベースライン（anchor_switching_enabled=False）では歯群を候補から外し、常に F を出す
                _single_select = anchor_switching_enabled and not anchor_fusion_enabled
                group_keys = [k for k in ANCHOR_POSE_SOURCE_KEYS if k in quality_by_key] if _single_select else []
                display_keys = [k for k in ("A", "B", "C", "D") if k in quality_by_key] if _single_select else []
                best_group = max(group_keys, key=lambda k: quality_by_key[k]) if group_keys else None
                best_display = max(display_keys, key=lambda k: quality_by_key[k]) if display_keys else None
                qF = quality_by_key.get("F", 0.0)

                def _switch_sane(p):
                    if last_good_pose is None:
                        return True
                    return (rotation_angle_deg(p, last_good_pose) <= ANCHOR_SWITCH_MAX_ROT_DEG
                            and float(np.linalg.norm(p[:3, 3] - last_good_pose[:3, 3])) <= ANCHOR_SWITCH_MAX_TRANS_M)

                now_t = time.time()
                on_group = active_anchor != "F"
                q_cur = quality_by_key.get(active_anchor, 0.0)
                if manual_anchor_override in ANCHOR_POSE_SOURCE_KEYS:
                    # 手動指定: その歯群が見えている（q ≥ 0.30）ときだけ従う。見えていなければ F に退避
                    # （見えない歯群の姿勢を出し続けると自動再ロックが連発してデモにならない）。
                    if quality_by_key.get(manual_anchor_override, 0.0) >= 0.30:
                        chosen = manual_anchor_override
                    else:
                        chosen = "F"
                elif best_group is None:
                    chosen = "F"
                elif on_group:
                    # 切替済み: ヒステリシス。F が十分回復するか、今の歯群が MIN−0.10 を割るまで維持
                    # （0.50 で入って 0.47 で戻る…の往復を防ぐ）。最低 ANCHOR_MIN_DWELL_SEC は留まる。
                    stay = (q_cur >= ANCHOR_GROUP_MIN_QUALITY - 0.15) and (qF < ANCHOR_F_RECOVER_QUALITY)
                    if now_t - anchor_switch_time < ANCHOR_MIN_DWELL_SEC:
                        stay = True
                    # 逸脱監視: 切替時点から回り過ぎ/動き過ぎた歯群は捨てて F（保持姿勢）へ
                    if anchor_switch_pose is not None:
                        p_cur = poses_by_key[active_anchor]
                        drift_deg = rotation_angle_deg(p_cur, anchor_switch_pose)
                        drift_m = float(np.linalg.norm(_group_center_cam(active_anchor, p_cur)
                                                       - _group_center_cam(active_anchor, anchor_switch_pose)))
                        if (not GROUP_ROTATION_FROM_F and drift_deg > ANCHOR_GROUP_MAX_DRIFT_DEG) \
                                or drift_m > ANCHOR_GROUP_MAX_DRIFT_M:
                            print(f"歯群 {active_anchor} の逸脱 ({drift_deg:.0f}°, {drift_m*1000:.0f} mm) → F へ戻す")
                            stay = False
                            group_block_until = now_t + ANCHOR_DRIFT_COOLDOWN_SEC
                            quality_by_key[active_anchor] = 0.0   # このフレームでは候補にしない
                    chosen = active_anchor if stay else "F"
                    # より良い歯群があれば乗り換え（マージン付き）
                    if stay and best_group != active_anchor and quality_by_key[best_group] > q_cur + ANCHOR_SWITCH_MARGIN \
                            and _switch_sane(poses_by_key[best_group]) and now_t - anchor_switch_time >= ANCHOR_MIN_DWELL_SEC:
                        chosen = best_group
                elif qF >= ANCHOR_F_OK_QUALITY or now_t < group_block_until:
                    chosen = "F"
                elif (quality_by_key[best_group] >= ANCHOR_GROUP_MIN_QUALITY
                      and quality_by_key[best_group] > qF + ANCHOR_SWITCH_MARGIN
                      and _switch_sane(poses_by_key[best_group])):
                    chosen = best_group
                else:
                    chosen = "F"
                if chosen != active_anchor:
                    print(f"アンカー切替: {active_anchor}({quality_by_key.get(active_anchor, 0):.2f}) → "
                          f"{chosen}({quality_by_key.get(chosen, 0):.2f})  [F={qF:.2f}]")
                    if chosen == "F":
                        anchor_switch_pose = None
                    elif active_anchor == "F" or anchor_switch_pose is None:
                        # 逸脱監視の基準は「F を離れた時点」。歯群間の乗り換え（B→A など）で基準を更新すると
                        # 20 mm 流れては乗り換え…を繰り返して 47 mm まで流れた（2026-10-01 12:35 B）。
                        anchor_switch_pose = poses_by_key[chosen].copy()
                    active_anchor = chosen
                    anchor_switch_time = now_t
                # Unity で強調する歯群: 切替中はその歯群、通常時は適合度が最も高い歯群。
                # 通常時はヒステリシス付き（今の強調歯群より 0.10 以上良い歯群が出たときだけ乗り換える）で、
                # 適合度が拮抗しているときに色がチラつかないようにする。
                if chosen != "F":
                    display_anchor = chosen
                elif not DISPLAY_BEST_GROUP_WHEN_F:
                    display_anchor = ""
                elif best_display is None or quality_by_key[best_display] < 0.30:
                    display_anchor = ""
                else:
                    q_disp = quality_by_key.get(display_anchor, -1.0) if display_anchor in display_keys else -1.0
                    if (display_anchor not in display_keys or q_disp < 0.30
                            or quality_by_key[best_display] > q_disp + 2 * ANCHOR_SWITCH_MARGIN):
                        display_anchor = best_display
            elif manual_anchor_override:
                if active_anchor != manual_anchor_override:
                    print(f"アンカーオーバーライド適用: {active_anchor} → {manual_anchor_override}")
                    active_anchor = manual_anchor_override
            else:
                # 自動切替の候補は AUTO_ANCHOR_KEYS のみ（D は手動指定時だけ）
                best_key = max(AUTO_ANCHOR_KEYS, key=lambda k: quality_by_key[k])
                if best_key != active_anchor and quality_by_key[best_key] > quality_by_key[active_anchor] + ANCHOR_SWITCH_MARGIN:
                    print(f"アンカー切替: {active_anchor}({quality_by_key[active_anchor]:.2f}) → "
                          f"{best_key}({quality_by_key[best_key]:.2f})")
                    active_anchor = best_key

            # ── 統合モード: 遮蔽でスコアが落ちた歯群を外し、見えている歯群すべてで歯列全体の並進を推定 ──────
            if (HIERARCHICAL_ANCHORS and anchor_fusion_enabled and anchor_switching_enabled
                    and manual_anchor_override not in ANCHOR_POSE_SOURCE_KEYS):
                qF_ = quality_by_key.get("F", 0.0)
                now_t = time.time()
                if not fusion_off_f:
                    # F が良好に出力している間に、歯群ボディの回転が出力回転からどちらへ偏るかを覚える（統合中の追従先の補正用）
                    if FUSION_ROT_FROM_GROUP and FUSION_ROT_BIAS_COMP and qF_ >= FUSION_ROT_BIAS_MIN_QF and group_poses_raw:
                        R_out = poses_by_key["F"][:3, :3].astype(np.float64)
                        for k in FUSION_ROT_SOURCE_KEYS:
                            if quality_by_key.get(k, 0.0) < FUSION_ROT_MIN_QUALITY.get(k, 0.6):
                                continue
                            rel = group_poses_raw[k][:3, :3].astype(np.float64).T @ R_out   # モデル座標: 歯群回転 → 出力回転
                            prev = fusion_rot_bias.get(k)
                            fusion_rot_bias[k] = rel if prev is None else _slerp_R(prev, rel, FUSION_ROT_BIAS_ALPHA)
                    # F で追跡中。F が落ちたら F を離れ、出力を最後の良好姿勢へ戻す（HOLD と同じ扱い。
                    # 遮蔽で流れた F 姿勢と last_good の差を跳び量ゲートに拒否させないため prev_output_pose を外す）。
                    if qF_ < ANCHOR_F_OK_QUALITY and last_good_pose is not None and now_t >= group_block_until:
                        fusion_off_f = True
                        _depart = last_good_pose
                        _from_guard = False
                        if (FUSION_DEPART_FROM_GUARD and f_guard_pose_now is not None
                                and float(np.linalg.norm(_group_center_cam("F", f_guard_pose_now)
                                                         - _group_center_cam("F", last_good_pose))) <= 0.010
                                and rotation_angle_deg(f_guard_pose_now, last_good_pose) <= 10.0):
                            # ⑰ 手が入りかけて F がずれ始めた姿勢（last_good）ではなく、見張りの PnP がこのフレームで解いた姿勢から始める
                            _depart = f_guard_pose_now.astype(np.float32).copy()
                            if FUSION_DEPART_FROM_GUARD == 2:
                                # 回転だけ見張りから。F 重心の位置は last_good のまま（見張りの並進は 2〜3 mm 偏ることがある）
                                _cF = ANCHOR_BY_KEY["F"]["centroid"]
                                _depart[:3, 3] = (last_good_pose[:3, :3] @ _cF + last_good_pose[:3, 3]) - _depart[:3, :3] @ _cF
                            _from_guard = True
                        anchor_switch_pose = _depart.copy()
                        anchor_switch_time = now_t
                        fusion_hold_pose = _depart.copy()
                        fusion_members = {}
                        fusion_enter_streak = {}
                        fusion_f_ok_since = None
                        fusion_no_member_since = None
                        prev_output_pose = None
                        # 回転追従の基準は離脱時の姿勢から（回転は固定スタート、F 重心の位置もここが基準）
                        fusion_rot_ref = _depart.copy()
                        fusion_rot_src = ""
                        fusion_rot_total_deg = 0.0
                        fusion_rot_streak = {}
                        fusion_rot_prev_R = {}
                        fusion_pnp = None
                        fusion_pnp_bad_since = None
                        if _from_guard and FUSION_DEPART_FROM_GUARD == 1 and FUSION_ROT_FROM_GROUP and FUSION_ROT_SOURCE == "pnp":
                            fusion_pnp = f_guard          # ⑰ 離脱時の姿勢を出した見張りの点をそのまま使う
                            print("統合: 離脱時の姿勢・回転源 PnP とも F の見張りから")
                        if (fusion_pnp is None and FUSION_ROT_FROM_GROUP and FUSION_ROT_SOURCE == "pnp" and pnp_last_good is not None
                                and now_t - pnp_last_good[2] <= FUSION_PNP_KEYFRAME_MAX_AGE_SEC):
                            fusion_pnp = pnp_keyframe_begin(pnp_last_good[0], pnp_last_good[1], pnp_last_good[2])
                        if (fusion_pnp is None and FUSION_PNP_FROM_GUARD and FUSION_ROT_FROM_GROUP
                                and FUSION_ROT_SOURCE == "pnp" and f_guard is not None and len(f_guard["X"]) >= FUSION_PNP_MIN_POINTS):
                            # ⑮ 直近の良好フレームが無いときは、F の見張りが追い続けている点の集合を引き継ぐ（キーフレームは
                            #   見張りが F と一致を確かめて作ったもの。点は今のフレームまで追えている）
                            fusion_pnp = f_guard
                            print("統合: 回転源 PnP は F の見張りの点を引き継ぐ")
                        if fusion_pnp is not None and fusion_pnp is f_guard and FUSION_PNP_ANCHORS:
                            fusion_pnp["anchor"] = _anchor_labels(fusion_pnp["X"])   # ⑲ 見張りの点は引き継いだときに歯群へ振り分ける
                        if PNP_DUMP_PATH and fusion_pnp is not None:
                            _pnp_dump_segments.append(fusion_pnp)
                        if FUSION_ROT_FROM_GROUP and FUSION_ROT_SOURCE == "pnp":
                            print("統合: 回転源 PnP のキーフレーム "
                                  + (f"{fusion_pnp['n0']} 点" if fusion_pnp else "作れず（直近の F 良好フレームなし／特徴点不足）→ 回転は保持"))
                        return_search_job = None   # 評価途中の再探索があれば捨てる（復帰前の画像なので意味がない）
                        _bias_text = ", ".join(f"{k}:{np.rad2deg(np.linalg.norm(cv2.Rodrigues(fusion_rot_bias[k])[0])):.1f}°"
                                               for k in FUSION_ROT_SOURCE_KEYS if k in fusion_rot_bias)
                        print(f"統合: F を離脱 (F={qF_:.2f})" + (f"  回転偏り補正: {_bias_text}" if _bias_text else ""))
                else:
                    fusion_f_ok_since = (fusion_f_ok_since or now_t) if qF_ >= ANCHOR_F_RECOVER_QUALITY else None
                    all_visible = all(quality_by_key.get(k, 0.0) >= FUSION_GROUP_EXIT_QUALITY for k in FUSION_KEYS)
                    since_leave = now_t - anchor_switch_time
                    reason = None
                    if fusion_f_ok_since is not None and all_visible and since_leave >= 1.0:
                        reason = "全歯群が見えている"
                    elif fusion_f_ok_since is not None and now_t - fusion_f_ok_since >= FUSION_F_RETURN_SEC:
                        reason = f"F が {FUSION_F_RETURN_SEC:.0f} s 以上良好"
                    elif (fusion_no_member_since is not None and now_t - fusion_no_member_since >= FUSION_NO_MEMBER_SEC
                          and qF_ >= ANCHOR_F_OK_QUALITY):
                        reason = "使える歯群なし"
                    if reason is not None:
                        print(f"統合解除 → F (F={qF_:.2f}, {reason})")
                        # 離脱中にどれだけ経ち・動いたか（① 再探索と ② 警告の要否判定）。状態をクリアする前に計算する
                        _occl_sec = now_t - anchor_switch_time
                        _occl_moved = (float(np.linalg.norm(_group_center_cam("F", fusion_hold_pose)
                                                            - _group_center_cam("F", anchor_switch_pose)))
                                       if fusion_hold_pose is not None and anchor_switch_pose is not None else 0.0)
                        _warn_rot = _occl_sec >= ROT_UNCERTAIN_AFTER_SEC or _occl_moved >= ROT_UNCERTAIN_MOVE_M
                        _do_search = (RETURN_ROT_SEARCH_ENABLED
                                      and (_occl_sec >= RETURN_ROT_SEARCH_MIN_OCCL_SEC or _occl_moved >= RETURN_ROT_SEARCH_MIN_MOVE_M))
                        fusion_off_f = False
                        fusion_active = False
                        fusion_keys = []
                        fusion_members = {}
                        fusion_enter_streak = {}
                        fusion_hold_pose = None
                        anchor_switch_pose = None
                        # ② 長い遮蔽／大きな移動の後は、復帰後しばらく「回転不確か」を出す（人が重畳を確認できるように）
                        if _warn_rot:
                            rotation_uncertain_until = now_t + ROT_UNCERTAIN_AFTER_RETURN_SEC
                            rotation_uncertain_after_return = (f"F resumed after {_occl_sec:.0f}s occlusion "
                                                               f"({_occl_moved*1000:.0f}mm moved) - verify alignment")
                        # ① 回転が古い疑いがあるときだけ、固定していた回転の周りを再探索して古い局所解から抜ける
                        _occl_text = (f"occl{_occl_sec:.1f}s/{_occl_moved*1000:.0f}mm"
                                      + (f"/rot{fusion_rot_total_deg:.0f}deg({fusion_rot_src or '-'})" if FUSION_ROT_FROM_GROUP else ""))
                        fusion_rot_ref = None
                        fusion_rot_src = ""
                        fusion_rot_motion_ref = None
                        fusion_rot_motion_until = 0.0
                        fusion_pnp = None
                        fusion_pnp_bad_since = None
                        f_guard_after_return = True   # ⑯ 戻った直後の F（統合のずれを引き継ぐ）で見張りのキーフレームを作らない
                        if _do_search:
                            # ① 候補はこの場では評価しない（1 フレーム 450 ms 止まる）。画像を控えて次フレームから分割評価する
                            return_search_job = return_search_begin(frame_img, dist_map, poses_by_key["F"], _occl_text)
                            frame_event = f"return_search_begin {len(return_search_job['cands'])}cands {_occl_text}"
                        else:
                            frame_event = f"return_no_search {_occl_text}"
                if fusion_off_f:
                    # 歯群の採用判定（スコアと逸脱の両方にヒステリシス）。逸脱は直前の出力姿勢から予測した重心との差。
                    new_members, dropped = {}, []
                    for k in FUSION_KEYS:
                        q_k = quality_by_key.get(k, 0.0)
                        dev_k = float(np.linalg.norm(_group_center_cam(k, poses_by_key[k])
                                                     - _group_center_cam(k, fusion_hold_pose)))
                        if k in fusion_members:
                            if q_k < FUSION_GROUP_EXIT_QUALITY:
                                dropped.append(f"{k}:q={q_k:.2f}")
                            elif dev_k > FUSION_GROUP_DEV_EXIT_M:
                                dropped.append(f"{k}:dev={dev_k*1000:.0f}mm")
                            else:
                                new_members[k] = fusion_members[k]
                        elif q_k >= FUSION_GROUP_ENTER_QUALITY and dev_k <= FUSION_GROUP_DEV_ENTER_M:
                            fusion_enter_streak[k] = fusion_enter_streak.get(k, 0) + 1
                            if fusion_enter_streak[k] >= FUSION_ENTER_FRAMES:
                                new_members[k] = now_t
                        else:
                            fusion_enter_streak[k] = 0
                    fusion_members = new_members
                    sane = sorted(fusion_members)
                    # 案 A: B/D 単独では統合しない（A を含むか 2 歯群以上が条件）
                    if FUSION_REQUIRE_A_OR_PAIR and sane and "A" not in sane and len(sane) < 2:
                        if fusion_solo_blocked != sane[0]:
                            print(f"統合: {sane[0]} 単独は使わない (q={quality_by_key.get(sane[0], 0):.2f}) → 保持")
                        fusion_solo_blocked = sane[0]
                        dropped.append(f"{sane[0]}:solo")
                        sane = []
                    elif sane:
                        fusion_solo_blocked = ""   # 使える構成になったらリセット（メンバーなしの間は保持して連続表示を防ぐ）
                    # ── 統合中の回転追従: 条件を満たす歯群（A 優先、次に B）の回転へ固定回転を少しずつ寄せる ──
                    if fusion_rot_ref is None:
                        fusion_rot_ref = anchor_switch_pose.copy()
                    rot_src_now = ""
                    fusion_pnp_info = None
                    # 静止時の積み上がり対策: 回転源（A）の投影範囲が基準フレームから動いたときだけ追従を許す
                    if FUSION_ROT_FROM_GROUP and FUSION_ROT_MOTION_GATE and FUSION_ROT_SOURCE != "pnp":
                        _mgray = _motion_gray(frame_img)
                        _mkey = FUSION_ROT_SOURCE_KEYS[0]
                        if fusion_rot_motion_ref is None:
                            fusion_rot_motion_ref = motion_ref_begin(_mgray, poses_by_key[_mkey], _mkey)
                            fusion_rot_motion_px = None
                        else:
                            fusion_rot_motion_px = motion_from_ref(fusion_rot_motion_ref, _mgray)
                            if fusion_rot_motion_px is None or fusion_rot_motion_px >= FUSION_ROT_MOTION_MIN_PX:
                                if fusion_rot_motion_px is not None:
                                    fusion_rot_motion_until = now_t + FUSION_ROT_MOTION_HOLD_SEC
                                # 動いた（または追えなくなった）ので、ここから先の動きを測り直す
                                fusion_rot_motion_ref = motion_ref_begin(_mgray, poses_by_key[_mkey], _mkey)
                    rot_moving = (not FUSION_ROT_MOTION_GATE) or now_t < fusion_rot_motion_until
                    # 2 歯群以上が見えているときだけ回転を追従し、回転源の回転が「2 歯群の重心を結ぶ向き（画像上）」と
                    # 矛盾したら追従を止めて離脱時の回転へ戻す（13:12 run: A 単独の回転が間違ったまま毎フレーム積み上がり、
                    # 35° 近くまで走って統合出力が 45 mm ずれた。重心位置は信頼できるので、その並びの向きで回転を検算する）
                    rot_pair_ok = len(sane) >= 2
                    _pair_conflict = False
                    if rot_pair_ok and FUSION_ROT_PAIR_CHECK:
                        _k0, _k1 = sane[0], sane[-1]
                        _obs = [_group_center_cam(k, poses_by_key[k]) for k in (_k0, _k1)]
                        _prd = [_group_center_cam(k, fusion_rot_ref) for k in (_k0, _k1)]
                        def _dir2d(p0, p1):
                            v = np.array([p1[0] / p1[2] - p0[0] / p0[2], p1[1] / p1[2] - p0[1] / p0[2]])
                            n = float(np.linalg.norm(v))
                            return v / n if n > 1e-9 else None
                        _do, _dp = _dir2d(*_obs), _dir2d(*_prd)
                        if _do is not None and _dp is not None:
                            _pair_deg = float(np.degrees(np.arccos(np.clip(np.dot(_do, _dp), -1.0, 1.0))))
                            if _pair_deg > FUSION_ROT_PAIR_MAX_DEG:
                                rot_pair_ok = False
                                _pair_conflict = True
                                if fusion_rot_total_deg > 0.5:
                                    print(f"統合: 回転追従を取り消し（{_k0}-{_k1} 重心の向きが現在の回転と {_pair_deg:.0f}° 矛盾、"
                                          f"累計 {fusion_rot_total_deg:.1f}° → 離脱時の回転へ戻す）")
                                    fusion_rot_ref = anchor_switch_pose.copy()
                                    fusion_rot_total_deg = 0.0
                                    fusion_rot_streak = {}
                    if FUSION_ROT_FROM_GROUP and FUSION_ROT_SOURCE == "pnp":
                        # キーフレーム特徴点 PnP の回転へ寄せる。キーフレームからの回転が ZUPT しきい値未満なら離脱時の回転へ
                        if fusion_pnp is None:
                            rot_src_now = "PNP(none)"
                        else:
                            if sane and not FUSION_PNP_ONLY:
                                _c_prior = _group_center_cam("F", _fuse_groups(sane, poses_by_key, fusion_rot_ref))
                            elif prev_output_pose is not None:
                                _c_prior = _group_center_cam("F", prev_output_pose)
                            else:
                                _c_prior = None
                            _anchor_ok = ({k for k in ("A", "B", "C", "D")
                                           if quality_by_key.get(k, 0.0) >= FUSION_PNP_ANCHOR_Q} if FUSION_PNP_ANCHORS else None)
                            fusion_pnp_info = pnp_track(fusion_pnp, _pnp_gray(frame_img), _c_prior, bgr=frame_img,
                                                        edge_ctx=(dist_map, list(sane)) if (sane and FUSION_PNP_USE_EDGES) else None,
                                                        anchor_ok=_anchor_ok)
                            if PNP_DUMP_PATH:
                                _a = np.where(fusion_pnp["alive"])[0]
                                _pnp_dump_frames.append({
                                    "t": time.time(), "seg": len(_pnp_dump_segments) - 1, "idx": _a,
                                    "cur": fusion_pnp["cur"][_a].copy(), "R": fusion_pnp["R"].copy(), "t_vec": fusion_pnp["t"].copy(),
                                    "ok": bool(fusion_pnp_info["ok"]), "still": bool(fusion_pnp_info.get("still", False)),
                                    "c_prior": None if _c_prior is None else np.asarray(_c_prior, np.float64).copy()})
                            R_target = None
                            if not fusion_pnp_info["ok"]:
                                rot_src_now = "PNP(bad)"          # 解を信用できない → 回転は今のまま
                            elif fusion_pnp_info["still"]:
                                rot_src_now = "PNP(still)"        # 特徴点の大多数が動いていない → 離脱時の回転
                                R_target = anchor_switch_pose[:3, :3].astype(np.float64)
                            elif _pair_conflict:
                                rot_src_now = "PNP(pair)"         # 歯群重心の並びと矛盾 → 上で離脱時の回転へ戻している
                            elif fusion_pnp_info["deg"] < max(FUSION_PNP_MIN_DEG, FUSION_PNP_ZUPT_SIGMA * fusion_pnp_info["std_deg"]):
                                rot_src_now = "PNP(hold)"         # ZUPT: 回っていない → 離脱時の回転
                                R_target = anchor_switch_pose[:3, :3].astype(np.float64)
                            else:
                                rot_src_now = "PNP"
                                R_target = fusion_pnp_info["R"]
                            if R_target is not None:
                                R_cur = fusion_rot_ref[:3, :3].astype(np.float64)
                                rv, _ = cv2.Rodrigues(R_target @ R_cur.T)
                                ang = float(np.rad2deg(np.linalg.norm(rv)))
                                if ang > 1e-3:
                                    step_deg = min(ang * FUSION_PNP_BLEND, FUSION_ROT_MAX_STEP_DEG)
                                    Rd, _ = cv2.Rodrigues(rv / np.linalg.norm(rv) * np.deg2rad(step_deg))
                                    cand_ref = _rotate_pose_about_point(fusion_rot_ref, Rd, _group_center_cam("F", fusion_rot_ref))
                                    total = rotation_angle_deg(cand_ref, anchor_switch_pose)
                                    if total <= FUSION_ROT_MAX_TOTAL_DEG:
                                        fusion_rot_ref = cand_ref
                                        fusion_rot_total_deg = total
                                    else:
                                        rot_src_now = "PNP(cap)"
                        if rot_src_now.split("(")[0] != fusion_rot_src.split("(")[0]:
                            print(f"統合: 回転源 {fusion_rot_src or '固定'} → {rot_src_now}")
                    elif FUSION_ROT_FROM_GROUP and rot_pair_ok:
                        for k in FUSION_ROT_SOURCE_KEYS:
                            prev_pk = fusion_rot_prev_R.get(k)
                            jitter = rotation_angle_deg(poses_by_key[k], prev_pk) if prev_pk is not None else 0.0
                            fusion_rot_prev_R[k] = poses_by_key[k].copy()
                            diff = rotation_angle_deg(poses_by_key[k], fusion_rot_ref)
                            # ヒステリシス: 追従中の歯群は q が EXIT 閾値を割るまで続ける（12:09 run で A の q が 0.60 前後を
                            # 行き来して 固定→A→固定 が数フレームごとに切り替わり、追従が細切れになった）
                            following = fusion_rot_src.split("(")[0] == k
                            q_min = FUSION_ROT_EXIT_QUALITY.get(k, 0.55) if following else FUSION_ROT_MIN_QUALITY.get(k, 0.6)
                            ok = (k in fusion_members
                                  and quality_by_key.get(k, 0.0) >= q_min
                                  and jitter <= FUSION_ROT_MAX_JITTER_DEG
                                  and diff <= FUSION_ROT_MAX_DIFF_DEG)
                            fusion_rot_streak[k] = fusion_rot_streak.get(k, 0) + 1 if ok else 0
                            if not rot_src_now and ok and (following or fusion_rot_streak[k] >= FUSION_ROT_MIN_FRAMES):
                                rot_src_now = k
                        if rot_src_now and not rot_moving:
                            rot_src_now = f"{rot_src_now}(still)"   # 模型が動いていないので寄せない（回転は保持）
                        elif rot_src_now:
                            R_cur = fusion_rot_ref[:3, :3].astype(np.float64)
                            R_src = poses_by_key[rot_src_now][:3, :3].astype(np.float64)
                            if FUSION_ROT_BIAS_COMP and rot_src_now in fusion_rot_bias:
                                R_src = R_src @ fusion_rot_bias[rot_src_now]   # 歯群の系統的な偏りを除いた「F 相当」の回転
                            rv, _ = cv2.Rodrigues(R_src @ R_cur.T)
                            ang = float(np.rad2deg(np.linalg.norm(rv)))
                            if ang > 1e-3:
                                step_deg = min(ang * FUSION_ROT_BLEND, FUSION_ROT_MAX_STEP_DEG)
                                Rd, _ = cv2.Rodrigues(rv / np.linalg.norm(rv) * np.deg2rad(step_deg))
                                cand_ref = _rotate_pose_about_point(fusion_rot_ref, Rd, _group_center_cam("F", fusion_rot_ref))
                                total = rotation_angle_deg(cand_ref, anchor_switch_pose)
                                if total <= FUSION_ROT_MAX_TOTAL_DEG:
                                    fusion_rot_ref = cand_ref
                                    fusion_rot_total_deg = total
                                else:
                                    rot_src_now = f"{rot_src_now}(cap)"   # 累計上限: これ以上は寄せない（回転は保持）
                        if rot_src_now.replace("(still)", "") != fusion_rot_src.replace("(still)", ""):
                            print(f"統合: 回転源 {fusion_rot_src or '固定'} → {rot_src_now or '固定'} "
                                  f"(累計 {fusion_rot_total_deg:.1f}°, " + ", ".join(f"{k}:q={quality_by_key.get(k, 0):.2f}"
                                                                              for k in FUSION_ROT_SOURCE_KEYS) + ")")
                    fusion_rot_src = rot_src_now
                    _pnp_trans_ok = (FUSION_PNP_USE_TRANSLATION and fusion_pnp is not None
                                     and fusion_pnp_info is not None and fusion_pnp_info["ok"])
                    _pnp_alive = fusion_pnp is not None and fusion_pnp_info is not None and fusion_pnp_info["ok"]
                    fusion_pnp_bad_since = None if _pnp_alive else (fusion_pnp_bad_since or now_t)
                    if sane and not FUSION_PNP_ONLY:
                        fusion_no_member_since = None
                        fused = _fuse_groups(sane, poses_by_key, fusion_rot_ref)
                        if _pnp_trans_ok:
                            fused = pnp_output_pose(fusion_pnp, fusion_rot_ref[:3, :3])   # ④ 並進は PnP の F 重心
                        cF = ANCHOR_BY_KEY["F"]["centroid"]
                        drift_m = float(np.linalg.norm((fused[:3, :3] @ cF + fused[:3, 3])
                                                       - (anchor_switch_pose[:3, :3] @ cF + anchor_switch_pose[:3, 3])))
                        if drift_m > (FUSION_PNP_MAX_DRIFT_M if _pnp_trans_ok else ANCHOR_GROUP_MAX_DRIFT_M):
                            # 統合姿勢が基準から流れ過ぎ → 歯群を信用せず保持姿勢のまま F へ（クールダウン付き）
                            print(f"統合姿勢の逸脱 ({drift_m*1000:.0f} mm) → F へ戻す")
                            poses_by_key["F"] = fusion_hold_pose.copy()
                            fusion_off_f = False
                            fusion_active = False
                            fusion_keys = []
                            fusion_members = {}
                            fusion_enter_streak = {}
                            anchor_switch_pose = None
                            group_block_until = now_t + ANCHOR_DRIFT_COOLDOWN_SEC
                        else:
                            if FUSION_TRANS_SMOOTH < 1.0:
                                fused[:3, 3] = (FUSION_TRANS_SMOOTH * fused[:3, 3]
                                                + (1.0 - FUSION_TRANS_SMOOTH) * fusion_hold_pose[:3, 3])
                            if sane != sorted(fusion_keys):
                                print(f"統合: {'+'.join(sane)} で姿勢を推定 (F={qF_:.2f}, "
                                      + ", ".join(f"{k}={quality_by_key.get(k, 0):.2f}" for k in sane)
                                      + (f"; 除外 {' '.join(dropped)}" if dropped else "") + ")")
                            fusion_active = True
                            fusion_keys = list(sane)
                            fusion_hold_pose = fused.copy()
                            poses_by_key["F"] = fused     # 出力は F キーのまま、中身を統合姿勢に差し替える
                            display_anchor = "+".join(sane)
                    else:
                        # 使える歯群がない: F の生の姿勢（遮蔽で流れている）ではなく直前の出力姿勢を保持する。
                        # 適合度が低いままなら従来どおり HOLD → 自動再ロックへ落ちる。
                        if sane:
                            # ⑱ PNP_ONLY: 歯群は見えている（切替・復帰の判断と表示には使う）が、姿勢には使わない
                            fusion_no_member_since = None
                            fusion_active = True
                            fusion_keys = list(sane)
                        else:
                            if fusion_no_member_since is None:
                                fusion_no_member_since = now_t
                            if fusion_active:
                                print("統合: 使える歯群なし → " + ("惰性追跡中なので F の生姿勢を通す" if coasting else "直前姿勢を保持")
                                      + (f" (除外 {' '.join(dropped)})" if dropped else ""))
                            fusion_active = False
                            fusion_keys = []
                        if _pnp_trans_ok:
                            # ④ 歯群が見えなくても特徴点 PnP が生きていれば、その姿勢で追う
                            fusion_hold_pose = pnp_output_pose(fusion_pnp, fusion_rot_ref[:3, :3])
                            poses_by_key["F"] = fusion_hold_pose.copy()
                        elif coasting:
                            # ① 惰性追跡中は凍結しない: 予測姿勢から進めた F の生姿勢をそのまま出力し、保持姿勢も更新する
                            fusion_hold_pose = poses_by_key["F"].copy()
                        elif (PREDICT_ENABLED and fusion_no_member_since is not None
                              and now_t - fusion_no_member_since <= FUSION_HOLD_PREDICT_MAX_SEC
                              and float(np.linalg.norm(pose_vel_t)) >= COAST_MIN_SPEED_M_PER_FRAME):
                            # 案 B: 動いている最中に歯群が全部隠れたら、保持姿勢を直前の速度で進める（速度は減衰させる）
                            fusion_hold_pose = _predict_pose_by_velocity(fusion_hold_pose, pose_vel_t, pose_vel_rv)
                            pose_vel_t = pose_vel_t * COAST_VEL_DECAY
                            pose_vel_rv = pose_vel_rv * COAST_VEL_DECAY
                            poses_by_key["F"] = fusion_hold_pose.copy()
                        else:
                            poses_by_key["F"] = fusion_hold_pose.copy()
                        display_anchor = ""
                    if (FUSION_PNP_ANCHORS and fusion_off_f and fusion_pnp_info is not None
                            and fusion_pnp_info.get("ok") and fusion_pnp_info.get("anchors")):
                        display_anchor = "+".join(fusion_pnp_info["anchors"])   # ⑲ 姿勢に使ったアンカーを表示する
                active_anchor = "F"
            pose = poses_by_key[active_anchor]
            if (HIERARCHICAL_ANCHORS and GROUP_ROTATION_FROM_F and active_anchor != "F"
                    and anchor_switch_pose is not None):
                pose = _group_pose_with_rotation(active_anchor, pose, anchor_switch_pose)
                poses_by_key[active_anchor] = pose
            # ── 跳び量ゲート ──────────────────────────────────────────────────────
            # M3T のボディは遮蔽や弱いエッジで 1 フレームに数十 mm 飛ぶことがある（奥行き方向が特に弱い）。
            # 前フレームの出力から大きく跳んだ姿勢・画面外・異常な奥行きは採用せず、前フレーム姿勢に
            # 全ボディを戻してやり直させる。実物が本当に速く動いた場合は追従が遅れるだけで、ロスト判定と
            # 再ロックが通常どおり働く。
            depth_window_hit = False
            if prev_output_pose is not None and time.time() - last_recovery_time > 0.5:
                jump_m = float(np.linalg.norm(pose[:3, 3] - prev_output_pose[:3, 3]))
                jump_deg = rotation_angle_deg(pose, prev_output_pose)
                depth_now = float(pose[2, 3])
                center_now = projected_center(pose)
                off_screen = (center_now is None or center_now[0] < 0 or center_now[0] > w
                              or center_now[1] < 0 or center_now[1] > h)
                hard_bad = not (POSE_DEPTH_MIN_M < depth_now < POSE_DEPTH_MAX_M) or off_screen
                # 奥行きの窓（2026-10-14 13:12 run）: 誤ロックで F が台座模様に合いながら 1 s で奥行き 13→28 cm に滑った
                # （1 フレームの跳びは 2〜3 mm なので跳び量ゲートを通り、q 0.9 なので last_good も更新され続けた）。
                # 追跡開始時の奥行きを基準に、そこから大きく外れた姿勢は不正とみなす（デモ中にカメラ距離が 1.5 倍変わることはない）。
                if session_depth_ref is not None and not (
                        session_depth_ref * DEPTH_WINDOW_MIN_RATIO < depth_now < session_depth_ref * DEPTH_WINDOW_MAX_RATIO):
                    hard_bad = True
                    depth_window_hit = True
                jumped = jump_m > POSE_JUMP_MAX_TRANS_M or jump_deg > POSE_JUMP_MAX_ROT_DEG
                accept_jump = False
                if jumped and not hard_bad:
                    _tm = compute_tooth_mask(frame_img)
                    q_jump = float(np.sqrt(estimate_rigid_quality(dist_map, pose) * estimate_color_quality(_tm, pose)))
                    q_hold = float(np.sqrt(estimate_rigid_quality(dist_map, prev_output_pose)
                                           * estimate_color_quality(_tm, prev_output_pose)))
                    better = q_jump >= QUALITY_GOOD_THRESHOLD and q_jump >= q_hold + POSE_JUMP_ACCEPT_MARGIN
                    same_target = (jump_target_pose is not None
                                   and float(np.linalg.norm(pose[:3, 3] - jump_target_pose[:3, 3])) < 0.005
                                   and rotation_angle_deg(pose, jump_target_pose) < 5.0)
                    if better:
                        jump_target_streak = jump_target_streak + 1 if same_target else 1
                        jump_target_pose = pose.copy()
                    else:
                        jump_target_streak = 0
                        jump_target_pose = None
                    accept_jump = better and jump_target_streak >= POSE_JUMP_ACCEPT_FRAMES
                    if accept_jump:
                        print(f"跳び量ゲート: {jump_m*1000:.0f} mm / {jump_deg:.0f}° だが跳び先の方が良く合う "
                              f"(q {q_jump:.2f} vs 保持 {q_hold:.2f}, {jump_target_streak} フレーム連続) → 受け入れ")
                        jump_target_streak = 0
                        jump_target_pose = None
                elif not jumped:
                    jump_target_streak = 0
                    jump_target_pose = None
                if (jumped and not accept_jump) or hard_bad:
                    # 拒否: 前フレーム姿勢へ戻す。2026-10-01 14:18 の run では、この処理が誤って else 側に
                    # 入っていて「跳んでいない通常フレーム」まで毎回前フレームに戻していた（＝動きに追従しない）。
                    pose_jump_rejects += 1
                    gate_reject_streak += 1
                    if pose_jump_rejects <= 20 or pose_jump_rejects % 50 == 0:
                        print(f"跳び量ゲート: {jump_m*1000:.0f} mm / {jump_deg:.0f}° / 奥行き {depth_now*1000:.0f} mm"
                              f"{' / 画面外' if off_screen else ''} → 前フレーム姿勢に戻す "
                              f"(アンカー {active_anchor}, 累計 {pose_jump_rejects})")
                    pose = prev_output_pose.copy()
                    reset_all_anchors(pose)
                    poses_by_key = {k: pose for k in poses_by_key}
                    if active_anchor != "F":
                        active_anchor = "F"       # 跳んだのが歯群なら F に戻す（歯群ボディは拘束が弱い）
                        anchor_switch_pose = None
                    if gate_reject_streak >= POSE_JUMP_STUCK_FRAMES:
                        print(f"跳び量ゲート: {gate_reject_streak} フレーム連続で拒否 → 保持姿勢が古いとみなして自動再ロック"
                              + ("（奥行き窓の外なので last_good を信用せず並進探索）" if depth_window_hit else ""))
                        force_relock = True
                        if depth_window_hit:
                            relock_fail_count = max(relock_fail_count, RELOCK_FAILS_BEFORE_SEARCH)
                        gate_reject_streak = 0
                else:
                    gate_reject_streak = 0
            # ① 出力速度の更新（惰性追跡の予測に使う）。直前フレームの品質が LOSS 以上のときだけ更新し、
            #    再ロック直後（prev_output_pose=None）や跳びは除外する。
            _pose_src = ("+".join(fusion_keys) if fusion_active and fusion_keys
                         else ("-" if fusion_off_f else active_anchor))
            _returned_to_f = (PREDICT_RESET_ON_RETURN_TO_F and _pose_src == "F"
                              and prev_pose_src is not None and prev_pose_src != "F")
            prev_pose_src = _pose_src
            if _returned_to_f:
                # 統合／保持 → F の切替で出力が跳ねる分を速度に入れない（予測の自己維持で奥行きが滑った）
                pose_vel_t = np.zeros(3)
                pose_vel_rv = np.zeros(3)
            elif prev_output_pose is not None and snap_frames == 0 and last_quality >= QUALITY_LOSS_THRESHOLD:
                _dtv = (pose[:3, 3] - prev_output_pose[:3, 3]).astype(np.float64)
                if float(np.linalg.norm(_dtv)) <= COAST_VEL_MAX_M:
                    _drv, _ = cv2.Rodrigues(pose[:3, :3].astype(np.float64) @ prev_output_pose[:3, :3].astype(np.float64).T)
                    pose_vel_t = COAST_VEL_ALPHA * _dtv + (1.0 - COAST_VEL_ALPHA) * pose_vel_t
                    pose_vel_rv = COAST_VEL_ALPHA * _drv.ravel() + (1.0 - COAST_VEL_ALPHA) * pose_vel_rv
            elif prev_output_pose is None or snap_frames > 0:
                pose_vel_t = np.zeros(3)
                pose_vel_rv = np.zeros(3)
            prev_output_pose = pose.copy()
            # ロスト/良好判定は単体アンカーの値ではなく剛体整合スコア（A/B/C 平均）で行う。
            # 1フレームの値は ±0.1 程度ばらつくので EMA で平滑化した値を判定に使う。
            quality_raw = estimate_rigid_quality(dist_map, pose)
            quality_raw_f = quality_raw   # F ジオメトリでの剛体整合（歯群のブーストなし）。last_good 更新はこの厳しい値で
            if LOST_SCORE_MODE == "top2":
                # ② 1 歯群が隠れただけで HOLD に落ちないように、A/B/C/D 上位 2 つの平均も見る（判定にだけ使う）
                quality_raw = max(quality_raw, estimate_rigid_quality(dist_map, pose, keys=LOST_SCORE_LENIENT_KEYS,
                                                                      top_k=LOST_SCORE_LENIENT_TOP_K))
            # ── 歯でない場所への誤ロック検知（F 追跡中だけ。統合中は手が歯を覆うので判定しない）──
            nontooth_reason = ""
            if NONTOOTH_CHECK_ENABLED and not fusion_off_f and active_anchor == "F" and snap_frames == 0:
                if frame_idx % NONTOOTH_EVAL_EVERY == 0:
                    _fill = estimate_fill_quality(compute_tooth_mask(frame_img), pose)
                    nontooth_fill_ema = _fill if nontooth_fill_ema is None else (
                        NONTOOTH_FILL_EMA_ALPHA * _fill + (1.0 - NONTOOTH_FILL_EMA_ALPHA) * nontooth_fill_ema)
                _now = time.time()
                if (nontooth_fill_ema is not None and nontooth_fill_ema < NONTOOTH_FILL_MIN
                        and last_quality >= NONTOOTH_MIN_QUALITY):
                    if nontooth_last_t is not None:
                        nontooth_low_sec += min(_now - nontooth_last_t, 0.2)
                    nontooth_reason = f"model projects onto non-tooth pixels ({nontooth_fill_ema:.2f}) for {nontooth_low_sec:.1f}s - likely false lock"
                    _trigger = (nontooth_low_sec >= NONTOOTH_SEC
                                or (nontooth_fill_ema < NONTOOTH_FILL_SEVERE and nontooth_low_sec >= NONTOOTH_SEVERE_SEC))
                    if _trigger and not force_relock:
                        print(f"誤ロック検知: q={last_quality:.2f} なのに F 内部点の歯色一致 {nontooth_fill_ema:.2f} < {NONTOOTH_FILL_MIN} が "
                              f"{nontooth_low_sec:.1f} s 続いた → 並進探索で再ロック")
                        force_relock = True
                        relock_fail_count = max(relock_fail_count, RELOCK_FAILS_BEFORE_SEARCH)   # last_good も汚れているので近傍ではなく並進探索へ
                        frame_event = (frame_event + ";" if frame_event else "") + f"nontooth_relock fill{nontooth_fill_ema:.2f}"
                        nontooth_low_sec = 0.0
                        nontooth_fill_ema = None
                else:
                    nontooth_low_sec = 0.0   # 歯色に乗っている（または q が低い）→ 疑いを解除
                nontooth_last_t = _now
            else:
                # 統合中／SNAP 中は評価しない。タイマーは止めるだけで、F に戻ったら続きから数える
                # （13:12 run では F ↔ 統合が 1 s ごとに入れ替わり、毎回リセットされて 13 s の誤ロックを見逃した）
                nontooth_last_t = None
                if snap_frames > 0:
                    nontooth_low_sec = 0.0
                    nontooth_fill_ema = None
            if HIERARCHICAL_ANCHORS and fusion_active and fusion_keys:
                # 統合中は、使っている歯群の適合度の平均も「追跡できている」根拠に含める
                quality_raw = max(quality_raw, float(np.mean([quality_by_key.get(k, 0.0) for k in fusion_keys])))
            elif HIERARCHICAL_ANCHORS and active_anchor != "F":
                # 副アンカーで姿勢を出している間は、その歯群の適合度も「追跡できている」根拠に含める
                # （F の適合度だけだと遮蔽中は必ず低く、HOLD/再ロックに落ちて切替の意味が無くなる）
                quality_raw = max(quality_raw, quality_by_key.get(active_anchor, 0.0))
            quality_smoothed = (QUALITY_EMA_ALPHA * quality_raw
                                + (1.0 - QUALITY_EMA_ALPHA) * quality_smoothed)
            last_quality = quality_smoothed

            if last_quality >= QUALITY_GOOD_THRESHOLD:
                lost_streak = 0
                lost_start_time = None
                # 歯群で姿勢を出している間は、歯群の適合度でブーストされた「良好」で基準姿勢を更新しない。
                # 更新すると、流れていく歯群の姿勢が切替検査・逸脱監視・HOLD の基準になってしまう。
                if quality_raw_f >= QUALITY_GOOD_THRESHOLD:
                    last_good_pose = pose.copy()
                    last_good_time = time.time()
                    if session_depth_ref is None and not fusion_off_f:
                        session_depth_ref = float(pose[2, 3])
                        print(f"奥行き基準を設定: {session_depth_ref*100:.1f} cm（許容 {session_depth_ref*DEPTH_WINDOW_MIN_RATIO*100:.0f}〜"
                              f"{session_depth_ref*DEPTH_WINDOW_MAX_RATIO*100:.0f} cm）")
                recovery_escalated = False
                # 自動登録用のストリークは GOOD よりさらに厳しい閾値で数える
                good_streak = good_streak + 1 if last_quality >= AUTO_CAPTURE_MIN_QUALITY else 0
            elif last_quality < QUALITY_LOSS_THRESHOLD:
                lost_streak += 1
                good_streak = 0
                if lost_start_time is None:
                    lost_start_time = time.time()
            else:
                # デッドゾーン: 「連続して悪い」を数えたいので、ここに居る間は少しずつ戻す。
                # 以前は維持していたため、GOOD に届かない状態で飛び飛びの低下が蓄積し、
                # 合っているのに 8 回目で誤リカバリ → 0.3〜0.5 秒フリーズ → 50〜100mm 飛ぶ、
                # を繰り返していた（2026-09-09 17:01 のログで 14 回）。
                lost_streak = max(0, lost_streak - 1)
                if lost_streak == 0:
                    lost_start_time = None

            if (AUTO_POSE_CAPTURE_ENABLED and good_streak >= AUTO_CAPTURE_MIN_GOOD_STREAK
                    and time.time() - last_auto_capture_time >= AUTO_CAPTURE_COOLDOWN_SEC
                    and time.time() - last_recovery_time >= AUTO_CAPTURE_AFTER_RECOVERY_SEC
                    and is_pose_novel_for_capture(pose)):
                target_slot = AUTO_CAPTURE_SLOTS[auto_capture_cursor % len(AUTO_CAPTURE_SLOTS)]
                auto_capture_cursor += 1
                save_pose_slot(target_slot, pose)
                last_auto_capture_time = time.time()
                good_streak = 0
                print(f"自動姿勢登録: スロット{target_slot}に保存"
                      f"（アンカー{active_anchor}, quality={last_quality:.2f}、"
                      f"良品質が{AUTO_CAPTURE_MIN_GOOD_STREAK}フレーム継続）")

            # ⑦ 統合から抜け出せない状態: 歯は見えているのに PnP が使えず、F も戻れないまま統合が続いている
            if (FUSION_STUCK_RELOCK and fusion_off_f and anchor_switch_pose is not None and not force_relock
                    and fusion_pnp_bad_since is not None):
                _now_s = time.time()
                _qA, _qB = quality_by_key.get("A", 0.0), quality_by_key.get("B", 0.0)
                if (_now_s - anchor_switch_time >= FUSION_STUCK_MIN_FUSION_SEC
                        and _now_s - fusion_pnp_bad_since >= FUSION_STUCK_PNP_BAD_SEC
                        and _qA >= FUSION_STUCK_GROUP_Q and _qB >= FUSION_STUCK_GROUP_Q
                        and quality_by_key.get("F", 0.0) < ANCHOR_F_RECOVER_QUALITY
                        and _now_s - fusion_stuck_relock_time >= FUSION_STUCK_COOLDOWN_SEC):
                    print(f"統合から抜け出せない: 統合 {_now_s - anchor_switch_time:.0f} s・PnP なし "
                          f"{_now_s - fusion_pnp_bad_since:.0f} s、A={_qA:.2f} B={_qB:.2f} は見えているのに "
                          f"F={quality_by_key.get('F', 0.0):.2f} が戻らない → 全探索で再ロック")
                    force_relock = True
                    relock_fail_count = max(relock_fail_count, 1)   # 0 だと直前の良好姿勢の近く（回転制限）だけを探す
                    relock_fail_count = min(relock_fail_count, RELOCK_FAILS_BEFORE_SEARCH - 1)
                    fusion_stuck_relock_time = _now_s
                    frame_event = (frame_event + ";" if frame_event else "") + "fusion_stuck_relock"

            if QUALITY_CHECK_ENABLED:
                now = time.time()
                # 自動再ロックは「連続して悪い」かつ「HOLD で待っても戻らなかった」ときだけ。
                # 遮蔽は数秒で去ることが多く、その間は last_good_pose を保持した方が全探索より良い。
                held_long_enough = (lost_start_time is not None
                                    and now - lost_start_time >= HOLD_BEFORE_RECOVERY_SEC)
                # ⑫ 遮蔽中で PnP が解けている間は見失っていない（出力は PnP で保持）→ 品質低下での再ロックは待つ
                pnp_holding = (FUSION_PNP_DEFER_RELOCK and fusion_off_f and fusion_pnp is not None
                               and fusion_pnp_bad_since is None and anchor_switch_pose is not None
                               and now - anchor_switch_time < FUSION_PNP_DEFER_MAX_SEC)
                if force_relock or (lost_streak >= QUALITY_LOSS_CONSECUTIVE and held_long_enough and not pnp_holding
                                    and now - last_recovery_time >= RECOVERY_COOLDOWN_SEC):
                    force_relock = False
                    use_search = relock_fail_count >= RELOCK_FAILS_BEFORE_SEARCH
                    use_guard = last_good_pose is not None and relock_fail_count == 0
                    print(f"追跡品質低下 ({last_quality:.2f} < {QUALITY_LOSS_THRESHOLD}) → 自動再ロック "
                          f"({'並進探索' if use_search else ('guard=on' if use_guard else '全探索')}, "
                          f"連続失敗 {relock_fail_count})")
                    _auto_select_info["quality"] = 0.0
                    auto_pose, auto_slot = pose_auto_select(
                        frame_img, dist_map, use_rotation_guard=use_guard and not use_search,
                        reference_pose=last_good_pose if use_guard else None,
                        translation_search=use_search,
                    )
                    pose = auto_pose
                    poses_by_key = {k: auto_pose for k in poses_by_key}
                    if auto_slot is not None:
                        selected_pose_slot = auto_slot
                    now = time.time()           # 探索に 0.2〜5 s かかるので取り直す
                    last_recovery_time = now
                    relock_q = _auto_select_info["quality"]
                    if relock_q >= QUALITY_LOSS_THRESHOLD:
                        last_good_pose = auto_pose  # リカバリ pose を基準にして次の同期で使う
                        last_good_time = now
                        quality_smoothed = 1.0      # スナップ後の最初の判定で即ロストにならないように
                        lost_streak = 0
                        lost_start_time = None
                        relock_fail_count = 0
                    else:
                        # 失敗: 基準姿勢は更新せず、品質も見つけた値のまま。次回はクールダウン後すぐ再試行
                        # （lost_streak/lost_start_time は維持）し、段階を上げる。
                        if lost_start_time is None or now - lost_start_time >= RELOCK_ESCALATE_AFTER_SEC:
                            relock_fail_count += 1
                        quality_smoothed = relock_q
                        print(f"再ロック失敗 (q={relock_q:.2f}, {relock_fail_count} 回連続) → "
                              f"次回は{'並進探索' if relock_fail_count >= RELOCK_FAILS_BEFORE_SEARCH else '全探索'}")
                    recovery_escalated = use_guard
                    snap_frames = SNAP_FRAME_COUNT
                    relock_reason = "auto"
                    # 再ロック先を次フレームの跳び量ゲートが「256 mm の跳び」として元に戻してしまった
                    # （2026-10-01 12:44。pose_auto_select に 0.4 s かかり時間ガード 0.5 s を抜けた）
                    prev_output_pose = None
                    _pf_mark("relock")   # pose_auto_select は 5 候補 × 8 step で 200 ms 前後かかる（仕様）

        # ── アンカー同期（品質連動）────────────────────────────────────────────
        # 4グループは同一剛体なので全アンカーが同一 pose を持つべき。
        # ただし「悪い pose を全員に押しつける」と収束不可になるため、
        # 品質に応じて同期先を変える:
        #   SNAP中 / quality 良好 → 現フレームの pose で同期（最新の追跡結果を全員に配布）
        #   quality 低下中        → last_good_pose で同期（最後の良好位置から全員が再探索）
        #   history なし          → 表示だけ統一（M3T 独立探索をそのまま使う）
        last_good_fresh = (last_good_pose is not None
                           and time.time() - last_good_time <= LAST_GOOD_MAX_AGE_SEC)
        holding = False
        hold_text = ""
        coasting = False
        if snap_frames > 0 or last_quality >= QUALITY_LOSS_THRESHOLD or not last_good_fresh:
            # 品質OK / スナップ中 / last_good_pose が古すぎる → 現在の pose で同期。
            # 古い last_good_pose へ毎フレーム引き戻すと「固まってから飛ぶ」動きになる。
            # 案 B: 統合中、または F 追跡中でも速く動いているときは、速度で 1 フレーム先を予測した姿勢から次の step を始める
            _start = pose
            if PREDICT_ENABLED and snap_frames == 0:
                _vel_t = pose_vel_t
                if not fusion_off_f and not PREDICT_F_DEPTH:
                    _vel_t = pose_vel_t.copy()
                    _vel_t[2] = 0.0   # F 追跡中は奥行きを予測しない（M3T が引き戻せず自己維持で滑る）
                if fusion_off_f or (PREDICT_F_ENABLED and float(np.linalg.norm(_vel_t)) >= PREDICT_F_MIN_SPEED_M):
                    _start = _predict_pose_by_velocity(pose, _vel_t, pose_vel_rv)
            _f_free = False
            if F_FREE_IN_FUSION and fusion_off_f and group_poses_raw is not None:
                # ⑬ 統合中は F を出力へ戻さず自分で追わせる（手が離れたら F が自力で正しい姿勢へ収束し、q で復帰できるように）。
                #   出力から大きく離れたら（遮蔽で流れた）出力へ戻す
                _fr = group_poses_raw["F"]
                _d_mm = float(np.linalg.norm(_group_center_cam("F", _fr) - _group_center_cam("F", _start))) * 1000.0
                _f_free = (_d_mm <= F_FREE_MAX_MM and rotation_angle_deg(_fr, _start) <= F_FREE_MAX_DEG)
            if _f_free:
                for a in ANCHORS:
                    if a["key"] != "F":
                        a["model"].reset_pose_cv(_start)
            else:
                reset_all_anchors(_start)
            poses_by_key = {a["key"]: pose for a in ANCHORS}
            coast_decided = False
        else:
            holding = True
            held = time.time() - lost_start_time if lost_start_time is not None else 0.0
            if not coast_decided:
                # ① HOLD に入った瞬間に 1 回だけ判定: 直前まで模型が動いていたなら惰性で追う
                coast_decided = True
                _speed = float(np.linalg.norm(pose_vel_t))
                if COAST_ENABLED and _speed >= COAST_MIN_SPEED_M_PER_FRAME:
                    coast_until = time.time() + COAST_MAX_SEC
                    print(f"HOLD: 直前の速度 {_speed*1000:.2f} mm/フレーム ({_speed*1000*24:.0f} mm/s) → "
                          f"{COAST_MAX_SEC:.1f} s は凍結せず惰性で追う")
                else:
                    coast_until = 0.0
            if time.time() < coast_until:
                # ① 惰性追跡: last_good に戻さず、速度で 1 フレーム先を予測した姿勢から M3T を進める
                coasting = True
                hold_text = f"  COAST {held:.1f}s"
                mode_text = f"{mode_text} coast"
                _pred = _predict_pose_by_velocity(pose, pose_vel_t, pose_vel_rv)
                pose_vel_t = pose_vel_t * COAST_VEL_DECAY
                pose_vel_rv = pose_vel_rv * COAST_VEL_DECAY
                reset_all_anchors(_pred)
                poses_by_key = {a["key"]: _pred for a in ANCHORS}
            else:
                # HOLD: 遮蔽の可能性があるので最後の良好姿勢から毎フレーム再試行する。
                # 遮蔽が去ればそのまま再ロック、HOLD_BEFORE_RECOVERY_SEC を過ぎれば自動再ロックへ。
                hold_text = f"  HOLD {held:.1f}s"
                mode_text = f"{mode_text} hold"
                reset_all_anchors(last_good_pose)
                poses_by_key = {a["key"]: last_good_pose for a in ANCHORS}

        # ── ② 回転不確か警告: 統合中（回転固定）が長い／大きく動いた、HOLD 中、F 復帰直後 ──
        rotation_uncertain_reason = ""
        if fusion_off_f and anchor_switch_pose is not None:
            _since = time.time() - anchor_switch_time
            _ref_pose = fusion_hold_pose if fusion_hold_pose is not None else anchor_switch_pose
            _moved = float(np.linalg.norm(_group_center_cam("F", _ref_pose) - _group_center_cam("F", anchor_switch_pose)))
            if _since >= ROT_UNCERTAIN_AFTER_SEC or _moved >= ROT_UNCERTAIN_MOVE_M:
                _rot_text = (f"rotation following group {fusion_rot_src} ({fusion_rot_total_deg:.0f} deg so far, weak)"
                             if fusion_rot_src else "while rotation is frozen")
                rotation_uncertain_reason = f"occluded {_since:.0f}s, arch moved {_moved*1000:.0f}mm, {_rot_text}"
        elif return_search_job is not None:
            rotation_uncertain_reason = (f"re-checking rotation after return "
                                         f"({len(return_search_job['results'])}/{len(return_search_job['cands'])})")
        elif nontooth_reason:
            rotation_uncertain_reason = nontooth_reason
        elif coasting:
            rotation_uncertain_reason = "lost sight - coasting on predicted motion (verify alignment)"
        elif holding:
            rotation_uncertain_reason = "holding last good pose (lost)"
        elif time.time() < rotation_uncertain_until:
            rotation_uncertain_reason = rotation_uncertain_after_return

        # マーカー基準との差はログ書き込みの前に計算する（HUD 側はこの値を表示するだけ）
        if marker_ref is not None and marker_ref.is_registered():
            marker_err = marker_ref.error(pose)
            # 基準が古い疑い: F が良好に追跡しているのに誤差が大きい状態が続く（ボードか模型が動いた）
            _stale_now = (marker_err is not None and not fusion_off_f and not holding
                          and quality_by_key.get("F", 0.0) >= MARKER_STALE_MIN_QF
                          and marker_err["mm"] > MARKER_STALE_ERR_MM)   # marker_err["mm"] は mm 単位
            if _stale_now:
                marker_stale_since = marker_stale_since or time.time()
                if not marker_stale_warned and time.time() - marker_stale_since >= MARKER_STALE_SEC:
                    marker_stale_warned = True
                    # 12:45 run: 復帰時再探索が誤った局所解に落とし q 0.92 のまま 32 mm ずれ続けた例があり、同じ条件で出る。
                    # 「基準が古い」とは限らないので、両方の可能性を示す
                    marker_status_text = (f"ArUco disagrees by {marker_err['mm']:.0f} mm / {marker_err['deg']:.0f} deg while q is high: "
                                          "tracker may be in a wrong local minimum (re-align), or the reference is stale (re-register)")
                    print(f"ArUco と不一致: F q={quality_by_key.get('F', 0.0):.2f} なのに誤差 {marker_err['mm']:.0f} mm / {marker_err['deg']:.0f}° が "
                          f"{MARKER_STALE_SEC:.0f} s 以上続いている → トラッカーが誤った局所解にいる（Quick Align で合わせ直し）か、"
                          "ボード/模型が動いて基準が古い（登録し直し）")
            else:
                marker_stale_since = None
                if marker_err is not None and marker_err["mm"] <= MARKER_STALE_ERR_MM * 0.5:
                    marker_stale_warned = False   # 登録し直して誤差が戻ったら、次に古くなったときまた警告できるように
        # mode 列の [..]: 統合中の姿勢源。統合で F を離れているのに使える歯群がない（保持/予測）フレームは "-"
        # （以前は "F" と出て、遮蔽なしの F 追跡と区別できなかった）
        _src_label = ("+".join(fusion_keys) if fusion_active and fusion_keys
                      else ("-" if fusion_off_f else active_anchor))
        log_frame(frame_idx, f"{mode_text}[{_src_label}]",
                  last_quality, pose, quality_by_key, quality_raw,
                  marker_err, group_poses=group_poses_raw, event=frame_event,
                  rot_uncertain=int(bool(rotation_uncertain_reason)),
                  fus_rot_src=fusion_rot_src if fusion_off_f else "",
                  fus_rot_deg=fusion_rot_total_deg if fusion_off_f else None,
                  fill_q=nontooth_fill_ema,
                  fus_motion_px=fusion_rot_motion_px if fusion_off_f else None,
                  fus_pnp=fusion_pnp_info if fusion_off_f else None)
        if diag_frame_dir is not None and frame_idx % DIAG_FRAME_DUMP_EVERY == 0:
            # ディスク書き込みはメインループから外す（数百 ms 固まる原因になり得る）
            threading.Thread(target=cv2.imwrite,
                             args=(os.path.join(diag_frame_dir, f"f{frame_idx:06d}.jpg"), frame_raw.copy(),
                                   [cv2.IMWRITE_JPEG_QUALITY, 90]), daemon=True).start()
        _pf_mark("log")

        if HEADLESS:
            show = _HEADLESS_SHOW
        elif view_mode == 0:
            show = viewer.view_color(False)
        elif view_mode == 1:
            show = draw_dual_wireframe(frame_img, poses_by_key, active_anchor)
        else:
            rendered = viewer.view_color(False)
            wire = draw_dual_wireframe(frame_img, poses_by_key, active_anchor)
            show = cv2.addWeighted(rendered, 0.45, wire, 0.55, 0)
        _pf_mark("render")

        quality_color = (0, 255, 0) if last_quality >= QUALITY_GOOD_THRESHOLD else (0, 165, 255)
        cv2.putText(show, f"FIT QUALITY: {last_quality:.2f} (raw {quality_raw:.2f}, anchor {active_anchor})  lost_streak: {lost_streak}{hold_text}",
                    (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
        q_text = "   ".join(f"{a['key']}:q={quality_by_key.get(a['key'], 0.0):.2f}" for a in ANCHORS)
        if HIERARCHICAL_ANCHORS:
            q_text = f"POSE from {active_anchor}   highlight {display_anchor or 'ALL'}   " + q_text
        cv2.putText(show, q_text, (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # マーカー基準との差（登録済みなら mm/deg、未登録なら登録を促す）
        if marker_ref is not None:
            if not HEADLESS:
                marker_ref.draw(show)
            if marker_ref.last is None:
                ref_text, ref_color = "REF: marker not visible", (120, 120, 120)
            elif marker_err is not None:
                _s = marker_ref.summary(marker_err["key"])
                _reg = (f"n={_s['n_used']}/{_s['n']}, spread {_s['rms_mm']:.1f}mm/{_s['rms_deg']:.1f}deg"
                        if _s and _s["n"] > 1 else f"n={_s['n'] if _s else 0}")
                ref_text = (f"REF({marker_err['key']}, {_reg}): err {marker_err['mm']:.1f} mm / {marker_err['deg']:.1f} deg"
                            f"   (dx,dy,dz)=({marker_err['dx_mm'][0]:+.1f},{marker_err['dx_mm'][1]:+.1f},{marker_err['dx_mm'][2]:+.1f})")
                ref_color = (255, 255, 0)
                # マーカーから予測した歯牙姿勢を白い細線で重ねる（追跡結果との視覚比較用）。
                # 階層アンカーでは F だけ描けば全歯を覆う。
                for a in ANCHORS:
                    if HEADLESS or (HIERARCHICAL_ANCHORS and a["key"] != "F"):
                        continue
                    _draw_wireframe_into(show, marker_err["predicted"].astype(np.float32),
                                         a["viz_draw_vertices"], a["viz_draw_edges"], (255, 255, 255), 1)
            else:
                ref_text, ref_color = f"REF({marker_ref.last['key']}) detected - press 'm' when aligned to register", (0, 200, 255)
            cv2.putText(show, ref_text, (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ref_color, 1)

        # 品質が悪いフレームは last_good_pose を送り続け、Unity 側のモデルが
        # 飛ばないようにする（この判定はプレビュー表示や FIT QUALITY 自体には
        # 影響しない。送信内容だけをフリーズさせる）。
        pose_to_smooth = pose
        if last_good_pose is not None and last_quality < QUALITY_LOSS_THRESHOLD:
            pose_to_smooth = last_good_pose
        display_pose = smooth_pose(pose_to_smooth)
    else:
        rotation_uncertain_reason = ""   # ADJUST 中は警告なし
        show = _HEADLESS_SHOW if HEADLESS else draw_dual_wireframe(frame_img, {a["key"]: pose for a in ANCHORS}, active_anchor)
        if marker_ref is not None:
            if not HEADLESS:
                marker_ref.draw(show)
            # ADJUST 中も登録状態を表示（'M' で消えたか、何件登録済みかが分かるように）
            if marker_ref.last is None:
                _t, _c = "REF: marker not visible", (120, 120, 120)
            elif marker_ref.is_registered():
                _s = marker_ref.summary(marker_ref.last["key"])
                _t = (f"REF({_s['key']}): registered n={_s['n_used']}/{_s['n']}"
                      + (f", spread {_s['rms_mm']:.1f}mm/{_s['rms_deg']:.1f}deg" if _s["n"] > 1 else "")
                      + "   ('x' clears all, Backspace undoes one)")
                _c = (255, 255, 0)
            else:
                _t, _c = f"REF({marker_ref.last['key']}) detected - no registration yet (press 'm' when aligned)", (0, 200, 255)
            cv2.putText(show, _t, (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _c, 1)
        if len(LANDMARK_POINTS_MODEL) >= 4:
            R_cur, t_cur = pose[:3, :3], pose[:3, 3]
            pts_cam = (R_cur @ LANDMARK_POINTS_MODEL.T).T + t_cur
            for i, p in enumerate(pts_cam):
                if p[2] <= 0.05:
                    continue
                px = int(round(K[0, 0] * p[0] / p[2] + K[0, 2]))
                py = int(round(K[1, 1] * p[1] / p[2] + K[1, 2]))
                # 咬頭=黄(鋭い点、狙いやすい)、歯の縁=シアン(輪郭のだいたいの端)
                color = (0, 255, 255) if LANDMARK_IS_CUSP[i] else (255, 255, 0)
                cv2.circle(show, (px, py), 6, color, 2)
                cv2.putText(show, str(i + 1), (px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if click_init_active:
                for i, (cx, cy) in enumerate(click_points):
                    cv2.circle(show, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                cur_is_cusp = LANDMARK_IS_CUSP[len(click_points)] if len(click_points) < len(LANDMARK_IS_CUSP) else True
                kind = "cusp(sharp)" if cur_is_cusp else "edge(approx ok)"
                cv2.putText(show, f"click point #{len(click_points)+1}/{len(LANDMARK_POINTS_MODEL)} [{kind}] (see 'reference' window)",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        mode_text = "ADJUST"
        display_pose = pose

    send_video_frame_to_unity(frame_img)   # このフレームの姿勢と組で送る（上の poll_unity_commands の前のコメント参照）
    _pf_mark("video")
    send_pose_to_unity(display_pose, mode_text)

    cv2.putText(show, f"MODE: {mode_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(show, f"VIEW: {VIEW_MODES[view_mode]}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    _anchor_summary = "  ".join(f"{a['key']}={a['name']}" for a in ANCHORS)
    cv2.putText(show, f"SLOT: {selected_pose_slot}  {_anchor_summary}", (20, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    if not tracking_active:
        cv2.putText(show, "a/d q/e w/s=move  i/k j/l u/o=rotate  c=click-init(PnP)  f=track(from-pose)  p=save  Space=track(safe)  g=track(global)  ESC=quit",
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    _pf_mark("draw")
    if not HEADLESS:
        cv2.imshow("few_teeth_tracking", show)
        if _ref_view["dirty"]:
            cv2.imshow("reference", _render_reference())
            _ref_view["dirty"] = False
    _pf_mark("imshow")
    _pf_end()

_grabber.stop()
cap.release()
if recorder is not None and recorder.active:
    recorder.stop()
if recorder is not None:
    _t_flush = time.monotonic()
    while not recorder.q.empty() and time.monotonic() - _t_flush < 10.0:   # 書き出し待ち（録画の途中で終了したとき）
        time.sleep(0.05)
    time.sleep(0.3)   # 最後の項目（stop）の書き出し
if PNP_DUMP_PATH:
    import pickle
    _keep = ("Ks", "X", "born", "born_poses", "born_uv", "R0", "t0")
    with open(PNP_DUMP_PATH, "wb") as _f:
        pickle.dump({"segments": [{k: s[k] for k in _keep} for s in _pnp_dump_segments],
                     "frames": _pnp_dump_frames, "F_centroid": ANCHOR_BY_KEY["F"]["centroid"]}, _f)
    print(f"PnP 診断を保存: {PNP_DUMP_PATH}（区間 {len(_pnp_dump_segments)}、フレーム {len(_pnp_dump_frames)}）")
log_file.close()
if unity_udp_sock is not None:
    unity_udp_sock.close()
if command_socket is not None:
    command_socket.close()
with video_clients_lock:
    for client in video_clients:
        try:
            client.close()
        except OSError:
            pass
    video_clients.clear()
cv2.destroyAllWindows()
print(f"終了。ログ: {log_path}")
