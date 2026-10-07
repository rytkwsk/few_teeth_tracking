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
PYM3T_DIR = r"C:\Users\rytkw\Projects\pym3t"
sys.path.insert(0, PYM3T_DIR)

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import csv
import json
import socket
import struct
import threading
import traceback
import time

import cv2
import numpy as np
import pym3t

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
FUSION_F_RETURN_SEC       = 2.0    # F のスコアがこれだけ連続して回復閾値以上なら歯群の見え方によらず F へ復帰
FUSION_NO_MEMBER_SEC      = 1.0    # 使える歯群がこれだけ無く F が OK 閾値以上なら F へ戻す
ANCHOR_F_OK_QUALITY       = 0.45   # F の単体適合度がこれ以上なら F を出力（切替しない）
ANCHOR_F_RECOVER_QUALITY  = 0.55   # 歯群へ切替後、F がこれ以上に戻ったら F へ復帰（0.45 で復帰すると往復した）
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
USE_TEXTURE_MODALITY = False

# 2アンカーは同じ typodont 座標系を共有するため姿勢スロットも共有でよいが、
# 単独運用の tracker_premolar_canine.py / tracker_canine.py の initial_poses/
# を汚さないよう、Anchor Switching 専用のキーで別ディレクトリに保存する。
_model_key        = "four_group_molar_premolar_canine_incisor"
pose_slots_dir    = os.path.join(SCRIPT_DIR, "initial_poses", _model_key)
calibration_path  = os.path.join(PYM3T_DIR, "camera_calibration.npz")
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
UNITY_VIDEO_PORT    = 5010
UNITY_VIDEO_FPS     = 20   # エンコードは別スレッドなので 10→20（背景のカクつき低減）
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
DIAG_FRAME_DUMP_EVERY = 20
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
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if CAMERA_WIDTH and CAMERA_HEIGHT:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    _aw, _ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (_aw, _ah) != (CAMERA_WIDTH, CAMERA_HEIGHT):
        print(f"カメラ解像度: 要求 {CAMERA_WIDTH}x{CAMERA_HEIGHT} に対し実際は {_aw}x{_ah}"
              "（配信アプリ側の解像度設定を確認）")

if CAMERA_MANUAL_EXPOSURE is not None:
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_MANUAL_EXPOSURE)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("カメラの読み取りに失敗しました")

current_exposure = CAMERA_MANUAL_EXPOSURE if CAMERA_MANUAL_EXPOSURE is not None else cap.get(cv2.CAP_PROP_EXPOSURE)

if CAMERA_MANUAL_EXPOSURE is not None:
    actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    actual_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    print(f"手動露出を設定: 要求={CAMERA_MANUAL_EXPOSURE} 実際の値(exposure={actual_exposure}, auto_exposure={actual_auto})")
    print("  ドライバによっては反映されないことがある。'[' で露出を下げる/']' で上げる（実行中に調整可能）。")

h, w = frame.shape[:2]

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

if os.path.exists(calibration_path):
    calib = np.load(calibration_path)
    calib_w = int(calib["image_width"][0])
    calib_h = int(calib["image_height"][0])
    if calib_w == w and calib_h == h:
        K    = calib["K"].astype(np.float32)
        dist = calib["dist"].astype(np.float32)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
        undistort_map = cv2.initUndistortRectifyMap(
            K, dist, None, new_K.astype(np.float32), (w, h), cv2.CV_16SC2)
        K = new_K.astype(np.float32)
        print("カメラキャリブレーション読み込み完了")
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
    marker_ref = MarkerReference(K, None, os.path.join(pose_slots_dir, "marker_registration.json"),
                                 scale=MARKER_PRINT_SCALE)
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


def estimate_rigid_quality(dist_map, candidate_pose, keys=None):
    """剛体整合スコア: 同じ候補姿勢を RIGID_QUALITY_KEYS の各アンカーのジオメトリで
    評価し、上位 RIGID_QUALITY_TOP_K の平均を返す。
    アンカー単体の最大値だと、3本歯のアンカーが歯列の別の歯に乗っただけで 1.0 に
    なり「4グループが同一剛体である」という制約を全く見ないため、ロスト/良好判定と
    自動姿勢選択の採点にはこちらを使う。"""
    keys = keys or RIGID_QUALITY_KEYS
    qs = sorted((estimate_anchor_quality(dist_map, candidate_pose, ANCHOR_BY_KEY[k]) for k in keys),
                reverse=True)
    top = qs[:max(1, min(RIGID_QUALITY_TOP_K, len(qs)))]
    return float(np.mean(top))


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
        "pose_anchor": ("+".join(fusion_keys) if fusion_active and fusion_keys else active_anchor),
        "quality": [round(float(last_anchor_quality.get(k, 0.0)), 3) for k in ("F", "A", "B", "C", "D")],
        "timestamp": time.time(),
        "mode": mode_text,
        "pose_cv": pose_mat.astype(float).reshape(16).tolist(),
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
            data, _ = command_socket.recvfrom(1024)
        except BlockingIOError:
            break
        except OSError as e:
            print(f"Unity コマンド UDP エラー: {e}")
            break

        command = data.decode("utf-8", errors="ignore").strip()
        cmd_lower = command.lower()
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
])

diag_frame_dir = None
if DIAG_FRAME_DUMP_EVERY > 0:
    diag_frame_dir = log_path[:-4] + "_frames"
    os.makedirs(diag_frame_dir, exist_ok=True)


def log_frame(frame_idx, mode_text, quality, pose_mat, quality_by_key=None, quality_raw_value=float("nan"),
              marker_err=None):
    R, t = pose_mat[:3, :3], pose_mat[:3, 3]
    qk = quality_by_key or {}
    if marker_err is not None:
        P = marker_err["predicted"]
        rv, _ = cv2.Rodrigues(P[:3, :3].astype(np.float64))
        ref_cols = [marker_err["key"], f"{marker_err['mm']:.3f}", f"{marker_err['deg']:.3f}",
                    f"{P[0,3]:.6f}", f"{P[1,3]:.6f}", f"{P[2,3]:.6f}",
                    f"{rv[0,0]:.6f}", f"{rv[1,0]:.6f}", f"{rv[2,0]:.6f}",
                    str(marker_err.get("n_samples", ""))]
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
    ret, raw = cap.read()
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
    if marker_ref is not None:
        marker_ref.detect(frame_raw)
        if MARKER_MASK_FOR_TRACKER and marker_ref.last is not None:
            # 以降の追跡・品質評価・表示はボードを消した画像で行う（基準が追跡を汚さないように）
            frame_img = marker_ref.mask_out(frame_raw, pad_px=MARKER_MASK_PAD_PX, method=MARKER_MASK_METHOD)
    _pf_mark("marker")
    frame_idx += 1

    send_video_frame_to_unity(frame_img)
    _pf_mark("video")
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
        if marker_ref is None:
            print("マーカー基準は無効です（MARKER_REF_ENABLED=False）")
        elif marker_ref.last is None:
            print("マーカーが検出されていないため登録できません")
        elif tracking_active and (time.time() - last_recovery_time < 3.0 or lost_streak > 0):
            # 再ロック直後は quality_smoothed が 1.0 に初期化されていて低品質判定をすり抜ける。
            # 2026-09-21 19:02 に HOLD/SNAP ループ中の姿勢が 2 件登録された事故があったため拒否する。
            print("再ロック直後 / 品質低下中は登録できません。数秒安定してから押してください")
        elif tracking_active and snap_frames > 0:
            # 2026-09-09: 自動リカバリ中（SNAP）にズレた姿勢を登録してしまった事故があったため拒否する
            print("SNAP（再ロック）中は登録できません。追跡が落ち着いて合っているのを確認してから押してください")
        elif tracking_active and last_quality < QUALITY_LOSS_THRESHOLD:
            print(f"品質が低い（{last_quality:.2f} < {QUALITY_LOSS_THRESHOLD}）ため登録しません。"
                  "PnP クリック初期化の直後（ADJUST のまま）に押すのが確実です")
        else:
            k = marker_ref.register(pose)
            s = marker_ref.summary(k)
            print(f"マーカー基準にサンプル追加: {k}（{marker_ref.last['n_markers']} 枚, 再投影 {marker_ref.last['rms_px']:.2f}px）"
                  f" → 登録 {s['n']} 件中 {s['n_used']} 件（外れ値除外後）の平均を基準に使用。"
                  f" ばらつき: RMS {s['rms_mm']:.1f} mm / {s['rms_deg']:.1f} deg, 最大 {s['max_mm']:.1f} mm / {s['max_deg']:.1f} deg")
            if s["n"] < 3:
                print("  → 視点・距離を変えて PnP 初期化をやり直し、あと数回 'm' を押すと偶発誤差が平均化されます（Backspace で直近 1 件を取り消し）")
    elif key == 8:  # Backspace: 直近の登録サンプルを 1 件取り消す
        if marker_ref is not None:
            k = marker_ref.undo_last_sample()
            s = marker_ref.summary(k) if k else None
            print(f"マーカー基準サンプルを 1 件取り消し: {k} → 残り {s['n'] if s else 0} 件" if k else "取り消せるサンプルがありません")
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
        pose = poses_by_key[active_anchor]
        quality_by_key = {}  # SNAPウォームアップ中は未評価（HUD表示は0.0のまま）
        if snap_frames > 0:
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
                    # F で追跡中。F が落ちたら F を離れ、出力を最後の良好姿勢へ戻す（HOLD と同じ扱い。
                    # 遮蔽で流れた F 姿勢と last_good の差を跳び量ゲートに拒否させないため prev_output_pose を外す）。
                    if qF_ < ANCHOR_F_OK_QUALITY and last_good_pose is not None and now_t >= group_block_until:
                        fusion_off_f = True
                        anchor_switch_pose = last_good_pose.copy()
                        anchor_switch_time = now_t
                        fusion_hold_pose = last_good_pose.copy()
                        fusion_members = {}
                        fusion_enter_streak = {}
                        fusion_f_ok_since = None
                        fusion_no_member_since = None
                        prev_output_pose = None
                        print(f"統合: F を離脱 (F={qF_:.2f})")
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
                        fusion_off_f = False
                        fusion_active = False
                        fusion_keys = []
                        fusion_members = {}
                        fusion_enter_streak = {}
                        fusion_hold_pose = None
                        anchor_switch_pose = None
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
                    if sane:
                        fusion_no_member_since = None
                        fused = _fuse_groups(sane, poses_by_key, anchor_switch_pose)
                        cF = ANCHOR_BY_KEY["F"]["centroid"]
                        drift_m = float(np.linalg.norm((fused[:3, :3] @ cF + fused[:3, 3])
                                                       - (anchor_switch_pose[:3, :3] @ cF + anchor_switch_pose[:3, 3])))
                        if drift_m > ANCHOR_GROUP_MAX_DRIFT_M:
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
                        if fusion_no_member_since is None:
                            fusion_no_member_since = now_t
                        if fusion_active:
                            print("統合: 使える歯群なし → 直前姿勢を保持"
                                  + (f" (除外 {' '.join(dropped)})" if dropped else ""))
                        fusion_active = False
                        fusion_keys = []
                        poses_by_key["F"] = fusion_hold_pose.copy()
                        display_anchor = ""
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
            if prev_output_pose is not None and time.time() - last_recovery_time > 0.5:
                jump_m = float(np.linalg.norm(pose[:3, 3] - prev_output_pose[:3, 3]))
                jump_deg = rotation_angle_deg(pose, prev_output_pose)
                depth_now = float(pose[2, 3])
                center_now = projected_center(pose)
                off_screen = (center_now is None or center_now[0] < 0 or center_now[0] > w
                              or center_now[1] < 0 or center_now[1] > h)
                hard_bad = not (POSE_DEPTH_MIN_M < depth_now < POSE_DEPTH_MAX_M) or off_screen
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
                        print(f"跳び量ゲート: {gate_reject_streak} フレーム連続で拒否 → 保持姿勢が古いとみなして自動再ロック")
                        force_relock = True
                        gate_reject_streak = 0
                else:
                    gate_reject_streak = 0
            prev_output_pose = pose.copy()
            # ロスト/良好判定は単体アンカーの値ではなく剛体整合スコア（A/B/C 平均）で行う。
            # 1フレームの値は ±0.1 程度ばらつくので EMA で平滑化した値を判定に使う。
            quality_raw = estimate_rigid_quality(dist_map, pose)
            quality_raw_f = quality_raw   # F ジオメトリでの剛体整合（歯群のブーストなし）
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

            if QUALITY_CHECK_ENABLED:
                now = time.time()
                # 自動再ロックは「連続して悪い」かつ「HOLD で待っても戻らなかった」ときだけ。
                # 遮蔽は数秒で去ることが多く、その間は last_good_pose を保持した方が全探索より良い。
                held_long_enough = (lost_start_time is not None
                                    and now - lost_start_time >= HOLD_BEFORE_RECOVERY_SEC)
                if force_relock or (lost_streak >= QUALITY_LOSS_CONSECUTIVE and held_long_enough
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
        if snap_frames > 0 or last_quality >= QUALITY_LOSS_THRESHOLD or not last_good_fresh:
            # 品質OK / スナップ中 / last_good_pose が古すぎる → 現在の pose で同期。
            # 古い last_good_pose へ毎フレーム引き戻すと「固まってから飛ぶ」動きになる。
            reset_all_anchors(pose)
            poses_by_key = {a["key"]: pose for a in ANCHORS}
        else:
            # HOLD: 遮蔽の可能性があるので最後の良好姿勢から毎フレーム再試行する。
            # 遮蔽が去ればそのまま再ロック、HOLD_BEFORE_RECOVERY_SEC を過ぎれば自動再ロックへ。
            holding = True
            held = time.time() - lost_start_time if lost_start_time is not None else 0.0
            hold_text = f"  HOLD {held:.1f}s"
            mode_text = f"{mode_text} hold"
            reset_all_anchors(last_good_pose)
            poses_by_key = {a["key"]: last_good_pose for a in ANCHORS}

        # マーカー基準との差はログ書き込みの前に計算する（HUD 側はこの値を表示するだけ）
        if marker_ref is not None and marker_ref.is_registered():
            marker_err = marker_ref.error(pose)
        log_frame(frame_idx, f"{mode_text}[{'+'.join(fusion_keys) if fusion_active and fusion_keys else active_anchor}]",
                  last_quality, pose, quality_by_key, quality_raw,
                  marker_err)
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

cap.release()
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
