#!/usr/bin/env bash
# few_teeth_tracking を Mac（Apple Silicon / Intel）で動かすためのセットアップ。
#
# 前提: 次の 2 つを同じ親フォルダに git clone 済み
#     <親>/few_teeth_tracking
#     <親>/unity-few-teeth-tracking
# pym3t は <親>/pym3t に本家（Jianxff/pym3t）を clone し、tools/pym3t_patches/pym3t_local.patch を当ててビルドする。
#
# 使い方（few_teeth_tracking フォルダで）:
#     bash tools/setup_mac.sh                          # 足りないものを入れて pym3t をビルド
#     bash tools/setup_mac.sh --check                  # 何も変更せず点検だけ
#     bash tools/setup_mac.sh --bundle ~/m3t_bundle    # 研究室 PC の初期姿勢・ArUco 登録・region キャッシュも持ってくる
#     bash tools/setup_mac.sh --rebuild                # pym3t をビルドし直す
#
# やること:
#   1. Xcode Command Line Tools と Homebrew の確認（無ければ入れ方を表示して止まる）
#   2. brew で cmake eigen glfw glew opencv libomp を入れる
#   3. conda（無ければ brew で Miniforge）と環境 pym3t39 を用意（tools/environment_pym3t39.yml）
#   4. pym3t を clone → パッチ → <親>/pym3t/dist_mac にビルド → import を確認
#   5. few_teeth_tracking/m3t_local.env（PC ごとの設定。Unity とトラッカーの両方が読む）を作る
#   6. Unity 2022.3.48f1・カメラキャリブの有無を点検して、次にやることを表示する

set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PARENT="$(dirname "$REPO")"
PYM3T="$PARENT/pym3t"
DIST="$PYM3T/dist_mac"
BUILD="$PYM3T/build_mac"
PATCH="$REPO/tools/pym3t_patches/pym3t_local.patch"
ENV_NAME="pym3t39"
LOCAL_ENV="$REPO/m3t_local.env"
CHECK=0
REBUILD=0
BUNDLE=""
NG=0

while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK=1 ;;
    --rebuild) REBUILD=1 ;;
    --bundle) shift; BUNDLE="${1:-}" ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "不明な引数: $1"; exit 2 ;;
  esac
  shift
done

ok()   { printf '\033[32m[OK]\033[0m %s\n' "$*"; }
ng()   { printf '\033[33m[NG]\033[0m %s\n' "$*"; NG=$((NG + 1)); }
info() { printf '\033[36m---\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[中止]\033[0m %s\n' "$*"; exit 1; }

echo "=== few_teeth_tracking Mac セットアップ ==="
echo "macOS $(sw_vers -productVersion 2>/dev/null)  $(uname -m)  repo: $REPO"
[ "$(uname)" = "Darwin" ] || die "Mac 以外では tools/setup_pc.ps1（Windows）を使う"

# ── 0. リポジトリの配置 ──────────────────────────────────────────────
if [ -f "$PARENT/unity-few-teeth-tracking/Assets/Script/M3TProcessController.cs" ]; then
  ok "Unity リポジトリ: $PARENT/unity-few-teeth-tracking"
else
  ng "Unity リポジトリが $PARENT/unity-few-teeth-tracking に無い（git clone https://github.com/rytkwsk/unity-few-teeth-tracking.git）"
fi

# ── 1. Xcode Command Line Tools / Homebrew ───────────────────────────
if xcode-select -p >/dev/null 2>&1; then
  ok "Xcode Command Line Tools: $(xcode-select -p)"
else
  ng "Xcode Command Line Tools が無い → ターミナルで  xcode-select --install  を実行し、終わったらこのスクリプトをもう一度"
  die "コンパイラが無いので先に進めない"
fi

if ! command -v brew >/dev/null 2>&1; then
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
    [ -x "$b" ] && eval "$("$b" shellenv)" && break
  done
fi
if command -v brew >/dev/null 2>&1; then
  ok "Homebrew: $(brew --prefix)"
else
  ng "Homebrew が無い → https://brew.sh のコマンドをターミナルに貼って入れ、表示される 'Next steps' の 2 行も実行してから、このスクリプトをもう一度"
  die "Homebrew が無いので先に進めない"
fi

# ── 2. brew のライブラリ ──────────────────────────────────────────────
for pkg in cmake eigen glfw glew opencv libomp; do
  if brew list --formula "$pkg" >/dev/null 2>&1; then
    ok "brew $pkg"
  elif [ $CHECK -eq 1 ]; then
    ng "brew $pkg が無い"
  else
    info "brew install $pkg（opencv は依存が多く 5〜15 分かかる）"
    brew install "$pkg" || ng "brew install $pkg が失敗した"
  fi
done

# ── 3. conda と環境 ──────────────────────────────────────────────────
CONDA_ROOT=""
for r in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" \
         /opt/homebrew/Caskroom/miniforge/base /usr/local/Caskroom/miniforge/base \
         /opt/homebrew/Caskroom/miniconda/base /opt/miniconda3 /opt/anaconda3; do
  if [ -x "$r/bin/conda" ]; then CONDA_ROOT="$r"; break; fi
done
if [ -z "$CONDA_ROOT" ]; then
  if [ $CHECK -eq 1 ]; then
    ng "conda が無い"
  else
    info "conda が無いので Miniforge を入れる（brew install --cask miniforge）"
    brew install --cask miniforge || die "Miniforge のインストールに失敗"
    for r in /opt/homebrew/Caskroom/miniforge/base /usr/local/Caskroom/miniforge/base; do
      [ -x "$r/bin/conda" ] && CONDA_ROOT="$r" && break
    done
  fi
fi
PY=""
if [ -n "$CONDA_ROOT" ]; then
  ok "conda: $CONDA_ROOT"
  PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
  if [ ! -x "$PY" ]; then
    if [ $CHECK -eq 1 ]; then
      ng "conda 環境 $ENV_NAME が無い"
      PY=""
    else
      info "conda 環境 $ENV_NAME を作成（数分）"
      "$CONDA_ROOT/bin/conda" env create -f "$REPO/tools/environment_pym3t39.yml" -n "$ENV_NAME" \
        || die "conda env create が失敗した"
    fi
  fi
  if [ -n "$PY" ] && [ -x "$PY" ]; then
    ok "conda 環境 $ENV_NAME: $("$PY" -c 'import sys,platform;print(sys.version.split()[0], platform.machine())')  ($PY)"
  fi
fi

# ── 4. pym3t の取得・パッチ・ビルド ───────────────────────────────────
BASE="$(sed -n 's/^# pym3t .*base commit: \([0-9a-f]*\).*/\1/p' "$PATCH" | head -1)"
if [ ! -d "$PYM3T/.git" ]; then
  if [ $CHECK -eq 1 ]; then
    ng "pym3t が $PYM3T に無い"
  else
    info "pym3t を clone: $PYM3T"
    git clone -q https://github.com/Jianxff/pym3t "$PYM3T" || die "pym3t の clone に失敗"
  fi
fi
if [ -d "$PYM3T/.git" ]; then
  if git -C "$PYM3T" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    ok "pym3t パッチ適用済み（base ${BASE:0:7}）"
  elif [ $CHECK -eq 1 ]; then
    ng "pym3t にパッチが当たっていない"
  else
    if [ -n "$(git -C "$PYM3T" status --porcelain --untracked-files=no)" ]; then
      ng "pym3t に別の変更があるのでパッチを当てない（$PYM3T を確認）"
    else
      git -C "$PYM3T" checkout -q "$BASE" && git -C "$PYM3T" apply "$PATCH" \
        && ok "pym3t を ${BASE:0:7} にしてパッチを当てた" || ng "パッチが当たらなかった"
    fi
  fi
fi

SO="$(ls "$DIST"/pym3t*.so 2>/dev/null | head -1)"
if [ -n "$PY" ] && [ -x "$PY" ] && [ -d "$PYM3T/source" ] && { [ -z "$SO" ] || [ $REBUILD -eq 1 ]; } && [ $CHECK -eq 0 ]; then
  info "pym3t をビルド（数分。ログ: $BUILD/build.log）"
  BREW="$(brew --prefix)"
  OMP="$(brew --prefix libomp)"
  PYINC="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["include"])')"
  PYLIB="$(ls "$CONDA_ROOT/envs/$ENV_NAME/lib"/libpython3.9*.dylib 2>/dev/null | head -1)"
  [ -z "$PYLIB" ] && PYLIB="$(ls "$CONDA_ROOT/envs/$ENV_NAME/lib"/libpython3.9*.a 2>/dev/null | head -1)"
  mkdir -p "$BUILD" "$DIST"
  {
    cmake -S "$PYM3T/source" -B "$BUILD" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.10 \
      -DPYTHON_EXECUTABLE="$PY" -DPython_EXECUTABLE="$PY" \
      -DPYTHON_INCLUDE_DIR="$PYINC" -DPYTHON_LIBRARY="$PYLIB" \
      -DCMAKE_PREFIX_PATH="$BREW;$OMP" \
      -DOpenMP_ROOT="$OMP" \
      -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$OMP/include" \
      -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$OMP/include" \
      -DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp \
      -DOpenMP_omp_LIBRARY="$OMP/lib/libomp.dylib" \
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$DIST" \
    && cmake --build "$BUILD" --config Release -j "$(sysctl -n hw.ncpu)"
  } >"$BUILD/build.log" 2>&1
  if [ $? -ne 0 ]; then
    tail -40 "$BUILD/build.log"
    ng "pym3t のビルドが失敗した（上のエラーと $BUILD/build.log を Claude Code に見せる）"
  fi
  SO="$(ls "$DIST"/pym3t*.so 2>/dev/null | head -1)"
fi
if [ -n "$SO" ]; then
  ok "pym3t ビルド済み: $SO"
else
  ng "pym3t の .so が $DIST に無い"
fi

if [ -n "$SO" ] && [ -n "$PY" ] && [ -x "$PY" ]; then
  OUT="$(cd "$REPO" && PYM3T_DIR="$DIST" "$PY" -W ignore -c "
import sys; sys.path.insert(0, '$DIST')
import numpy as np, cv2, pym3t
print('import OK', 'cv2', cv2.__version__, 'aruco', hasattr(cv2, 'aruco'))
" 2>&1)"
  if echo "$OUT" | grep -q "import OK"; then ok "import pym3t: $(echo "$OUT" | tail -1)"; else ng "import pym3t が失敗: $OUT"; fi
fi

# ── 5. m3t_local.env ─────────────────────────────────────────────────
if [ -f "$LOCAL_ENV" ]; then
  ok "m3t_local.env あり（上書きしない）: $LOCAL_ENV"
elif [ $CHECK -eq 1 ]; then
  ng "m3t_local.env が無い"
else
  cat >"$LOCAL_ENV" <<EOF
# この PC 専用の設定（gitignore）。Unity（M3TProcessController）とトラッカー（camera_io.py）の両方が読む。
# 環境変数が設定されていればそちらが優先。
PYM3T_DIR=$DIST
M3T_CONDA_ENV=$ENV_NAME
M3T_PYTHON_EXE=$PY
# カメラ番号は python tools/list_cameras.py で調べて書く（連係カメラの iPhone は 0 とは限らない）
M3T_CAMERA_INDEX=0
# 1280x720 が取れないカメラなら実際の解像度にし、その解像度でキャリブし直す
M3T_CAMERA_WIDTH=1280
M3T_CAMERA_HEIGHT=720
EOF
  ok "m3t_local.env を作った: $LOCAL_ENV（M3T_CAMERA_INDEX はあとで直す）"
fi

# ── bundle（任意）────────────────────────────────────────────────────
if [ -n "$BUNDLE" ]; then
  [ -d "$BUNDLE" ] || die "bundle が無い: $BUNDLE"
  if [ -d "$BUNDLE/initial_poses" ]; then
    for d in "$BUNDLE/initial_poses"/*/; do
      name="$(basename "$d")"
      if [ -d "$REPO/initial_poses/$name" ]; then
        echo "  initial_poses/$name は既にあるのでコピーしない"
      elif [ $CHECK -eq 0 ]; then
        mkdir -p "$REPO/initial_poses" && cp -R "$d" "$REPO/initial_poses/$name" && echo "  initial_poses/$name をコピーした"
      fi
    done
  fi
  if [ -d "$BUNDLE/models_meta" ] && [ $CHECK -eq 0 ]; then
    for f in "$BUNDLE/models_meta"/*.meta; do
      [ -f "$REPO/models/$(basename "$f")" ] || cp "$f" "$REPO/models/"
    done
    echo "  region キャッシュ（models_meta）をコピーした"
  fi
  # dist_release（Windows 用 .pyd）は Mac では使えない。カメラキャリブも連係カメラでは取り直す
fi

# ── 6. 点検 ──────────────────────────────────────────────────────────
UNITY_APP="/Applications/Unity/Hub/Editor/2022.3.48f1/Unity.app"
if [ -d "$UNITY_APP" ]; then
  ok "Unity 2022.3.48f1: $UNITY_APP"
else
  ng "Unity 2022.3.48f1 が無い → Unity Hub を入れてから、ブラウザで  unityhub://2022.3.48f1/8bf49c377ebf  を開いてインストール"
fi

CALIB="$DIST/camera_calibration.npz"
if [ -f "$CALIB" ] && [ -n "$PY" ] && [ -x "$PY" ]; then
  ok "カメラキャリブ: $("$PY" -c "import numpy as np;d=np.load('$CALIB');print('%dx%d rms %.2f' % (int(d['image_width'].ravel()[0]), int(d['image_height'].ravel()[0]), float(d['rms'].ravel()[0])))")"
else
  ng "カメラキャリブが無い（$CALIB）→ 下の『次にやること』の 2"
fi

META_N="$(ls "$REPO/models"/*.meta 2>/dev/null | wc -l | tr -d ' ')"
if [ "$META_N" -ge 10 ]; then ok "region キャッシュ $META_N 個"; else echo "[--] region キャッシュ $META_N/10 個（初回起動で 5 分ほど生成。止めないこと）"; fi

echo
if [ $NG -eq 0 ]; then
  printf '\033[32mセットアップ完了。\033[0m\n'
else
  printf '\033[33mNG が %d 件。上の指示に従って直し、もう一度  bash tools/setup_mac.sh\033[0m\n' "$NG"
fi
cat <<EOF

=== 次にやること ===
1. iPhone を連係カメラとして使う（同じ Apple ID・Wi-Fi と Bluetooth オン・iPhone を横向きで固定）。
   どの番号が iPhone か調べる（初回はターミナルにカメラ許可のダイアログが出る → 許可）:
     cd "$REPO" && "$PY" tools/list_cameras.py
   logs/camera_probe/cam<番号>.jpg を見て、m3t_local.env の M3T_CAMERA_INDEX を直す。
2. その解像度でカメラキャリブ（tools/checkerboard_10x7_inner9x6.png を別の画面に大きく表示して 30〜40 枚）:
     "$PY" tools/calibrate_camera.py --width 1280 --height 720 --cols 9 --rows 6 --square-mm <実寸 mm> \\
         --image-dir "$PYM3T/calibration_images_mac" --output "$CALIB"
3. ターミナルから単体で動作確認（OpenCV のウィンドウが出る。q で終了）:
     "$PY" tracker_four_group.py
4. Unity Hub で unity-few-teeth-tracking を開き、AnchorSwitchingDemoScene で Play → 1. Launch。
   初回は Unity にカメラ許可のダイアログが出る → 許可（出ない／拒否したら システム設定 → プライバシーとセキュリティ → カメラ）。
EOF
