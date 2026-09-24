# few_teeth_tracking

少数歯牙クラスタ（対象歯＋隣接 1〜2 本）を対象とした AR 根管治療支援用トラッキングサブプロジェクト。

親プロジェクト `pym3t` の `test_tracker.py` を踏台に、以下の 3 スクリプトで構成する。

## ファイル構成

```
few_teeth_tracking/
├── tracker.py              # メイントラッカー（クラスタモデル専用）
├── extract_cluster_mesh.py # フル歯列 OBJ からサブメッシュ切り出し
├── evaluate_dual.py        # フル歯列 vs クラスタの同時追跡・誤差評価
├── models/                 # クラスタ OBJ と .meta ファイルを置く
├── initial_poses/          # クラスタ用初期姿勢スロット (.npy)
└── logs/                   # CSV トラッキングログ（自動生成）
```

---

## セットアップ手順

### 1. クラスタメッシュの切り出し

まず `extract_cluster_mesh.py --info` でフル歯列モデルの座標範囲を確認する:

```bash
cd few_teeth_tracking
python extract_cluster_mesh.py --input ../models/typodont.obj --info
```

切り出し方法は 2 種類ある:

**`segment_teeth.py`（推奨）** — ウォーターシェッド分割で歯と歯の間の谷（解剖学的境界）を
自動検出し、対象歯の歯冠を一切削らずに切り出す。中心座標に近い歯から N 本を選ぶ:

```bash
python segment_teeth.py \
    --input ../models/typodont.obj \
    --output models/cluster.obj \
    --search-xmin -0.9 --search-xmax 0.9 \
    --search-zmin -1.4 --search-zmax -0.5 \
    --target-x 0.0 --target-z -1.15 --num-teeth 3
```

`--search-x/z-min/max` は対象歯＋隣接歯を含む広めの探索範囲、
`--target-x/z` は残したい歯列の中心付近の座標、`--num-teeth` は残す本数。

**`extract_cluster_mesh.py`（バウンディングボックス版）** — 6 平面 bisect による直方体カット。
境界は直線になるが、対象歯の隣にある歯がボックスの側面で寸断されることがある
（対象歯自体がボックス範囲にきっちり収まっている場合のみ使う）:

```bash
python extract_cluster_mesh.py \
    --input ../models/typodont.obj \
    --output models/cluster.obj \
    --xmin -0.5 --xmax 0.5 \
    --ymin -0.3 --ymax 0.5 \
    --zmin -0.1 --zmax 0.5
```

> **ヒント**: `test_tracker.py` を起動してワイヤーフレームモード(v キー)で座標感覚をつかんでから範囲を決めると効率的。

#### サンプル歯牙パターン（`models/`）

typodont.obj の歯列全体を調べたところ、歯冠（局所的な高さのピーク）は
中央〜片側にかけて連続して並んでおり、反対側の奥は大部分が歯の無い
歯肉のみの範囲、その先端に孤立した奥歯が1本あるだけだった
（`--target-x/z` で座標を変えれば別の歯にも対応可能）。
下記は `segment_teeth.py` で位置を変えて切り出した例:

| ファイル | 内容 | target-x, target-z |
|---|---|---|
| `cluster.obj` | 前歯（中切歯付近）3本 | 0.0, -1.15 |
| `cluster_molar_far.obj` | 最も奥（遠心側）の大臼歯 | -1.2, 0.7 |
| `cluster_molar_premolar.obj` | 大臼歯〜小臼歯の移行部 3本 | -1.0, -0.05 |
| `cluster_premolar_canine.obj` | 小臼歯〜犬歯の移行部 3本 | -0.75, -0.65 |
| `cluster_canine_lateral.obj` | 犬歯〜側切歯の移行部 3本 | -0.5, -1.05 |
| `cluster_right_canine.obj` | 反対側の犬歯付近 2〜3本 | 0.5, -0.9 |
| `cluster_molar_isolated.obj` | 歯列の反対端、孤立した奥歯 | 1.3, 1.05 |

`cluster_boxcut_v1.obj` / `cluster_legacy_manual.obj` は旧方式（直方体カット /
手動頂点削除）で切り出した過去バージョンの参考用バックアップ。

> **注意**: 歯列の右奥（x > 0.9 付近）はほぼ全域が歯の無い歯肉のみで、
> 孤立した突起がひとつあるだけ（実際の歯ではなく顎の隅/枝の一部と思われる）。
> そのため「孤立した奥歯」パターンは用意していない。

#### 歯冠のみ（歯肉を除く）の抽出

`--crown-only-output` を指定すると、通常のクラスタ出力に加えて
歯肉を除いた歯冠のみの OBJ も書き出す（歯根はスキャンデータに
含まれないため近似）。各歯ごとに「ピーク高さ - crown-depth」を
しきい値にして残すため、歯ごとに独立した基準になり境界がなめらか:

```bash
python segment_teeth.py --input ../models/typodont.obj \
    --output models/cluster.obj \
    --crown-only-output models/cluster_crownonly.obj --crown-depth 0.35 \
    --search-xmin -0.9 --search-xmax 0.9 --search-zmin -1.4 --search-zmax -0.5 \
    --target-x 0.0 --target-z -1.15 --num-teeth 3
```

歯冠のみにすると歯の本数だけ独立した島に分かれるのが正常（歯根がないため
歯同士は元々つながっていない）。`--crown-depth` は各歯のピーク高さから
何ユニット下まで歯冠とみなすかの深さ（デフォルト 0.35）。
`models/cluster_crownonly_demo.obj` に前歯クラスタでの実行例あり。

`--output`（歯肉付きの通常クラスタ）と `--crown-only-output`（歯冠のみ）は
常に別ファイルとして両方保存される。歯冠のみに切り分ける前の、歯肉が
繋がった状態のモデルが必要な場合は `--output` 側（例: `cluster.obj` /
`cluster_molar_premolar.obj` など）を使う。

**実装メモ**: 境界 (-2) 頂点は、稜線に沿って連結してさえいれば
シードからの距離に制限なく取り込まれる。歯冠のみ抽出時にこれをそのまま
使うと、この歯と無関係な遠くの境界まで拾って、切断面ぎりぎりの高さで
孤立した尖った破片（裏側のギザギザ）ができることがあったため、
`crown_only_mesh_bisect` では境界頂点も対象歯自身のシードから
`--max-radius` 以内に制限している。

### 2. M3T メタファイルの生成

`tracker.py` を初回起動すると `.meta` ファイルが自動生成される（数分かかる）:

```bash
python tracker.py
```

生成物:
- `models/cluster.obj.region.meta`
- `models/cluster.obj.depth.meta`（depth を使う場合）

### 3. 初期姿勢の設定

1. `tracker.py` を起動
2. キーボードでモデルを動かし（マウス or キー）、カメラ映像と合わせる
3. `s` キーでスロット 1 に保存
4. `Space` で追跡開始

---

## 操作キー一覧

### tracker.py

| キー | 動作 |
|---|---|
| Space | 追跡開始 / 停止 |
| 1〜9, 0, -, = | 姿勢スロット切り替え（計 12 スロット） |
| s | 現在の姿勢を現在のスロットに保存 |
| v | 表示モード切替（M3T / WIREFRAME / BLEND） |
| q | 終了 |

### evaluate_dual.py

| キー | 動作 |
|---|---|
| Space | 追跡開始 / 停止 |
| 1〜9 | フル歯列モデルの姿勢スロット切り替え |
| !@#... (Shift+1〜9) | クラスタモデルの姿勢スロット切り替え |
| s | 両モデルの姿勢を現在スロットに保存 |
| q | 終了 |

---

## 評価指標（CSV ログ）

`logs/eval_dual_YYYYMMDD_HHMMSS.csv` に以下が記録される:

| カラム | 説明 |
|---|---|
| frame | フレーム番号 |
| timestamp | UNIX 時刻 |
| full_quality | フル歯列モデルの追跡品質スコア（0〜1） |
| cluster_quality | クラスタモデルの追跡品質スコア（0〜1） |
| trans_error_mm | 並進誤差 (mm)（フル歯列を基準） |
| rot_error_deg | 回転誤差 (deg)（フル歯列を基準） |
| full_lost_streak | フル歯列モデルの連続品質低下フレーム数 |
| cluster_lost_streak | クラスタモデルの連続品質低下フレーム数 |

---

## 既知の課題と対策（設計メモより）

| 課題 | 対策 |
|---|---|
| 前歯は長軸回転の拘束情報が乏しい | 隣接歯を含めてクラスタ化し輪郭情報を増やす |
| バキュームやバーによる遮蔽 | クランプをクラスタに加え非対称性を補う（将来作業） |
| ラバーダム単独分離時に隣接歯が見えない | 複数歯分離か、クランプモデルとの組み合わせを検討 |

---

## 依存関係

親プロジェクト `pym3t` と同じ環境で動作する（追加インストール不要）:
- Python 3.9
- OpenCV (`cv2`)
- NumPy
- `pym3t.cp39-win_amd64.pyd`（親ディレクトリを自動的に sys.path に追加）
