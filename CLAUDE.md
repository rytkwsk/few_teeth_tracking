# few_teeth_tracking — 引き継ぎメモ（Claude Code 用）

M3T（pym3t）で歯列模型を追跡し、Unity（`../unity-few-teeth-tracking`）に姿勢と映像を送るトラッカー。
用途はインプラント手術ガイドの研究で、**精度は safety-critical**。見た目や fit quality ではなく、
ArUco 基準の実測（mm / deg）で判断すること。

## 構成（3 つのフォルダ）

| フォルダ | 役割 | Git |
|---|---|---|
| `few_teeth_tracking/`（ここ） | `tracker_four_group.py`（本体）、`marker_reference.py`、`models/`、`analysis/`、`tools/` | rytkwsk/few_teeth_tracking |
| `../unity-few-teeth-tracking/` | Unity 2022.3.48f1。`M3TProcessController` がこのトラッカーを conda 経由で起動、`M3TPoseReceiver` が UDP 5005 で姿勢受信、映像は TCP 5010、コマンドは UDP 5006 | rytkwsk/unity-few-teeth-tracking |
| `../pym3t/` | M3T の Python バインディング（C++）。**remote は本家 Jianxff/pym3t で Git 管理されていない**。ビルド成果物・キャリブ・モデルはここにローカルに置かれている | なし |

## PC ごとに必要な設定（Git では同期されないもの）

1. **conda 環境名**: シーンの既定は `pym3t`。環境名が違う PC は環境変数 `M3T_CONDA_ENV` で上書き
   （研究室 PC は `pym3t39`）。Unity は起動時の環境変数しか見ないので、設定後は Unity Hub ごと再起動。
2. **`PYM3T_DIR`**: `pym3t.pyd`＋依存 DLL＋`camera_calibration.npz` を置いたフォルダ。未設定なら `../pym3t`。
   研究室 PC では Release ビルド一式 `../pym3t/dist_release/` を指す。**このフォルダを USB 等で持ち運べば他の PC でも同じ構成になる**
   （必要なのは VC++ 再頒布可能パッケージだけ）。`.pyd` と同じフォルダの DLL は Windows が自動で見つける。
3. **カメラキャリブ**: トラッカーは `PYM3T_DIR/camera_calibration.npz` を読み、**解像度（image_width/height）がカメラ設定
   （`CAMERA_WIDTH/HEIGHT`=1280×720）と一致しないと仮の K（fx=800）にフォールバック**する（スケールしない）。仮 K だと
   重畳が小さく・ずれて見え、Quick Align も合わない。同じ iPhone＋iVCam＋1280×720 なら npz は PC 間で流用できる。
   取り直すときは `tools/calibrate_camera.py --width 1280 --height 720 --cols 9 --rows 6 --square-mm 30 --image-dir <新フォルダ> --output <PYM3T_DIR>/camera_calibration.npz`
   （`tools/checkerboard_10x7_inner9x6.png` を画面に大きく表示して撮る。40 枚で rms 0.13 px だった）。
4. **ArUco 基準の登録** `initial_poses/<model_key>/marker_registration.json`（gitignore）: ボードが台座に固定されたままなら
   物理的な関係は同じなのでコピー可。動かしたら登録し直す（Unity パネルの Register marker を視点を変えて 3 回以上）。
5. **ログ** `logs/tracking_*.csv` と `logs/*_frames/`（生フレーム 20 枚ごと）は gitignore。各 PC に溜まる。

## カメラ（iPhone SE + iVCam）で分かっていること

- 仮想カメラなので **24 fps 固定、露出・フォーカスは PC から制御不可**（tracker の露出設定は無効）。
- 大学 WiFi 経由は不安定だった → **USB 接続**を標準に。PC 側に **Apple 公式サイト版 iTunes** が必要
  （Microsoft Store 版は Apple Mobile Device Support を登録しないため iVCam から使えない）。
- iPhone 側でフォーカス／露出をロックしないとボードがボケて ArUco が不安定になる。
- **背景が決定的**: 明るい木目の机では q が 0.3 台で再ロックを繰り返した。暗い無地のマット＋模型は机上＋近めで 0.6 台に回復。

## 起動と評価の手順

Unity: Play → 「1. Launch camera / tracker」→ Quick Align → Start Demo。初回は region model 生成で白画面 1〜2 分（止めない）。
評価: ボードを映し、合っている瞬間に Register marker × 3 以上 → パネルに毎フレームの誤差（mm/deg）が出る。
ログ集計: `python analysis/marker_stats.py`（誤差分布・q との相関）、`analysis/group_rot_stats.py`（歯群ごとの回転精度・復帰時再探索・警告の効き）、
`analysis/marker_detail.py`（姿勢源別・区間別）、`analysis/csv_stats.py`（q の分布）、`analysis/err_axes.py`（軸別の偏り/ゆらぎ）。
**ArUco の 1 フレーム反転（err 100 mm 超）は誤検出なので除外する。**

## 設計上の割り切りと既知の問題（2026-10-09 時点）

- fit quality **q は精度の指標ではない**（Spearman ≈ −0.2〜−0.3、q≥0.6 でも 36〜49% が 3〜5 mm 超、15° ずれても 0.6 出る）。
- 統合モード（FUSION）は **並進のみ更新・回転は F 離脱時のまま固定・奥行き固定**（165〜230 行の設計メモ）。
  遮蔽中に模型を回すと回転が追従せず、F 復帰後も古い回転の局所解に落ちる（実測 11〜16°、q は良好に見える＝静かな誤り）。
  対策として ① F 復帰時の回転再探索（`RETURN_ROT_SEARCH_*`、離脱 ≥1.5 s または重心 ≥6 mm 移動のときだけ、1 回 ≈0.3 s）と
  ② 回転不確か警告（`ROT_UNCERTAIN_*`、Unity に赤バナー）を入れた。①は大きなずれ（10°超）は直すが、q で採用判定するため
  たまに 4〜5° 悪化させることもある。
- ⑤ 歯群ボディ自身の回転精度（CSV `gA/gB/gD_rot_deg`）: 遮蔽なし A 1.9° / B 3.9° / D 4.7°、統合中 A 3.7° / B 6.5° / D 7.3°。
  **A（臼歯）の回転は F 並みに信頼できる**（10/1 に 23° と測って回転固定を決めた頃は仮 K＋テクスチャなしだった）。
  B の重心は奥行き方向に常に 6.5〜8 mm 偏る（メッシュと実物の差の疑い）。
- 遮蔽なし F の精度（条件が良いとき）: 並進中央値 1.6 mm / 回転 2.5°、p90 3.7 mm。
- `USE_TEXTURE_MODALITY` は既定 ON（2.7→1.6 mm に改善、m3t_step 6→15 ms だがカメラ 24 fps 律速で fps は不変）。
- pym3t のバインディングは M3T の反復回数・スケール・標準偏差を公開していない（M3T 既定で固定）。
- 閾値（`ANCHOR_GROUP_MIN_QUALITY=0.60` など）は 10/1〜10/5 に振動対策で実測調整された。根拠データなしに緩めない。

## 作業ルール

- **Unity の Play 中に C# を編集しない**（再コンパイルで受信ソケットが残り、以降ポート 5005 が bind できなくなった。
  Preferences → Script Changes While Playing を "Recompile After Finished Playing" に）。Python の変更は Launch し直すだけで反映。
- トラッカーの子プロセスが固まって残ると PC 再起動が必要になることがある（受信ソケットは継承不可にして再発防止済み）。
- `../pym3t/dist_release/` と `build_release/` はビルド成果物。Git に入れない。

## 次のステップ

1. 計測条件を整える（iVCam フォーカスロック、ArUco ボードを硬い板に、模型は机上）→ 基準を取り直す。
2. 統合中に A の回転を使う（A が統合メンバーで q≥0.6 が 3 フレーム続いたら固定回転を A へ平滑化して寄せる。
   1 フレーム 2° まで・累計 25° まで・A が落ちたら停止）→ ArUco で判定。
3. 半歯列（左右 6〜7 本）のボディを階層に追加（Blender で分割 → .meta 生成）。遮蔽ロバスト性の本命。
4. pym3t ラッパーに `set_n_corr_iterations` / `set_scales` / `set_standard_deviations` を公開して再ビルド（復帰時だけ粗→細）。
5. 模型とメッシュの一致確認（黄色い歯・ガーゼ・歯冠際の赤テープ）、60 fps の UVC カメラの検討。

## 2026-10-09 の変更（研究室 PC で実施）

Release 再ビルド（VS2022 + vcpkg、`dist_release/`）、1280×720 再キャリブ、カメラ読み取りの別スレッド化（最新フレームのみ）、
Unity へ K/画像サイズ送信（`M3TCameraIntrinsics` が実行時に同期）、`marker_register`/`marker_undo` コマンドとパネル表示、
復帰時回転再探索、回転不確か警告、CSV に歯群ごとの精度列・event・rot_uncertain を追加。
