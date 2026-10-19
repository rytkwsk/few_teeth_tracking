<#
few_teeth_tracking を新しい PC（OMEN・自宅 PC など）で動かすためのセットアップと点検。

前提: 次の 2 つを同じ親フォルダに git clone 済みであること
    <親>\few_teeth_tracking
    <親>\unity-few-teeth-tracking

使い方（PowerShell で few_teeth_tracking フォルダから）:
    powershell -ExecutionPolicy Bypass -File tools\setup_pc.ps1 -Bundle E:\m3t_bundle
    powershell -ExecutionPolicy Bypass -File tools\setup_pc.ps1 -CheckOnly      # 何も変更せず点検だけ

-Bundle には研究室 PC で作った USB 用フォルダを渡す。中身:
    dist_release\      pym3t.pyd（Release）+ 依存 DLL + camera_calibration.npz（1280x720）
    initial_poses\     初期姿勢スロットと ArUco 登録（台座とボードの関係が同じ間だけ有効）
    models_meta\       region model のキャッシュ（任意。無ければ初回起動時に 5 分ほどかけて生成）

やること（-CheckOnly のときは点検だけ）:
    1. bundle の dist_release を <親>\pym3t\dist_release へ、initial_poses と models_meta を few_teeth_tracking へコピー
       （既にあるものは上書きしない。上書きしたいときは -Force）
    2. conda 環境 pym3t39 が無ければ tools\environment_pym3t39.yml から作成（要インターネット）
    3. ユーザー環境変数 M3T_CONDA_ENV / PYM3T_DIR を設定
    4. import pym3t・キャリブ解像度・VC++ ランタイム・Unity 2022.3.48f1・iVCam・Apple Mobile Device Support を点検
#>
param(
    [string]$Bundle = "",
    [string]$EnvName = "pym3t39",
    [switch]$CheckOnly,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$parent = Split-Path -Parent $repo
$pym3tDir = Join-Path $parent "pym3t\dist_release"
$results = New-Object System.Collections.Generic.List[object]

function Add-Result([string]$item, [bool]$ok, [string]$detail) {
    $results.Add([pscustomobject]@{ Item = $item; OK = $(if ($ok) { "OK" } else { "NG" }); Detail = $detail })
    $color = if ($ok) { "Green" } else { "Yellow" }
    Write-Host ("[{0}] {1}: {2}" -f $(if ($ok) { "OK" } else { "NG" }), $item, $detail) -ForegroundColor $color
}

function Copy-IfMissing([string]$src, [string]$dst, [string]$label) {
    if (-not (Test-Path $src)) { return }
    if ((Test-Path $dst) -and -not $Force) {
        Write-Host "  $label は既にあるのでコピーしない（上書きは -Force）: $dst"
        return
    }
    if ($CheckOnly) { Write-Host "  (CheckOnly) $label をコピーする予定: $src -> $dst"; return }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    Copy-Item -Recurse -Force $src $dst
    Write-Host "  $label をコピーした: $dst"
}

Write-Host "=== few_teeth_tracking セットアップ ($env:COMPUTERNAME) ===" -ForegroundColor Cyan
Write-Host "repo: $repo"

# ── 0. リポジトリの配置 ───────────────────────────────────────────────
$unityRepo = Join-Path $parent "unity-few-teeth-tracking"
Add-Result "Unity リポジトリ" (Test-Path (Join-Path $unityRepo "Assets\Script\M3TProcessController.cs")) $unityRepo

# ── 1. bundle からコピー ──────────────────────────────────────────────
if ($Bundle) {
    if (-not (Test-Path $Bundle)) { throw "bundle が見つからない: $Bundle" }
    Write-Host "--- bundle からコピー: $Bundle"
    Copy-IfMissing (Join-Path $Bundle "dist_release") $pym3tDir "dist_release"
    $ipSrc = Join-Path $Bundle "initial_poses"
    if (Test-Path $ipSrc) {
        Get-ChildItem $ipSrc -Directory | ForEach-Object {
            Copy-IfMissing $_.FullName (Join-Path $repo "initial_poses\$($_.Name)") "initial_poses\$($_.Name)"
        }
    }
    $metaSrc = Join-Path $Bundle "models_meta"
    if (Test-Path $metaSrc) {
        Get-ChildItem $metaSrc -Filter *.meta | ForEach-Object {
            Copy-IfMissing $_.FullName (Join-Path $repo "models\$($_.Name)") "models\$($_.Name)"
        }
    }
}

# ── 2. conda と環境 ───────────────────────────────────────────────────
$condaCandidates = @(
    (Join-Path $env:USERPROFILE "anaconda3\condabin\conda.bat"),
    (Join-Path $env:USERPROFILE "miniconda3\condabin\conda.bat"),
    (Join-Path $env:ProgramData "anaconda3\condabin\conda.bat"),
    (Join-Path $env:ProgramData "miniconda3\condabin\conda.bat")
)
$conda = $condaCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $conda) {
    $cmd = Get-Command conda.bat -ErrorAction SilentlyContinue
    if ($cmd) { $conda = $cmd.Source }
}
Add-Result "conda" ([bool]$conda) $(if ($conda) { $conda } else { "見つからない → Anaconda か Miniconda を入れてから再実行" })

$envPython = $null
if ($conda) {
    $condaRoot = Split-Path -Parent (Split-Path -Parent $conda)
    $envPython = Join-Path $condaRoot "envs\$EnvName\python.exe"
    if (-not (Test-Path $envPython)) {
        if ($CheckOnly) {
            Add-Result "conda 環境 $EnvName" $false "無い（-CheckOnly なので作らない）"
        } else {
            Write-Host "--- conda 環境 $EnvName を作成（数分かかる）"
            & $conda env create -f (Join-Path $PSScriptRoot "environment_pym3t39.yml") -n $EnvName
            if ($LASTEXITCODE -ne 0) { throw "conda env create が失敗した（exit $LASTEXITCODE）" }
        }
    }
    if (Test-Path $envPython) {
        $ver = & $envPython -c "import sys; print('%d.%d.%d' % sys.version_info[:3])"
        Add-Result "conda 環境 $EnvName" ($ver -like "3.9.*") "python $ver ($envPython)"
    }
}

# ── 3. 環境変数 ───────────────────────────────────────────────────────
$wantPym3t = (Resolve-Path $pym3tDir -ErrorAction SilentlyContinue)
$wantPym3t = if ($wantPym3t) { $wantPym3t.Path } else { $pym3tDir }
foreach ($pair in @(@("M3T_CONDA_ENV", $EnvName), @("PYM3T_DIR", $wantPym3t))) {
    $name, $value = $pair
    $cur = [Environment]::GetEnvironmentVariable($name, "User")
    $same = ($cur -eq $value)
    if (-not $same -and $cur -and $name -eq "PYM3T_DIR") {
        # 表記ゆれ（/ と \、大文字小文字、末尾の \）は同じ場所とみなす
        try { $same = ([IO.Path]::GetFullPath($cur).TrimEnd('') -ieq [IO.Path]::GetFullPath($value).TrimEnd('')) } catch { }
    }
    if (-not $same -and -not $CheckOnly) {
        [Environment]::SetEnvironmentVariable($name, $value, "User")
        Write-Host "  ユーザー環境変数 $name = $value（以前: $cur）"
        $cur = $value
        $same = $true
    }
    Add-Result "環境変数 $name" $same "現在値 '$cur'（期待 '$value'）"
}

# ── 4. 点検 ──────────────────────────────────────────────────────────
$pyd = Join-Path $pym3tDir "pym3t.cp39-win_amd64.pyd"
Add-Result "pym3t.pyd（Release）" (Test-Path $pyd) $pyd

foreach ($dll in "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "vcomp140.dll") {
    $p = Join-Path $env:WINDIR "System32\$dll"
    if (-not (Test-Path $p)) {
        Add-Result "VC++ ランタイム" $false "$dll が無い → https://aka.ms/vs/17/release/vc_redist.x64.exe を入れる"
        break
    }
    if ($dll -eq "vcomp140.dll") { Add-Result "VC++ ランタイム" $true "System32 に 4 DLL あり" }
}

if ($envPython -and (Test-Path $envPython) -and (Test-Path $pyd)) {
    $code = @"
import sys, os
sys.path.insert(0, r'$pym3tDir')
os.add_dll_directory(r'$pym3tDir')
import pym3t, numpy as np, cv2
print('import OK', 'cv2', cv2.__version__, 'aruco', hasattr(cv2, 'aruco'))
p = os.path.join(r'$pym3tDir', 'camera_calibration.npz')
if os.path.exists(p):
    d = np.load(p)
    w = int(np.asarray(d['image_width']).ravel()[0]) if 'image_width' in d else -1
    h = int(np.asarray(d['image_height']).ravel()[0]) if 'image_height' in d else -1
    print('calib', w, h)
else:
    print('calib missing')
"@
    # Windows PowerShell 5.1 は native コマンドの stderr を 2>&1 で受けるとエラー扱いにするので、ここだけ Continue にする
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $out = & $envPython -W ignore -c $code 2>&1 | Out-String
    $ErrorActionPreference = $prevEap
    $importOk = $out -match "import OK"
    Add-Result "import pym3t" $importOk ($out.Trim() -split "`r?`n" | Select-Object -First 1)
    if (-not $importOk) { Write-Host $out }
    if ($out -match "calib (\d+) (\d+)") {
        Add-Result "カメラキャリブ" ($Matches[1] -eq "1280" -and $Matches[2] -eq "720") "$($Matches[1])x$($Matches[2])（1280x720 でないと仮 K で動いて重畳がずれる）"
    } else {
        Add-Result "カメラキャリブ" $false "camera_calibration.npz が dist_release に無い"
    }
}

$unityExe = "C:\Program Files\Unity\Hub\Editor\2022.3.48f1\Editor\Unity.exe"
Add-Result "Unity 2022.3.48f1" (Test-Path $unityExe) $(if (Test-Path $unityExe) { $unityExe } else { "Unity Hub から 2022.3.48f1 を入れる" })

$ivcam = @("C:\Program Files\e2eSoft\iVCam\iVCam.exe", "C:\Program Files (x86)\e2eSoft\iVCam\iVCam.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
Add-Result "iVCam" ([bool]$ivcam) $(if ($ivcam) { $ivcam } else { "e2esoft.com から PC 版を入れる" })

$amds = Test-Path "HKLM:\SOFTWARE\Apple Inc.\Apple Mobile Device Support"
Add-Result "Apple Mobile Device Support" $amds $(if ($amds) { "あり（iVCam の USB 接続が使える）" } else { "無い → Apple 公式サイト版 iTunes を入れる（Microsoft Store 版は不可）" })

$reg = Get-ChildItem (Join-Path $repo "initial_poses") -Recurse -Filter marker_registration.json -ErrorAction SilentlyContinue | Select-Object -First 1
Add-Result "ArUco 登録" ([bool]$reg) $(if ($reg) { "$($reg.FullName)（台座とボードを動かしたら登録し直す）" } else { "無い → Unity パネルの Register marker を 3 回" })

$metaCount = (Get-ChildItem (Join-Path $repo "models") -Filter *.meta -ErrorAction SilentlyContinue | Measure-Object).Count
Add-Result "region model キャッシュ" ($metaCount -ge 10) "$metaCount / 10 個（足りなければ初回起動で 5 分ほど生成。止めないこと）"

Write-Host ""
Write-Host "=== まとめ ===" -ForegroundColor Cyan
$results | Format-Table -AutoSize | Out-String | Write-Host
$ng = @($results | Where-Object { $_.OK -eq "NG" })
if ($ng.Count -eq 0) {
    Write-Host "すべて OK。Unity Hub を一度終了してから開き直し、AnchorSwitchingDemoScene で Play → 1. Launch。" -ForegroundColor Green
} else {
    Write-Host "NG が $($ng.Count) 件。上の指示に従って直し、もう一度このスクリプトを実行する。" -ForegroundColor Yellow
}
