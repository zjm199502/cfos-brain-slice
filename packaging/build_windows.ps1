param(
    [Parameter(Mandatory)][string]$Runtime,
    [Parameter(Mandatory)][string]$Resources,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [Parameter(Mandatory)][string]$VCRedistDirectory,
    [string]$DeepSliceDirectory,
    [string]$OfflineCellposeDirectory,
    [switch]$CompactOffline,
    [string]$Compiler = "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
)
$ErrorActionPreference = 'Stop'
$repo = (Get-Item -LiteralPath $PSScriptRoot).Parent.FullName
$Runtime = (Resolve-Path -LiteralPath $Runtime).Path
$Resources = (Resolve-Path -LiteralPath $Resources).Path
$output = (New-Item -Type Directory -Force $OutputDirectory).FullName
$python = Join-Path $Runtime 'python.exe'
$versionText = Get-Content -LiteralPath "$repo\cfos\__init__.py" -Raw -Encoding UTF8
$version = [regex]::Match($versionText, '(?m)^__version__\s*=\s*"([\d.]+)"').Groups[1].Value
if (!$version) { throw 'Cannot read the FosFolio version.' }
if (!(Test-Path -LiteralPath $Compiler)) { throw 'Specify the Inno Setup 6 compiler with -Compiler.' }
if ($CompactOffline -and !$OfflineCellposeDirectory) { throw 'CompactOffline requires OfflineCellposeDirectory.' }

# Use a new stage so removed modules or untracked files from an older build cannot leak into a release.
$stage = Join-Path $output ("win-stage-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
$appDir = (New-Item -Type Directory -Path "$stage\app\resources").Parent.FullName
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
function Copy-Tree([string]$Source, [string]$Target, [string[]]$Options) {
    & robocopy $Source $Target /E /XJ /R:1 /W:1 /NFL /NDL /NJH /NJS /NP @Options | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copy failed ($LASTEXITCODE): $Source" }
}
foreach ($required in @('pythonw.exe','Lib\ensurepip\_bundled','Lib\site-packages\PySide6','Lib\site-packages\onnxruntime')) {
    if (!(Test-Path -LiteralPath (Join-Path $Runtime $required))) { throw "Missing runtime component: $required" }
}
# Input is a clean, relocatable FosFolio x64 runtime (or the python directory of an official installation).
# Keep ensurepip's bundled pip wheel for the on-demand native Cellpose environment.
$omitDirs = @('__pycache__', '.git', '.pytest_cache', 'stardist', 'stardist*.dist-info', 'csbdeep', 'csbdeep*.dist-info',
    'tensorflow', 'tensorflow*.dist-info', 'keras', 'keras-*.dist-info', 'tf2onnx', 'tf2onnx-*.dist-info',
    'DeepSlice', 'deepslice-*.dist-info', 'clang', 'libclang-*.dist-info')
Copy-Tree $Runtime "$stage\python" (@('/XD') + $omitDirs + @('/XF', '*.pyc', '*.pyo', '*.pdb', 'scipy-*.whl', 'direct_url.json'))
& $python -I "$PSScriptRoot\copy_windows_crt.py" $VCRedistDirectory "$stage\python"
if ($LASTEXITCODE) { throw 'Could not stage the x64 Microsoft Visual C++ runtime.' }
Copy-Tree "$repo\cfos" "$appDir\cfos" @('/XD','__pycache__','.pytest_cache','/XF','*.pyc','*.pyo')
foreach ($file in @('main.py','README.md','THIRD_PARTY.md','requirements.txt')) {
    Copy-Item -LiteralPath (Join-Path $repo $file) -Destination $appDir
}
Copy-Item -LiteralPath "$PSScriptRoot\cfos.ico","$PSScriptRoot\诊断启动（出问题时用）.bat" -Destination $appDir
Copy-Item -LiteralPath "$PSScriptRoot\更新公告\$version.md" -Destination "$appDir\更新公告.md"
foreach ($file in @('atlas\allen_ccfv3_25um.npz','atlas\structures.csv','demo\示例脑片.ome.tif')) {
    $target = Join-Path "$appDir\resources" $file
    New-Item -Type Directory -Force (Split-Path -Parent $target) | Out-Null
    Copy-Item -LiteralPath (Join-Path $Resources $file) -Destination $target
}
# Only known teaching figures are distributable; this folder can also appear in an existing local installation.
$teaching = @('coloc_example','contrast_bright','contrast_dim','contrast_good','detect_good','detect_high','detect_low',
    'heatmap_example','qc_example','reg_bad','reg_good','sections_good') | ForEach-Object { "$_.png" }
New-Item -Type Directory "$appDir\resources\teaching" | Out-Null
foreach ($file in ($teaching + '示例答案.json')) {
    Copy-Item -LiteralPath "$Resources\teaching\$file" -Destination "$appDir\resources\teaching"
}
$weights = 'deepslice_Allen_Mixed_Best','deepslice_Synthetic_data_final'
$models = Join-Path $Resources 'deepslice'
if ($DeepSliceDirectory) { $models = (Resolve-Path -LiteralPath $DeepSliceDirectory).Path }
if (!(Test-Path -LiteralPath "$models\$($weights[0]).onnx") -or !(Test-Path -LiteralPath "$models\$($weights[1]).onnx")) {
    $env:CFOS_BASE_APP = (New-Item -Type Directory "$stage\conversion\resources\deepslice").Parent.Parent.FullName
    foreach ($file in @('Allen_Mixed_Best.h5','Synthetic_data_final.hdf5','xception_top.npz')) {
        Copy-Item -LiteralPath "$models\$file" -Destination "$env:CFOS_BASE_APP\resources\deepslice"
    }
    & $python "$PSScriptRoot\make_onnx.py"
    if ($LASTEXITCODE) { throw 'DeepSlice conversion failed.' }
    $models = "$env:CFOS_BASE_APP\resources\deepslice"
}
New-Item -Type Directory "$appDir\resources\deepslice" | Out-Null
foreach ($weight in $weights) {
    foreach ($ext in @('onnx','sig')) { Copy-Item -LiteralPath "$models\$weight.$ext" -Destination "$appDir\resources\deepslice" }
}
$flags = @('/Q', "/DAppVersion=$version", "/DStage=$stage", "/O$output")
$artifactName = "FosFolio-$version-Windows-x64-Setup"
if ($OfflineCellposeDirectory) {
    $bundle = (Resolve-Path -LiteralPath $OfflineCellposeDirectory).Path
    foreach ($item in @('manifest.json','cpsam_v2','wheels')) {
        if (!(Test-Path -LiteralPath (Join-Path $bundle $item))) { throw "Missing offline component: $item" }
    }
    $manifest = Get-Content -LiteralPath "$bundle\manifest.json" -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.format -ne 1 -or $manifest.platform -ne 'win-amd64') { throw 'Expected a Windows x64 Cellpose bundle.' }
    $flags += "/DCellposeBundle=$bundle"
    $artifactName = "FosFolio-$version-Windows-x64-Offline-Setup"
    if ($CompactOffline) {
        $transport = (New-Item -Type Directory (Join-Path $output ("win-compact-" + [Guid]::NewGuid().ToString('N').Substring(0, 8)))).FullName
        $wheelParts = Join-Path $transport 'wheel-parts'
        $modelPayload = Join-Path $transport 'cpsam_v2.fsp'
        & "$stage\python\python.exe" -I "$PSScriptRoot\windows_compact_wheel.py" pack $bundle $wheelParts
        if ($LASTEXITCODE) { throw 'Preparing the compact torch wheel failed.' }
        & "$stage\python\python.exe" -I "$PSScriptRoot\offline_model_transport.py" pack "$bundle\cpsam_v2" $modelPayload
        if ($LASTEXITCODE) { throw 'Preparing the compact Cellpose model failed.' }
        $wheelName = ($manifest.wheels | Where-Object { $_.file -like 'torch-*' }).file
        $restoredBytes = (Get-Item -LiteralPath "$bundle\wheels\$wheelName").Length + (Get-Item -LiteralPath "$bundle\cpsam_v2").Length
        $flags += @("/DCompactWheelParts=$wheelParts", "/DCompactModel=$modelPayload",
                    "/DCompactWheelName=$wheelName", "/DCompactRestoredBytes=$restoredBytes")
        $artifactName = "FosFolio-$version-Windows-x64-Offline-Compact-Setup"
    }
}
& $Compiler @flags "$PSScriptRoot\windows.iss"
if ($LASTEXITCODE) { throw "Inno Setup exited with $LASTEXITCODE." }
Get-Item -LiteralPath "$output\$artifactName.exe"
if ($OfflineCellposeDirectory) {
    Get-ChildItem -LiteralPath $output -File -Filter "$artifactName-*.bin"
    & $python -I "$PSScriptRoot\windows_offline_media.py" "$output\$artifactName.exe" $stage $bundle
    if ($LASTEXITCODE) { throw 'Offline installation media verification failed.' }
}
