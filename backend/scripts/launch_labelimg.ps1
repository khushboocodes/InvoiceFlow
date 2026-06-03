# Launches LabelImg pre-configured for the signature/stamp annotation task.
#
# Usage (from backend/):
#   .\scripts\launch_labelimg.ps1
#
# What this does:
#   * Activates the backend venv if not already active
#   * Points LabelImg at train_data_idfc/yolo/images/
#   * Saves YOLO-format labels to train_data_idfc/yolo/labels/
#   * Pre-loads the classes.txt (signature, stamp)

$ErrorActionPreference = "Stop"

$root = Resolve-Path "$PSScriptRoot\.."
$repoRoot = Resolve-Path "$root\.."
$yoloDir = Join-Path $repoRoot "train_data_idfc\yolo"
$imagesDir = Join-Path $yoloDir "images"
$labelsDir = Join-Path $yoloDir "labels"
$classesFile = Join-Path $yoloDir "classes.txt"

if (-not (Test-Path $imagesDir)) {
    Write-Host "Image folder not found: $imagesDir" -ForegroundColor Red
    Write-Host "Run scripts/prepare_label_set.py first." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $labelsDir)) {
    New-Item -ItemType Directory -Path $labelsDir | Out-Null
}

if (-not (Test-Path $classesFile)) {
    Set-Content -Path $classesFile -Value "signature`nstamp"
}

# Use the venv's bundled labelImg.exe directly. python -m labelImg does NOT
# work because the package has no __main__ module.
$labelImgExe = Join-Path $root ".venv\Scripts\labelImg.exe"
if (-not (Test-Path $labelImgExe)) {
    Write-Host "labelImg.exe not found at $labelImgExe" -ForegroundColor Red
    Write-Host "Reinstall with: .\.venv\Scripts\python.exe -m pip install labelImg" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Launching LabelImg..." -ForegroundColor Cyan
Write-Host "  Images:  $imagesDir"
Write-Host "  Labels:  $labelsDir"
Write-Host "  Classes: $classesFile"
Write-Host ""
Write-Host "INSIDE LABELIMG:" -ForegroundColor Yellow
Write-Host "  1. View -> Auto Save mode (toggle on)"
Write-Host "  2. Click 'PascalVOC' button bottom-left to switch to 'YOLO'"
Write-Host "  3. Press W to draw a box, label as 'signature' or 'stamp'"
Write-Host "  4. Press D for next image, A for previous"
Write-Host ""

& $labelImgExe $imagesDir $classesFile $labelsDir
