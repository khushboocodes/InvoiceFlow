# InvoiceFlow Air-Gapped Verification Script (Windows PowerShell - Task 11.5)
# Tests container execution with networking completely disabled (--network none)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "BUILDING AIR-GAPPED DOCKER IMAGE" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

docker build -t invoiceflow:latest -f docker/Dockerfile .

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "RUNNING AIR-GAPPED CONTAINER TEST (--network none)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$testInput = (Get-Location).Path + "\datasets\raw"
$testOutput = (Get-Location).Path + "\docker_output"

if (!(Test-Path $testOutput)) {
    New-Item -ItemType Directory -Path $testOutput | Out-Null
}

# Run with zero network connectivity
docker run --rm `
    --gpus all `
    --network none `
    -v "${testInput}:/data/input:ro" `
    -v "${testOutput}:/data/output:rw" `
    invoiceflow:latest `
    --input "/data/input/172427893_3_pg11.png" `
    --output "/data/output/airgap_test_output.json" `
    --verbose

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nAIR-GAPPED ZERO-NETWORK TEST PASSED (Exit 0)" -ForegroundColor Green
} else {
    Write-Host "`nAIR-GAPPED TEST FAILED" -ForegroundColor Red
}
