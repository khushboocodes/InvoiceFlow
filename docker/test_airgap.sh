#!/bin/bash
# InvoiceFlow Air-Gapped Verification Script (Linux Bash - Task 11.5)
set -e

echo "=========================================="
echo "BUILDING AIR-GAPPED DOCKER IMAGE"
echo "=========================================="
docker build -t invoiceflow:latest -f docker/Dockerfile .

echo ""
echo "=========================================="
echo "RUNNING AIR-GAPPED CONTAINER TEST (--network none)"
echo "=========================================="

mkdir -p docker_output

# Run with zero network connectivity
docker run --rm \
    --gpus all \
    --network none \
    -v "$(pwd)/datasets/raw:/data/input:ro" \
    -v "$(pwd)/docker_output:/data/output:rw" \
    invoiceflow:latest \
    --input "/data/input/172427893_3_pg11.png" \
    --output "/data/output/airgap_test_output.json" \
    --verbose

echo ""
echo "AIR-GAPPED ZERO-NETWORK TEST PASSED (Exit 0)"
