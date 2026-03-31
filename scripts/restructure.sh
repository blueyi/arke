#!/usr/bin/env bash
# Arke project restructure script
set -e
cd /home/blueyi/workspace/repos/arke

echo "=== Phase 1: Create new directories ==="
mkdir -p arke/agent/tools
mkdir -p arke/agent/providers
mkdir -p arke/integration
mkdir -p arke/backend/triton_templates
mkdir -p docs/spec
mkdir -p benchmarks/baselines

echo "=== Phase 2: Rename schedule.py → strategy.py ==="
# Keep old file for git to detect rename
cp arke/ir/schedule.py arke/ir/strategy.py

echo "=== Done creating structure ==="
find arke/ -type d | sort
