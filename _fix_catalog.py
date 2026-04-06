#!/usr/bin/env python3
"""Fix the broken EMBEDDING fragment in catalog.py"""

path = "/home/blueyi/workspace/repos/arke/arke/ir/ops/catalog.py"

with open(path) as f:
    lines = f.readlines()

# lines 665-671 (0-indexed 664-670) are the broken fragment
# Find first occurrence of "EMBEDDING = _register"
first_idx = None
second_idx = None
for i, line in enumerate(lines):
    if line.strip().startswith("EMBEDDING = _register"):
        if first_idx is None:
            first_idx = i
        else:
            second_idx = i
            break

print(f"First EMBEDDING at line {first_idx+1}, second at {second_idx+1}")

# Remove lines from first_idx up to (but not including) second_idx
fixed = lines[:first_idx] + lines[second_idx:]

with open(path, "w") as f:
    f.writelines(fixed)

print(f"Fixed. Total lines now: {len(fixed)}")
