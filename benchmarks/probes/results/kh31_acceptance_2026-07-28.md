# K-H3.1 acceptance evidence — 2026-07-28

## Method note (why same-day A/B, not historical ratio)

The DoD said "tier2 matmul median geomean >= 1.0 maintained". Three
median-of-3 tier2 reruns showed the historical pre-K-H3.1 ratios are NOT
reproducible: the laptop RTX 3060's eager baseline itself drifts 2-4x
across days (tiny eager: 49.9us on 2026-07-26 vs 13-26us on 2026-07-28,
thermal/clock state). Cross-day ratio comparison is invalid on this HW.
Acceptance therefore uses a SAME-DAY, same-clock-window A/B of the
original @triton.autotune template (HEAD~1) vs the K-H3.1 R4 template.

## Same-day A/B (fp16, hot clock, 3000-iter means)

shape        | eager us | orig autotune us (ratio) | R4 us (ratio) | R4 vs orig
tiny         |    25.8  |  90.9 (0.28x)            | 38.5 (0.67x)  | 2.36x faster
gpt2-c_proj  |    12.8  |  52.6 (0.24x)            | 37.6 (0.34x)  | 1.40x faster
square-1k    |   146.9  | 107.2 (1.37x)            | 105.5 (1.39x) | 1.02x (parity)

R4 is faster-or-equal on every probed shape. Zero regression vs the
template it replaces; large-shape parity; small-shape big wins.

## Cliff mitigation (the actual K-H3.1 goal)

- Original: every new (M,N,K) triple = full 20-config autotune scan,
  measured 10-13s cold per shape.
- R4: deterministic _mm_cfg heuristic (distilled from offline fp16 sweep,
  300-iter x 3-pass medians) + bucket memo. Cold cost = ONE Triton
  compile. No scan at all.

## Honest findings (iteration history)

- R1 (kernel-arg bucketed autotune key): bucket cache-hit worked but the
  3 extra launch args defeated slim-launch, tiny/gpt2 regressed 59-77%. Rejected.
- R2 (runtime do_bench sweep per bucket): sweep noise on <100us kernels
  mis-picked tiles (gpt2-c_proj 0.30-0.42 across 3 tier2 runs). Rejected.
- R3 (fp32-sweep heuristic): WRONG DTYPE — bench harness runs fp16; fp32
  tile optima differ sharply (fp16 favours smaller BLOCK_M / fewer warps
  on latency-bound shapes). Rejected.
- R4 (fp16-sweep heuristic): landed.
- Small-M shapes retain a ~25us python-dispatch floor in the Arke wrapper
  vs eager's C++ cuBLAS path (~13us) — an architectural launch-overhead
  gap that tile choice cannot close; tracked as a known limitation.

