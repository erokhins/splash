# INT8 NAX prefill — context for future work

**Date:** 2026-09-28
**Status:** investigation only, nothing implemented in Splash yet.
**Question:** mlx-vlm speeds up prefill with W8A8 int8 GEMMs on the M5 neural
accelerators (NAX). Does Splash do this, and how would it fit?

**Short answer:** no. Splash never quantizes activations, so the 2× int8 MMA
rate on M5 is unused. Splash already has the *other* half of mlx-vlm's idea
(4-bit weights fed straight to `matmul2d`, no dequantized copy), which makes
an `int8 × int4b → int32` variant a natural next step.

---

## 1. What mlx-vlm found and shipped

Source: `../mlx-vlm/research/int8-nax/README.md` (full notes, 2026-08-03/04),
production code `../mlx-vlm/mlx_vlm/int8_prefill.py`, prototypes
`w8a8.py`, `int8_gemm_sweep.py`, `mma_rate.py` in the same directory.
Machine: M5 Max, 40-core GPU, mlx 0.32, model Qwen3.6-27B affine 4-bit gs64.

### Raw NAX MMA rates (register-resident `matmul2d` loop)

| operands → accumulator | TOPS |
|---|---|
| fp16 × fp16 → fp32 | 58.9 |
| bf16 × bf16 → fp32 | 60.4 |
| **int8 × int8 → int32** | **~120** |

int8 is exactly 2× the float rate. MLX's bf16 GEMM already runs at ~95% of the
float peak, so int8 was the only real headroom they found. There is no FP4/FP8
datapath on M5.

### Prefill-shape GEMM, 5120 → 17408 (MLP)

| M | bf16 GEMM | 4-bit qmm (dequant in tile) |
|---|---|---|
| 512 | 1.76 ms | 1.91 ms |
| 2048 | 6.03 ms (60.5 TF) | 6.69 ms |
| 8192 | 24.1 ms | 26.1 ms |

4-bit qmm is only ~8% behind bf16 at prefill; the in-tile dequant is nearly
free. Dequantizing to a bf16 copy per call (`--dequant-prefill`) *lost* 10%
on M5 (it wins +11% on M4, which has no NAX).

### Their W8A8 kernel

- `Y = (Xq_i8 @ Wq_i8ᵀ) · s_x[m] · s_w[n]`, int32 accumulate, scales applied
  in-register via `get_multidimensional_index`.
- Activations: per-row dynamic symmetric int8 (`absmax/127`), separate
  row-quant kernel (0.27 ms at M=2048). Cached on input identity so q/k/v and
  gate/up quantize a shared input once.
- Weights: per-output-channel symmetric int8, **requantized from the resident
  4-bit weights on every prefill call** by a fused kernel (0.4–0.7 ms/layer,
  ~0.2 s per model pass), freed right after the GEMM. Channel scale from a
  closed-form bound `max(|b|, |15s + b|)` over the affine groups.
- Best tile: 128×128, 8 simdgroups, K loop inside. ~91 TOPS-eq = 76% of int8
  peak, 1.55× bf16 GEMM, 1.66× 4-bit qmm at M=2048.
- Routing: calls with ≥512 rows; eligibility `N % 128 == 0`, `K % 32 == 0`,
  min dim ≥ 1024, N ≤ 32768 (excludes lm_head and tiny `in_proj_a/b`).
  Decode never takes it.

### End-to-end

| prompt | prefill tok/s | output |
|---|---|---|
| 6.4k, MLP only | 832–887 → 1000–1008 (+14–20%) | bit-identical greedy |
| 6.4k, all projections | 819 → 1061 (+29.5%) | bit-identical greedy |
| 11.8k document, all projections | 726 → 1070 (+47.3%) | diverges, semantically equivalent |

Below the ~1.3–1.4× Amdahl estimate because prefill is not all matmul: the 48
GDN (linear-attention) layers take a large share. The per-call requant costs
~5% versus keeping a resident int8 copy (+24 GB); they chose no copy.

mlx-vlm's untried levers: (1) dequant 4-bit → int8 inside the GEMM's B loader
(no requant pass), (5) use MPP's `int8 × int4b` directly. **Splash's kernel
structure is already where both of those point.**

## 2. What Splash does today

### MLX affine 4-bit targets (the 27B blend package)

`runtime/metal/kernels/prefill/linear_q4.metal`, `q4_mpp_prefill_tile`
(decode-side tiles in `common/q4_mpp_tiles.h`):

- `matmul2d_descriptor(TileM, TileN, 64, …)`: A = bf16 activations,
  B = `tensor<device uint4b_format, …, tensor_inline>` over the packed weights,
  fp32 destination. **The 4-bit codes go straight to the NAX; no dequantized
  copy.**
- K is walked one 64-wide quant group at a time. Each group's `partial` is
  folded into the fp32 accumulator with the affine epilogue:
  `acc += partial * scale[g,n] + rowsum[g,m] * bias[g,n]`.
- `rowsum` (Σx over each 64-group per row) comes from
  `prefill_linear_q4_sums32`, or is fused into the RMS norm
  (`prefill_norm_rms_sums32`, `runtime/ops/Normalization.cpp:44`). The 8-simdgroup
  variants stage 256 groups of sums in threadgroup memory.
- Weights are stored in 256-column tiles. Kernels: `n128`, `n256`, `_sg4`,
  `_residual`, `_up_silu_sums` (gate·SiLU fused into up). Selection is in
  `runtime/ops/Linear.cpp` (~line 215); per-family occupancy rules for
  Apple9/Apple10 follow it (~lines 320–380).
- Math therefore runs at the **bf16 rate** (~60 TOPS class on M5), i.e. half
  of what int8 would give.

### GGUF targets

On Apple10 (M5), `runtime/metal/kernels/shared/gguf_linear.metal` +
`common/gguf_staged.h` dequantize each weight to **half in threadgroup
memory**, then run `matmul2d` on bf16/half. Also float-rate. Described in
`DEVELOPMENT.md` (~line 555).

### Things named "int8" in Splash that are *not* this

- `--kv-format int8`: Q8 KV cache (`attention_q8*.metal`, `q8_paging.h`).
- GGUF Q8_0 and friends (`quant_formats.h`).
- `tensor<device uint8_t>` in `shared/moe.metal`: MoE expert weight operand,
  float activations.

No kernel quantizes activations; no `matmul2d` has an integer A operand.

## 3. MPP `matmul2d` operand combinations (local SDK)

From `MetalPerformancePrimitives.framework/Headers/MPPTensorOpsMatMul2d.h`
(Xcode SDK on this machine), the relevant integer rows:

| A | B | destination |
|---|---|---|
| int8_t | int8_t | int32_t |
| uint8_t | uint8_t | int32_t |
| **int8_t** | **int4b_format** | **int32_t** |
| uint8_t | uint4b_format | int32_t |
| bfloat | uint4b_format | bfloat (float-rate; current Splash path uses fp32 dst) |

**`int8 × uint4b` is not supported**: signed activations pair only with
signed `int4b`.

## 4. Proposed Splash design: W4A8 on the existing Q4 tile

Keep the current kernel shape; change operand types and the epilogue.

1. **Signed weight codes, exact.** Affine weight `w = s·q + b`, `q ∈ [0,15]`.
   Store `q' = q − 8 ∈ [−8,7]` as `int4b` and fold into the bias:
   `w = s·q' + (b + 8s)`. Done once at weight preparation
   (`runtime/model/AffinePreparation.cpp` / prepared-weight layout); no extra
   memory, no numeric change. Needs its own prepared format/version so old
   packages are not misread. Note bias changes from bf16 `b` to `b + 8s`:
   compute in fp32 and check the rounding against the current outputs, or
   keep `b` and add `8s·rowsum` in the epilogue instead (exact, one more FMA).
2. **Per-row int8 activations.** `s_x[m] = absmax(row)/127`, `xq = round(x/s_x)`.
   Fuse into the producer that already writes `rowsum`
   (`prefill_norm_rms_sums32` / `prefill_linear_q4_sums32`), so it writes
   int8 rows + `s_x` + group sums in one pass. Per-row scale keeps the
   epilogue simple; per-group activation scales (one per 64-K) are a finer
   option if accuracy needs it, and fit the per-group epilogue for free.
3. **Epilogue per quant group:**
   `acc += float(partial_i32) · s_x[m] · s[g,n] + rowsum[g,m] · b'[g,n]`.
   `rowsum` stays the bf16-activation sum, so the bias term keeps today's
   accuracy; only the `q·x` term is quantized.
4. **Routing:** prefill phase only, same shapes as today's Q4 prefill; decode
   and verify (DFlash) untouched. Start with MLP gate/up/down (largest share,
   attention/KV numerics untouched), then optionally the attention and GDN
   projections, as mlx-vlm did.
5. **Apple10 only.** M3/M4 have no NAX int8 path; mlx-vlm found int8 to be
   pure overhead on M4. Gate by GPU family like the existing Apple9/Apple10
   rules in `Linear.cpp`.

Advantages over mlx-vlm's version: no per-call requant pass (−0.2 s/chunk in
their numbers), no int8 weight copy, and it reads half the weight bytes.

## 5. Risks and open questions

- **Per-group epilogue cost.** The K loop issues one 64-deep `matmul2d` per
  quant group and does an fp32 epilogue after each. With int8 the MMA part
  halves but the epilogue (int→float convert, 2 FMAs, sum lookup) does not.
  If the epilogue already takes a large share, the 2× shrinks a lot. **This is
  the main go/no-go question.** Check whether `int8 × int4b` at K=64 even
  reaches the 2× rate (mlx-vlm measured 16×32×16 and 16×32×32 tiles only).
- **Accuracy.** int8 activations change the numerics. Splash tracks top-token
  agreement against llama.cpp (`docs/performance.md`) and bit-exactness of
  kernel changes (`dev/benchmarks/*.md`); int8 prefill must be an explicit,
  measured accuracy trade, probably behind a flag at first. Outlier channels
  after RMSNorm are the usual W8A8 failure mode; per-group activation scales
  or SmoothQuant are the fallbacks.
- **Amdahl.** GDN kernels, attention, and norms are not touched. mlx-vlm got
  +14–47%, not 2×. Splash's own prefill profile will decide the ceiling; get
  the matmul share first.
- **DFlash draft / verify.** Only prefill-phase rows should take the path;
  the prompt-end rows the draft context reads (`draft_context.*`) are
  produced by the target prefill, so int8 target prefill also shifts the
  draft's inputs slightly. Measure acceptance (see §6).
- **GGUF.** Different kernel family (half staging, many formats); a separate
  port, if ever.

## 6. Suggested plan

1. **Microbenchmark, go/no-go.** Extend `dev/benchmarks/q4_prefill_profile.mm`
   with an `int8 × int4b → int32` variant of `q4_mpp_prefill_tile` at the MLP
   shapes (5120 → 17408, 17408 → 5120; M = 512/2048/8192). Compare against
   the current `n128`/`n256` kernels. Go if ≥ ~1.4× at M ≥ 2048.
2. Fused norm + activation-quant producer; correctness against the bf16
   kernel on real layer inputs (max-abs error, cosine).
3. Prepared-weight `int4b` layout with adjusted bias; Apple10 routing in
   `Linear.cpp`; flag to enable.
4. End-to-end A/B:
   - prefill throughput: `./bench.sh` (Junie replay; reports prefill tok/s over
     new tokens and DFlash acceptance), plus a long cold prompt;
   - accuracy: top-token agreement vs the current kernels on the
     `docs/performance.md` text set.
5. Extend to attention/GDN projections only if 4 is clean.

## 7. Baseline to compare against

`./bench.sh` on this machine, `local/qwen-blend-dflash2-v03`
(Qwen3.8-3.6-27B blend + DFlash2 v0.3 package), 2026-09-28:

```
warm_up.json: cached=0/14604 (0%) | prefill_new=14604tok @919tok/s (15.9s) | not counted in means
mean over 11 requests: KV cached 93% (246400/263749) | prefill 698tok/s over 17349 new tokens (24.8s) | generation 57.3tok/s (4362 tokens, 75.5s) | accept 29% of drafted (3.04tok/round)
```

The replay is mostly warm (93% cached, small suffixes), so the cold warm-up
line (14.6k tokens @ 919 tok/s) is the better prefill signal. A dedicated
long cold prompt benchmark is needed for step 4.

---

## 8. Go/no-go microbenchmark results (2026-09-28, M5 Max 40-core)

Code: `w4a8_bench.metal` / `w4a8_bench.mm` / `run.sh` in this directory
(`./run.sh` builds with the system Metal toolchain and runs; `--sustain` runs
only the sustained-load test). Raw output: `out/w4a8_bench_results.txt`,
`out/decode_profile_8192.txt`, `out/q4_prefill_profile.txt`.

The bf16 kernel is `q4_mpp_prefill_tile` copied verbatim (TileM=32, K walked
per 64-group, 256-column weight tiles, 8 simdgroups, staged sums). The int8
kernel changes only the operand types (`int8_t` A, `int4b_format` B, `int32_t`
partial) and the epilogue.

### Correctness (M=64, K=1024, N=512, random data with outlier channels)

- int8 kernel matches an fp64 model of the same arithmetic to 0.12 abs on
  outputs up to 37 (the bf16 kernel is at 0.116 on the same scale), so the
  nibble layout and the signed-code fold are right: **store `code ^ 8` (i.e.
  `q − 8` two's complement), bias' = bias + 8·scale.** No repacking beyond an
  XOR per byte; the weight bytes stay the same size.
- int8 vs bf16 kernel: 1.6% relative RMS, max abs 0.88, on synthetic
  N(0,1) activations with 12× outliers every 977th channel. Real-layer
  accuracy remains to be measured (§9).

### GEMM time, isolated, median of 7 (ms; n256 tile unless noted)

| shape | M | bf16 | int8 | ratio | bf16 raw* | int8 raw* | raw ratio | int8 scale-only** | ratio |
|---|---|---|---|---|---|---|---|---|---|
| ffn_up 5120→17408 | 512 | 1.61 | 1.25 | 1.29 | 1.48 | 0.93 | 1.58 | 1.17 | 1.37 |
| ffn_up 5120→17408 | 2048 | 6.26 | 4.82 | 1.30 | 5.82 | 3.61 | 1.61 | 4.60 | 1.36 |
| ffn_up 5120→17408 | 8192 | 24.7 | 19.2 | 1.29 | 23.0 | 14.5 | 1.59 | 18.2 | 1.36 |
| ffn_down 17408→5120 | 2048 | 6.64 | 5.05 | 1.31 | 5.99 | 3.86 | 1.55 | 4.58 | 1.45 |
| attn_in 5120→14336 | 2048 | 5.10 | 3.88 | 1.32 | 4.71 | 2.94 | 1.60 | 3.65 | 1.40 |
| gdn_in 5120→16640 | 2048 | 5.97 | 4.59 | 1.30 | 5.69 | 3.50 | 1.63 | 4.48 | 1.33 |
| mixer_out 6144→5120 | 2048 | 2.21 | 1.68 | 1.31 | 2.06 | 1.29 | 1.60 | 1.60 | 1.38 |

\* raw = the same K loop with no per-group epilogue (partials accumulated in
place): the MMA + weight-load rate at K=64 granularity.
\*\* scale-only = epilogue `acc += float(partial) · scale[g,n]` with the
`rowsum · bias` term removed (it would move to a side GEMM, see §9).

Equivalent rates at M=2048: bf16 ≈ 58–59 TFLOPS (at the float peak, as
mlx-vlm found); int8 with the production epilogue ≈ 72–78 TOPS; int8 raw ≈
95–102 TOPS (≈ 80–85% of the ~120 TOPS int8 peak). **`int8 × int4b` at K=64
does reach the 2× MMA rate; the per-group fp32 epilogue is what eats it.**

Epilogue cost at ffn_up M=2048: bf16 0.45 ms of 6.26 (7%); int8 1.22 ms of
4.82 (25%). The int8 epilogue is dearer because the int32 partial needs a
convert and cannot be accumulated in place across groups.

Other variants: per-(row, group) activation scales (`i8_n256_gscale`, one
more threadgroup lookup and multiply per element per group) cost ≈ 4%:
1.25× instead of 1.30×. bf16 scale-only gains only 1–4%, so the bias
side-GEMM is not worth doing for the current bf16 kernels alone.

Producers at M=2048: `sums32` 0.08 ms (K=5120) / 0.33 ms (K=17408);
`quant_rows` (absmax, int8 write, scaled sums) 0.19 / 0.87 ms. Standalone,
not fused into the norm.

Sustained load (96 dispatches per command buffer, 8 rounds alternating
kernels): both kernels drift ≈ 20% slower over ~15 s (bf16 6.27 → 7.46 ms,
int8 4.86 → 5.67 ms); the ratio stays **1.29×**. This drift also explains
why production dispatches in the profile below run 15–25% slower than the
isolated numbers; the tile choice is not the cause (`q4_prefill_profile`:
`n128_sg4` 6.21 ms ≈ `n256` 6.26 ms at ffn_up M=2048 on this GPU).

### What prefill spends its time on (real model, `decode-profile`)

`build/engine-tests/decode-profile build/splash.metallib <blend package>
--prompt-tokens 8192`, four 2048-token chunks (`prefill_token_budget` 2048),
9.37 s = 874 tok/s:

| kernel group | share |
|---|---|
| `prefill_linear_q4_n128_sg4` (gate, gdn/attn input, draft context) | 39.3% |
| `prefill_linear_q4_n128_residual_sg4` (mixer out, ffn down) | 28.4% |
| `prefill_linear_q4_n128_up_silu_sums_sg4` (up · silu(gate)) | 21.6% |
| **Q4 projections total** | **89.3%** |
| attention (`q8_split` + qkv/reduce/gate/store) | 4.9% |
| GDN (scan + prepare + gate) | 5.0% |
| norms, sums, everything else | 0.9% |

Unlike mlx-vlm's stack, where GDN dominated, Splash's prefill is almost all
Q4 GEMM, so the GEMM ratio carries through nearly intact:

| GEMM speedup g | end-to-end prefill = 1 / (0.107 + 0.893 / g) |
|---|---|
| 1.29 (measured, production epilogue) | **1.25×** |
| 1.40 (scale-only epilogue + side GEMM) | 1.34× |
| 1.60 (raw rate; unreachable with per-group scales) | 1.50× |

minus activation-quant overhead, estimated 1–3% of a chunk (§9). So **~874 →
~1050–1150 tok/s cold prefill on this machine** is the realistic range.
Only 16 attention layers exist in this 27B; the 48 GDN layers' projections
are the same Q4 kernels and get the same gain.

### Verdict: GO, with a ceiling of ~1.25–1.35× end-to-end

The ≥1.4× GEMM threshold from §6 is met only with the restructured epilogue;
but §6's threshold assumed a smaller GEMM share than the 89% measured, so the
expected end-to-end gain (1.25× on the plain design) already exceeds what
mlx-vlm shipped (+29% on a GDN-heavy profile).

## 9. Revised design notes

1. **Weights.** `int4b` codes = current nibbles XOR 8; bias' = bias + 8·scale
   computed in fp32 and rounded to bf16 once at preparation (the verify step
   above used exactly this and stayed within bf16-kernel error). A prepared
   format version bump; no size change. Decode keeps reading `uint4b` with
   the original bias, so either store both bias planes (+1/64 of weight
   bytes, ~200 MB for the 27B) or have decode use the signed codes too
   (decode's `q4_mpp_tile` would need the same operand switch; measure
   separately, decode is memory-bound so it should be neutral).
2. **Epilogue.** Ship the plain per-group epilogue first (1.29×). Then try
   moving `Σ_g rowsum[m,g] · bias'[g,n]` out of the K loop into one
   `matmul2d` of `float(32×G) × bfloat(G×TileN) → float` at the end of the
   tile (supported pairing; K=G is 80–272, 1/64 of the main GEMM's flops) —
   measured upper bound 1.36–1.45×.
3. **Activation quantization.** Per-row absmax/127. Fuse into
   `prefill_norm_rms_sums32` for the gdn/attention input and ffn gate/up
   inputs (the norm already has the whole row). Two inputs need a producer:
   - mixer output input (6144 wide, written by the attention/GDN output
     kernels): a standalone `quant_rows` pass, 0.26 ms at M=2048;
   - ffn down input (17408 wide, written by `up_silu_sums` in 68 column
     tiles): the row absmax spans tiles, so either a standalone pass
     (0.87 ms per layer, ≈ 56 ms per chunk ≈ 2.4%) or per-(row, 64-group)
     activation scales computed inside `up_silu_sums` (each 256-column tile
     owns whole groups) at the measured 4% epilogue cost. Per-group scales
     are also the better fallback for outlier channels, so consider them
     for all inputs if accuracy needs it.
   Store scaled sums (`rowsum / s_x`) so the epilogue keeps the current
   instruction count, and multiply by `s_x` once at the end.
4. **Routing.** Apple10 only, prefill phase only (`LinearPhase::Prefill` in
   `Linear.cpp`), behind a flag until accuracy is qualified. Decode/verify
   untouched. Draft-context rows (`prefill_linear_q4_*` on 25600→5120) can
   take it too once the target path is qualified, since the draft's inputs
   come from the target prefill either way.
5. **Accuracy qualification** (not yet done): top-token agreement against
   the bf16 kernels on the `docs/performance.md` text set, DFlash acceptance
   rate via `./bench.sh` (baseline: 29% of drafted, 3.04 tok/round), and a
   long cold prompt for tok/s. Start with MLP projections only, add
   attention/GDN projections if clean.
