// W4A8 prefill go/no-go microbenchmark.
//
// Mirrors runtime/metal/kernels/prefill/linear_q4.metal's q4_mpp_prefill_tile
// (TileM=32, K walked one 64-wide quant group at a time, weights in 256-column
// tiles, 8 simdgroups, staged input sums) and adds an int8-activation variant
// that feeds int8 x int4b -> int32 to the NAX. Not production code.

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

struct Params {
  uint output_size;
  uint input_size;
};

constant constexpr ushort TileM = 32;
constant constexpr ushort WeightTileN = 256;
constant constexpr ushort SumBatch = 256;

// Production bf16 sums: sums[group * TileM + row] per row tile.
kernel void sums32(device const bfloat *input [[buffer(0)]],
                   device float *sums [[buffer(1)]],
                   constant Params &params [[buffer(2)]],
                   uint tile [[threadgroup_position_in_grid]],
                   uint simd_lane [[thread_index_in_simdgroup]],
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {
  uint quant_groups = params.input_size / 64;
  input += ulong(tile) * TileM * params.input_size;
  sums += ulong(tile) * TileM * quant_groups;
  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    for (uint row = simd_group; row < TileM; row += 8) {
      uint origin = row * params.input_size + quant_group * 64 + simd_lane;
      float sum = simd_sum(float(input[origin]) + float(input[origin + 32]));
      if (simd_lane == 0)
        sums[quant_group * TileM + row] = sum;
    }
  }
}

// Activation producer for the int8 path: per-row symmetric int8
// (absmax / 127), plus the per-group bf16 sums pre-divided by the row scale
// so the tile epilogue keeps the same instruction count as the bf16 one:
//   acc += float(partial_i32) * scale[g,n] + (sum[g,m] / s_x[m]) * bias[g,n]
//   out  = acc * s_x[m]
// One threadgroup per 32-row tile, one simdgroup per row at a time.
kernel void quant_rows(device const bfloat *input [[buffer(0)]],
                       device char *quantized [[buffer(1)]],
                       device float *row_scales [[buffer(2)]],
                       device float *scaled_sums [[buffer(3)]],
                       constant Params &params [[buffer(4)]],
                       uint tile [[threadgroup_position_in_grid]],
                       uint simd_lane [[thread_index_in_simdgroup]],
                       uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint width = params.input_size;
  const uint quant_groups = width / 64;
  input += ulong(tile) * TileM * width;
  quantized += ulong(tile) * TileM * width;
  row_scales += ulong(tile) * TileM;
  scaled_sums += ulong(tile) * TileM * quant_groups;
  for (uint row = simd_group; row < TileM; row += 8) {
    device const bfloat *x = input + ulong(row) * width;
    float peak = 0.0f;
    for (uint k = simd_lane; k < width; k += 32)
      peak = max(peak, fabs(float(x[k])));
    peak = simd_max(peak);
    const float scale = peak > 0.0f ? peak / 127.0f : 1.0f;
    const float inverse = 1.0f / scale;
    device char *q = quantized + ulong(row) * width;
    for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
      uint k0 = quant_group * 64 + simd_lane;
      float a = float(x[k0]), b = float(x[k0 + 32]);
      q[k0] = char(rint(clamp(a * inverse, -127.0f, 127.0f)));
      q[k0 + 32] = char(rint(clamp(b * inverse, -127.0f, 127.0f)));
      float sum = simd_sum(a + b);
      if (simd_lane == 0)
        scaled_sums[quant_group * TileM + row] = sum * inverse;
    }
    if (simd_lane == 0)
      row_scales[row] = scale;
  }
}

// Epilogue modes.
enum class Mode : ushort {
  Affine,     // production per-group scale/bias epilogue
  RawOnly,    // no epilogue: accumulate partials directly (MMA + load rate)
  ScaleOnly,  // acc += partial * scale[g,n]; bias term moved to a side GEMM
  GroupScale, // per-(row, group) activation scales: acc += partial*scale*sx
  AffineHoist, // Affine with element coordinates resolved once, outside K
};

// AElem: bfloat or int8_t. BFmt: uint4b_format or int4b_format.
// Acc: float (bf16 A) or int32_t (int8 A).
template <class AElem, class BFmt, class Acc, ushort TileN, Mode mode>
inline void prefill_tile(device AElem *input, device uchar *weights,
                         device bfloat *scales, device bfloat *biases,
                         device bfloat *output, uint output_size,
                         uint input_size, device const float *precomputed_sums,
                         device const float *row_scales,
                         device const ushort *colsums, uint output_origin,
                         uint simd_lane, uint simd_group,
                         threadgroup float *input_sums) {
  constexpr bool Int8 = is_same_v<AElem, int8_t> || is_same_v<AElem, uint8_t>;
  constexpr bool Unsigned = is_same_v<AElem, uint8_t>;
  auto a = tensor(input, dextents<int, 2>{int(input_size), TileM},
                  array<int, 2>{1, int(input_size)});
  auto c = tensor(output, dextents<int, 2>{int(output_size), TileM},
                  array<int, 2>{1, int(output_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(TileM, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<8>> operation;
  auto a0 = a.template slice<64, TileM>(0, 0);
  uint quant_groups = input_size / 64;
  uint tile = output_origin / WeightTileN;
  uint tile_column = output_origin % WeightTileN;
  device uchar *tile_weights =
      weights + (ulong(tile) * quant_groups * WeightTileN + tile_column) * 64 / 2;
  tensor<device BFmt, dextents<int, 2>, tensor_inline> first_b(
      tile_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
  auto b0 = first_b.template slice<64, TileN>(0, 0);
  // fp32 running accumulator for Affine; raw partial type for RawOnly.
  using Running = conditional_t<mode == Mode::RawOnly, Acc, float>;
  auto accumulated = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), Running>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i)
    accumulated[i] = Running(0);

  auto load_sums = [&](uint start) {
    uint count = min(uint(SumBatch), quant_groups - start);
    uint thread_index = simd_group * 32 + simd_lane;
    for (uint index = thread_index; index < count * TileM; index += 256) {
      uint quant_group = start + index / TileM;
      uint row = index % TileM;
      input_sums[index] = precomputed_sums[quant_group * TileM + row];
    }
  };
  // GroupScale stages per-(row, group) activation scales beside the sums;
  // halve the batch so both fit in 32 KiB. (Timing only: it reads the same
  // precomputed buffer for both arrays.)
  constexpr ushort Batch = mode == Mode::GroupScale ? SumBatch / 2 : SumBatch;
  threadgroup float *input_scales = input_sums + TileM * Batch;
  auto load_scales = [&](uint start) {
    uint count = min(uint(Batch), quant_groups - start);
    uint thread_index = simd_group * 32 + simd_lane;
    for (uint index = thread_index; index < count * TileM; index += 256) {
      uint quant_group = start + index / TileM;
      uint row = index % TileM;
      input_scales[index] = precomputed_sums[quant_group * TileM + row];
    }
  };
  if (mode == Mode::Affine || mode == Mode::GroupScale || mode == Mode::AffineHoist) {
    load_sums(0);
    if (mode == Mode::GroupScale) load_scales(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  constexpr ushort MaxCap = (TileM * TileN) / 256;
  ushort elem_row[MaxCap];
  ushort elem_col[MaxCap];
  if constexpr (mode == Mode::AffineHoist) {
#pragma unroll
    for (ushort i = 0; i < MaxCap; ++i) {
      auto index = accumulated.get_multidimensional_index(i);
      elem_row[i] = ushort(index[1]);
      elem_col[i] = ushort(index[0]);
    }
  }
  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    uint input_origin = quant_group * 64;
    auto a_slice = a.template slice<64, TileM>(input_origin, 0);
    device uchar *group_weights =
        tile_weights + ulong(quant_group) * WeightTileN * 64 / 2;
    tensor<device BFmt, dextents<int, 2>, tensor_inline> b(
        group_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b_slice = b.template slice<64, TileN>(0, 0);
    if constexpr (mode == Mode::RawOnly) {
      // C = A*B + C straight into the running accumulator.
      operation.run(a_slice, b_slice, accumulated);
    } else if constexpr (mode == Mode::AffineHoist) {
      auto partial = operation.template get_destination_cooperative_tensor<
          decltype(a_slice), decltype(b_slice), Acc>();
      operation.run(a_slice, b_slice, partial);
      const ulong group_base =
          (ulong(tile) * quant_groups + quant_group) * WeightTileN + tile_column;
      const uint sum_base = (quant_group % SumBatch) * TileM;
#pragma unroll
      for (ushort i = 0; i < MaxCap; ++i) {
        ulong parameter = group_base + elem_col[i];
        float sum = input_sums[sum_base + elem_row[i]];
        accumulated[i] += float(partial[i]) * float(scales[parameter]) +
                          sum * float(biases[parameter]);
      }
      if (quant_group % SumBatch == SumBatch - 1 &&
          quant_group + 1 < quant_groups) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load_sums(quant_group + 1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    } else if constexpr (mode == Mode::ScaleOnly) {
      auto partial = operation.template get_destination_cooperative_tensor<
          decltype(a_slice), decltype(b_slice), Acc>();
      operation.run(a_slice, b_slice, partial);
#pragma unroll
      for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        auto index = accumulated.get_multidimensional_index(i);
        ulong parameter =
            (ulong(tile) * quant_groups + quant_group) * WeightTileN +
            tile_column + index[0];
        accumulated[i] += float(partial[i]) * float(scales[parameter]);
      }
    } else {
      auto partial = operation.template get_destination_cooperative_tensor<
          decltype(a_slice), decltype(b_slice), Acc>();
      operation.run(a_slice, b_slice, partial);
#pragma unroll
      for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        auto index = accumulated.get_multidimensional_index(i);
        uint row = index[1];
        ulong parameter =
            (ulong(tile) * quant_groups + quant_group) * WeightTileN +
            tile_column + index[0];
        uint slot = (quant_group % Batch) * TileM + row;
        float sum = input_sums[slot];
        float p = float(partial[i]);
        // Unsigned activations xq + 128: subtract 128 * colsum(codes), exact
        // in fp32 (both integers below 2^24).
        if constexpr (Unsigned) p = fma(-128.0f, float(colsums[parameter]), p);
        float scaled = p * float(scales[parameter]);
        if constexpr (mode == Mode::GroupScale) scaled *= input_scales[slot];
        accumulated[i] += scaled + sum * float(biases[parameter]);
      }
      if (quant_group % Batch == Batch - 1 && quant_group + 1 < quant_groups) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load_sums(quant_group + 1);
        if (mode == Mode::GroupScale) load_scales(quant_group + 1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }

  auto converted = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), bfloat>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    float value = float(accumulated[i]);
    if constexpr (Int8 && mode == Mode::Affine) {
      auto index = accumulated.get_multidimensional_index(i);
      value *= row_scales[index[1]];
    }
    converted[i] = bfloat(value);
  }
  converted.store(c.template slice<TileN, TileM>(output_origin, 0));
}

#define TILE_KERNEL(NAME, AELEM, BFMT, ACC, TILEN, MODE)                        \
  kernel void NAME(device AELEM *input [[buffer(0)]],                          \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *output [[buffer(4)]],                        \
                   device const float *sums [[buffer(5)]],                     \
                   device const float *row_scales [[buffer(6)]],               \
                   device const ushort *colsums [[buffer(7)]],                 \
                   constant Params &params [[buffer(8)]],                      \
                   uint2 group [[threadgroup_position_in_grid]],               \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[TileM * SumBatch];                            \
    uint row_tile = group.x;                                                   \
    uint output_tile = group.y;                                                \
    sums += ulong(row_tile) * TileM * (params.input_size / 64);                \
    row_scales += ulong(row_tile) * TileM;                                     \
    input += ulong(row_tile) * TileM * params.input_size;                      \
    output += ulong(row_tile) * TileM * params.output_size;                    \
    prefill_tile<AELEM, BFMT, ACC, TILEN, MODE>(                               \
        input, weights, scales, biases, output, params.output_size,            \
        params.input_size, sums, row_scales, colsums, output_tile * TILEN,     \
        simd_lane,                                                             \
        simd_group, input_sums);                                               \
  }

// Production-equivalent bf16 x uint4b -> fp32.
TILE_KERNEL(bf16_n128, bfloat, uint4b_format, float, 128, Mode::Affine)
TILE_KERNEL(bf16_n256, bfloat, uint4b_format, float, 256, Mode::Affine)
// W4A8: int8 x int4b -> int32, same per-group epilogue.
TILE_KERNEL(i8_n128, int8_t, int4b_format, int32_t, 128, Mode::Affine)
TILE_KERNEL(i8_n256, int8_t, int4b_format, int32_t, 256, Mode::Affine)
// Unsigned activations (xq + 128) against the unchanged uint4b weights.
TILE_KERNEL(u8_n128, uint8_t, uint4b_format, int32_t, 128, Mode::Affine)
TILE_KERNEL(u8_n256, uint8_t, uint4b_format, int32_t, 256, Mode::Affine)
TILE_KERNEL(u8_n256_raw, uint8_t, uint4b_format, int32_t, 256, Mode::RawOnly)
// Hoisted element coordinates.
TILE_KERNEL(bf16_n256_hoist, bfloat, uint4b_format, float, 256, Mode::AffineHoist)
TILE_KERNEL(bf16_n128_hoist, bfloat, uint4b_format, float, 128, Mode::AffineHoist)
TILE_KERNEL(i8_n256_hoist, int8_t, int4b_format, int32_t, 256, Mode::AffineHoist)
TILE_KERNEL(i8_n128_hoist, int8_t, int4b_format, int32_t, 128, Mode::AffineHoist)
// Restructured epilogues.
TILE_KERNEL(bf16_n256_scale, bfloat, uint4b_format, float, 256, Mode::ScaleOnly)
TILE_KERNEL(i8_n256_scale, int8_t, int4b_format, int32_t, 256, Mode::ScaleOnly)
TILE_KERNEL(i8_n256_gscale, int8_t, int4b_format, int32_t, 256, Mode::GroupScale)
// Raw MMA + load rate at K=64 granularity, no epilogue.
TILE_KERNEL(bf16_n128_raw, bfloat, uint4b_format, float, 128, Mode::RawOnly)
TILE_KERNEL(bf16_n256_raw, bfloat, uint4b_format, float, 256, Mode::RawOnly)
TILE_KERNEL(i8_n128_raw, int8_t, int4b_format, int32_t, 128, Mode::RawOnly)
TILE_KERNEL(i8_n256_raw, int8_t, int4b_format, int32_t, 256, Mode::RawOnly)

// ---------------------------------------------------------------------------
// Register-operand W4A8: int8 x int8 -> int32 per simdgroup (16 x 32 x 16),
// weight nibbles unpacked into the right-operand cooperative tensor straight
// from the unchanged packed Q4 stream (codes 0..15 as int8), after
// jundot/omlx#3548 ("v8"). Activations are stored with the K positions of
// each 64-group permuted so that a lane's 16 codes are one contiguous 8-byte
// run of the weight stream and one contiguous 16-byte run of the activation
// row:  position 16c + 4t + j  holds  k = 16c + 8(t>>1) + 2j + (t&1).
// Fragment coordinates follow MLX steel nax.h (BaseNAXFrag::get_coord); the
// CPU check in the host verifies them.
// ---------------------------------------------------------------------------
template <int WM, int WN>
inline void i8reg_tile(device const int8_t *qa, device const uchar *weights,
                       device const bfloat *scales, device const bfloat *biases,
                       device bfloat *output, device const float *scaled_sums,
                       device const float *row_scales, uint output_size,
                       uint input_size, uint row_origin, uint output_origin,
                       uint simd_group, uint simd_lane) {
  constexpr int BM = 32 * WM;
  constexpr int BN = 32 * WN;
  const uint groups = input_size / 64;
  const int sg_m = int(simd_group) % WM;
  const int sg_n = int(simd_group) / WM;
  const uint row_base = row_origin + uint(sg_m) * 32;
  const uint col_base = output_origin + uint(sg_n) * 32;

  const ushort qid = simd_lane >> 2;
  const ushort fm = (qid & 4) | ((simd_lane >> 1) & 3);
  const ushort fn = ((qid & 2) | (simd_lane & 1)) * 4;
  const uint cx = fn >> 2;

  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_set = matmul2d_descriptor(
      16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  matmul2d<desc_set, execution_simdgroup> op_set;
  auto ct_a = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto acc0 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();
  auto acc1 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();

  float Cf[2][16];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int e = 0; e < 16; ++e) Cf[h][e] = 0.0f;

  // Weight column runs: q = 0..3 -> column col_base + fm + 8*(q&1) + 16*(q>>1).
  const uint tile = col_base / 256;
  device const uchar *wtile = weights + ulong(tile) * groups * 256 * 32;
  device const uchar *wcol[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint tc = (col_base % 256) + fm + 8 * (q & 1) + 16 * (q >> 1);
    wcol[q] = wtile + tc * 32 + cx * 8;
  }
  // Activation rows: r = 0..1, hf = 0..1 -> row row_base + fm + 8r + 16hf.
  device const int8_t *arow[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    arow[i] = qa + ulong(row_base + fm + 8 * (i & 1) + 16 * (i >> 1)) * input_size + cx * 16;

  const uint row_tile = row_base / 32; // sums are per 32-row tile
  const uint fm_row = fm;              // row within the 32-row tile: fm + 8r + 16hf

  for (uint g = 0; g < groups; ++g) {
    uint2 wg[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
      wg[q] = *reinterpret_cast<device const uint2 *>(wcol[q] + ulong(g) * 256 * 32);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint word = wg[q][t >> 1];
        const char4 quad = as_type<char4>(((t & 1) ? (word >> 4) : word) & 0x0f0f0f0fu);
        const int base = (q >> 1) * 8 + (q & 1) * 4;
        ct_b[base + 0] = quad.x; ct_b[base + 1] = quad.y;
        ct_b[base + 2] = quad.z; ct_b[base + 3] = quad.w;
      }
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const char4 quad = as_type<char4>(*reinterpret_cast<device const uint *>(
              arow[hf * 2 + r] + ulong(g) * 64 + t * 4));
          ct_a[r * 4 + 0] = quad.x; ct_a[r * 4 + 1] = quad.y;
          ct_a[r * 4 + 2] = quad.z; ct_a[r * 4 + 3] = quad.w;
        }
        if (t == 0) { if (hf == 0) op_set.run(ct_a, ct_b, acc0); else op_set.run(ct_a, ct_b, acc1); }
        else        { if (hf == 0) op.run(ct_a, ct_b, acc0);     else op.run(ct_a, ct_b, acc1); }
      }
    }
    // Affine correction for the group.
    const ulong pbase = (ulong(tile) * groups + g) * 256 + (col_base % 256);
    device const float *gsums = scaled_sums + (ulong(row_tile) * groups + g) * 32;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
      for (int e = 0; e < 16; ++e) {
        const uint n_local = (e >> 3) * 16 + fn + (e & 3);
        const uint m_local = fm_row + ((e & 7) >> 2) * 8 + hf * 16;
        const float sw = float(scales[pbase + n_local]);
        const float bw = float(biases[pbase + n_local]);
        const float s = gsums[m_local];
        const int32_t dot = hf == 0 ? acc0[e] : acc1[e];
        Cf[hf][e] += float(dot) * sw + s * bw;
      }
    }
  }
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      const uint n = col_base + (e >> 3) * 16 + fn + (e & 3);
      const uint m = row_base + fm + ((e & 7) >> 2) * 8 + hf * 16;
      output[ulong(m) * output_size + n] = bfloat(Cf[hf][e] * row_scales[m]);
    }
  }
}

#define I8REG_KERNEL(NAME, WM, WN)                                             \
  kernel void NAME(device const int8_t *input [[buffer(0)]],                   \
                   device const uchar *weights [[buffer(1)]],                  \
                   device const bfloat *scales [[buffer(2)]],                  \
                   device const bfloat *biases [[buffer(3)]],                  \
                   device bfloat *output [[buffer(4)]],                        \
                   device const float *sums [[buffer(5)]],                     \
                   device const float *row_scales [[buffer(6)]],               \
                   constant Params &params [[buffer(7)]],                      \
                   uint2 group [[threadgroup_position_in_grid]],               \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    i8reg_tile<WM, WN>(input, weights, scales, biases, output, sums,           \
                       row_scales, params.output_size, params.input_size,      \
                       group.x * (32 * WM), group.y * (32 * WN), simd_group,   \
                       simd_lane);                                             \
  }
I8REG_KERNEL(i8reg_m32n64, 1, 2)
I8REG_KERNEL(i8reg_m64n32, 2, 1)
I8REG_KERNEL(i8reg_m64n64, 2, 2)
I8REG_KERNEL(i8reg_m128n32, 4, 1)
I8REG_KERNEL(i8reg_m32n128, 1, 4)
I8REG_KERNEL(i8reg_m128n64, 4, 2)
I8REG_KERNEL(i8reg_m64n128, 2, 4)
