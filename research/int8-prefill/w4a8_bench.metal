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
};

// AElem: bfloat or int8_t. BFmt: uint4b_format or int4b_format.
// Acc: float (bf16 A) or int32_t (int8 A).
template <class AElem, class BFmt, class Acc, ushort TileN, Mode mode>
inline void prefill_tile(device AElem *input, device uchar *weights,
                         device bfloat *scales, device bfloat *biases,
                         device bfloat *output, uint output_size,
                         uint input_size, device const float *precomputed_sums,
                         device const float *row_scales, uint output_origin,
                         uint simd_lane, uint simd_group,
                         threadgroup float *input_sums) {
  constexpr bool Int8 = is_same_v<AElem, int8_t>;
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
  if (mode == Mode::Affine || mode == Mode::GroupScale) {
    load_sums(0);
    if (mode == Mode::GroupScale) load_scales(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
        float scaled = float(partial[i]) * float(scales[parameter]);
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
                   constant Params &params [[buffer(7)]],                      \
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
        params.input_size, sums, row_scales, output_tile * TILEN, simd_lane,   \
        simd_group, input_sums);                                               \
  }

// Production-equivalent bf16 x uint4b -> fp32.
TILE_KERNEL(bf16_n128, bfloat, uint4b_format, float, 128, Mode::Affine)
TILE_KERNEL(bf16_n256, bfloat, uint4b_format, float, 256, Mode::Affine)
// W4A8: int8 x int4b -> int32, same per-group epilogue.
TILE_KERNEL(i8_n128, int8_t, int4b_format, int32_t, 128, Mode::Affine)
TILE_KERNEL(i8_n256, int8_t, int4b_format, int32_t, 256, Mode::Affine)
// Restructured epilogues.
TILE_KERNEL(bf16_n256_scale, bfloat, uint4b_format, float, 256, Mode::ScaleOnly)
TILE_KERNEL(i8_n256_scale, int8_t, int4b_format, int32_t, 256, Mode::ScaleOnly)
TILE_KERNEL(i8_n256_gscale, int8_t, int4b_format, int32_t, 256, Mode::GroupScale)
// Raw MMA + load rate at K=64 granularity, no epilogue.
TILE_KERNEL(bf16_n128_raw, bfloat, uint4b_format, float, 128, Mode::RawOnly)
TILE_KERNEL(bf16_n256_raw, bfloat, uint4b_format, float, 256, Mode::RawOnly)
TILE_KERNEL(i8_n128_raw, int8_t, int4b_format, int32_t, 128, Mode::RawOnly)
TILE_KERNEL(i8_n256_raw, int8_t, int4b_format, int32_t, 256, Mode::RawOnly)
