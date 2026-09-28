#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/rms_inverse.h"

// Affine Q4 prefill projections on int8 activations (Apple10): the neural
// accelerator's int8 x int8 -> int32 matmul runs at twice its bf16 rate. Each
// simdgroup multiplies a 16 x 16 int8 activation fragment against a 32 x 16
// int8 weight fragment that it unpacks from the packed Q4 stream in registers
// (codes 0..15 as int8), accumulates the four fragments of one 64-input quant
// group in int32, and folds the group into fp32 with the affine epilogue:
//   acc[m][n] += dot * scale[g][n] * s_x[g][m] + sum[g][m] * bias[g][n]
//   out[m][n]  = bf16(acc[m][n])
// where s_x[g][m] = max|x[m][g]| / 127 is the activation scale of the row's
// group (one per 64 inputs, aligned with the weight groups, so an outlier
// channel costs only its own group's resolution), the int8 rows are
// round(x / s_x), and sum[g][m] is the bf16 row's group sum, so only the x·q
// term is quantized. The weights are read as stored; decode reads the same
// bytes with its bf16 tiles.
//
// Fragment layout (16 x 16, 8 elements per lane; the 16 x 32 right operand and
// destination are two such halves): lane l holds rows fm and fm + 8 and
// columns fn .. fn + 3, with fm = (l >> 4 << 2) | ((l >> 1) & 3) and
// fn = ((l >> 3 & 1) << 2 | (l & 1)) * 4. The K positions a fragment slot
// stands for are free to choose, since the four K steps of a group are summed
// before the epilogue, so the activation rows are stored with each group
// permuted: position 16c + 4t + j holds input 16c + 8(t >> 1) + 2j + (t & 1).
// Lane column group c then reads its 16 codes of a weight column as one
// 8-byte run (words 2c, 2c + 1 of the 32-byte group) and step t takes the
// even (t & 1 == 0) or odd nibbles of word t >> 1; its activations for step t
// are the 4 contiguous bytes at 16c + 4t. (After jundot/omlx#3548.)

namespace q4_int8 {

// A threadgroup of two simdgroups computes TileRows x TileColumns outputs,
// one 32 x 32 tile per simdgroup.
constant constexpr ushort TileRows = 32;
constant constexpr ushort TileColumns = 64;
constant constexpr ushort WeightTileN = 256;  // weights are stored in 256-column tiles

// The input a permuted position of a 64-group holds.
inline uint permuted_input(uint position) {
  const uint c = position >> 4, t = (position >> 2) & 3, j = position & 3;
  return 16 * c + 8 * (t >> 1) + 2 * j + (t & 1);
}

// Quantizes one row held as `value(k)` by an 8-simdgroup threadgroup, each
// simdgroup a 64-group at a time: the int8 permuted row, and the group's
// activation scale (absmax / 127, or 1 for an all-zero group) and bf16 sum,
// both in the prefill sums layout ([row tile][quant group][32 rows]).
template <class Value>
inline void quantize_row(Value value, device char *quantized, device float *scales,
                         device float *sums, uint width, uint row, uint lane,
                         uint simd_group) {
  const uint quant_groups = width / 64;
  const uint row_tile = row / TileRows, row_in_tile = row % TileRows;
  device char *quantized_row = quantized + ulong(row) * width;
  for (uint quant_group = simd_group; quant_group < quant_groups; quant_group += 8) {
    const uint origin = quant_group * 64;
    const float first = value(origin + permuted_input(lane));
    const float second = value(origin + permuted_input(lane + 32));
    const float peak = simd_max(max(fabs(first), fabs(second)));
    const float scale = peak > 0.0f ? peak / 127.0f : 1.0f;
    const float inverse = 1.0f / scale;
    quantized_row[origin + lane] = char(rint(clamp(first * inverse, -127.0f, 127.0f)));
    quantized_row[origin + lane + 32] = char(rint(clamp(second * inverse, -127.0f, 127.0f)));
    const float group_sum = simd_sum(first + second);
    if (lane == 0) {
      const ulong slot = (ulong(row_tile) * quant_groups + quant_group) * TileRows + row_in_tile;
      scales[slot] = scale;
      sums[slot] = group_sum;
    }
  }
}

// Rows `TileRows` x `TileColumns` of output_size columns: the threadgroup's
// two simdgroups each own 32 rows x 32 columns.
template <bool AddResidual, bool MultiplySiluGate>
inline void prefill_tile(device const char *quantized, device const uchar *weights,
                         device const bfloat *scales, device const bfloat *biases,
                         device bfloat *output, device const bfloat *auxiliary,
                         device const float *sums, device const float *activation_scales,
                         uint output_size, uint input_size, uint row_origin,
                         uint output_origin, uint simd_group, uint simd_lane) {
  const uint quant_groups = input_size / 64;
  const uint row_base = row_origin;
  const uint col_base = output_origin + simd_group * 32;

  const ushort quad_id = simd_lane >> 2;
  const ushort fm = (quad_id & 4) | ((simd_lane >> 1) & 3);
  const ushort fn = ((quad_id & 2) | (simd_lane & 1)) * 4;
  const uint cx = fn >> 2;

  constexpr auto accumulate = matmul2d_descriptor(
      16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto set = matmul2d_descriptor(
      16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<accumulate, execution_simdgroup> op;
  matmul2d<set, execution_simdgroup> op_set;
  auto a = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto b = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto dot0 = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), int32_t>();
  auto dot1 = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), int32_t>();

  float accumulated[2][16];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int e = 0; e < 16; ++e) accumulated[h][e] = 0.0f;

  // Weight runs: q -> column col_base + fm + 8 (q & 1) + 16 (q >> 1).
  const uint tile = col_base / WeightTileN;
  device const uchar *tile_weights = weights + ulong(tile) * quant_groups * WeightTileN * 32;
  device const uchar *column_runs[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint tile_column = (col_base % WeightTileN) + fm + 8 * (q & 1) + 16 * (q >> 1);
    column_runs[q] = tile_weights + tile_column * 32 + cx * 8;
  }
  // Activation runs: i -> row row_base + fm + 8 (i & 1) + 16 (i >> 1).
  device const char *row_runs[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    row_runs[i] = quantized + ulong(row_base + fm + 8 * (i & 1) + 16 * (i >> 1)) * input_size + cx * 16;

  const uint row_tile = row_base / TileRows;
  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    uint2 words[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
      words[q] = *reinterpret_cast<device const uint2 *>(column_runs[q] + ulong(quant_group) * WeightTileN * 32);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint word = words[q][t >> 1];
        const char4 codes = as_type<char4>(((t & 1) ? (word >> 4) : word) & 0x0f0f0f0fu);
        const int base = (q >> 1) * 8 + (q & 1) * 4;
        b[base + 0] = codes.x; b[base + 1] = codes.y;
        b[base + 2] = codes.z; b[base + 3] = codes.w;
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const char4 values = as_type<char4>(*reinterpret_cast<device const uint *>(
              row_runs[h * 2 + r] + ulong(quant_group) * 64 + t * 4));
          a[r * 4 + 0] = values.x; a[r * 4 + 1] = values.y;
          a[r * 4 + 2] = values.z; a[r * 4 + 3] = values.w;
        }
        if (t == 0) { if (h == 0) op_set.run(a, b, dot0); else op_set.run(a, b, dot1); }
        else        { if (h == 0) op.run(a, b, dot0);     else op.run(a, b, dot1); }
      }
    }
    const ulong parameter_base = (ulong(tile) * quant_groups + quant_group) * WeightTileN + (col_base % WeightTileN);
    const ulong slot_base = (ulong(row_tile) * quant_groups + quant_group) * TileRows;
    device const float *group_sums = sums + slot_base;
    device const float *group_scales = activation_scales + slot_base;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int e = 0; e < 16; ++e) {
        const uint column = (e >> 3) * 16 + fn + (e & 3);
        const uint row = fm + ((e & 7) >> 2) * 8 + h * 16;
        const float scale = float(scales[parameter_base + column]);
        const float bias = float(biases[parameter_base + column]);
        const int32_t dot = h == 0 ? dot0[e] : dot1[e];
        accumulated[h][e] += float(dot) * (scale * group_scales[row]) + group_sums[row] * bias;
      }
    }
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      const uint column = col_base + (e >> 3) * 16 + fn + (e & 3);
      const uint row = row_base + fm + ((e & 7) >> 2) * 8 + h * 16;
      const ulong index = ulong(row) * output_size + column;
      // One bf16 rounding, after the epilogue.
      float value = accumulated[h][e];
      if constexpr (MultiplySiluGate) {
        const float gate = float(auxiliary[index]);
        value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * value;
      }
      if constexpr (AddResidual) value += float(auxiliary[index]);
      output[index] = bfloat(value);
    }
  }
}

} // namespace q4_int8

// Quantizes `width`-wide bf16 rows, one threadgroup of 256 threads per row.
kernel void prefill_quantize_int8(device const bfloat *input [[buffer(0)]],
                                  device char *quantized [[buffer(1)]],
                                  device float *scales [[buffer(2)]],
                                  device float *sums [[buffer(3)]],
                                  constant uint &width [[buffer(4)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simd_group [[simdgroup_index_in_threadgroup]]) {
  device const bfloat *values = input + ulong(row) * width;
  q4_int8::quantize_row([&](uint column) { return float(values[column]); }, quantized, scales, sums,
                        width, row, lane, simd_group);
}

// RMS norm fused with the quantization: the normalized row is rounded to bf16
// as the bf16 norm writes it, then quantized; only the int8 form is stored.
kernel void prefill_norm_rms_int8(device const bfloat *input [[buffer(0)]],
                                  device const bfloat *weight [[buffer(1)]],
                                  device char *quantized [[buffer(2)]],
                                  device float *scales [[buffer(3)]],
                                  device float *sums [[buffer(4)]],
                                  constant uint &width [[buffer(5)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint thread_index [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float reductions[8];
  device const bfloat *values = input + ulong(row) * width;
  const float inverse_rms = rms_inverse(values, width, reductions, thread_index, lane, simd_group);
  q4_int8::quantize_row(
      [&](uint column) { return float(bfloat(float(values[column]) * inverse_rms * float(weight[column]))); },
      quantized, scales, sums, width, row, lane, simd_group);
}

kernel void prefill_linear_q4_int8_m32n64(
    device const char *quantized [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const bfloat *scales [[buffer(2)]], device const bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]], device const float *sums [[buffer(5)]],
    device const float *activation_scales [[buffer(6)]],
    constant Q4PrefillParams &params [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  q4_int8::prefill_tile<false, false>(
      quantized, weights, scales, biases, output, output, sums, activation_scales, params.output_size,
      params.input_size, group.x * q4_int8::TileRows, group.y * q4_int8::TileColumns, simd_group,
      simd_lane);
}

kernel void prefill_linear_q4_int8_m32n64_residual(
    device const char *quantized [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const bfloat *scales [[buffer(2)]], device const bfloat *biases [[buffer(3)]],
    device const bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]], device const float *activation_scales [[buffer(7)]],
    constant Q4PrefillParams &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  q4_int8::prefill_tile<true, false>(
      quantized, weights, scales, biases, output, residual, sums, activation_scales, params.output_size,
      params.input_size, group.x * q4_int8::TileRows, group.y * q4_int8::TileColumns, simd_group,
      simd_lane);
}

// The up projection times silu(gate) of the same rows.
kernel void prefill_linear_q4_int8_m32n64_up_silu(
    device const char *quantized [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const bfloat *scales [[buffer(2)]], device const bfloat *biases [[buffer(3)]],
    device const bfloat *gate [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]], device const float *activation_scales [[buffer(7)]],
    constant Q4PrefillParams &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  q4_int8::prefill_tile<false, true>(
      quantized, weights, scales, biases, output, gate, sums, activation_scales, params.output_size,
      params.input_size, group.x * q4_int8::TileRows, group.y * q4_int8::TileColumns, simd_group,
      simd_lane);
}
