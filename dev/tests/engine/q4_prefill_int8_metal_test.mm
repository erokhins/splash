// The int8-activation affine prefill path (kernels/prefill/linear_q4_int8.metal)
// against a CPU model of its arithmetic: the activation producers (standalone
// and fused with the RMS norm) and the three projection epilogues. The
// kernels are not bit-exact with the bf16 tiles, so each output is checked
// against a double-precision evaluation of the int8 model within the bf16
// rounding of the projection, and the whole path is compared with the bf16
// kernels as a sanity bound on the quantization error.
#include "../../../runtime/metal/MetalBackend.hpp"
#include "metal/abi/Linear.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::ComputeDispatch;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

constexpr uint32_t kQuantGroup = 64;
constexpr uint32_t kTileRows = 32;
constexpr uint32_t kTileColumns = 64;
constexpr uint32_t kWeightTileN = 256;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

MetalBuffer shared(MetalBackend &backend, uint64_t bytes, const char *label) {
  return backend.allocateBuffer(std::max<uint64_t>(bytes, 16), BufferStorage::Shared, label);
}

uint16_t toBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}
float fromBf16(uint16_t value) {
  const uint32_t bits = uint32_t{value} << 16;
  float result;
  std::memcpy(&result, &bits, sizeof result);
  return result;
}
float roundBf16(float value) { return fromBf16(toBf16(value)); }
// One bf16 ulp at `value`'s magnitude.
float bf16Ulp(float value) {
  const float magnitude = std::fabs(value);
  if (magnitude < 1e-30f) return 1e-30f;
  return std::ldexp(1.0f, static_cast<int>(std::floor(std::log2(magnitude))) - 7);
}

// The input a permuted position of a 64-group holds (the kernel's order).
uint32_t permutedInput(uint32_t position) {
  const uint32_t c = position >> 4, t = (position >> 2) & 3, j = position & 3;
  return 16 * c + 8 * (t >> 1) + 2 * j + (t & 1);
}

// Q4 planes in the prefill layout: [N/256][K/64][256 columns] of 32 code
// bytes (low nibble first), a bf16 scale and a bf16 bias.
struct Shape {
  uint32_t rows, inputSize, outputSize;
};

uint64_t parameterIndex(const Shape &s, uint32_t column, uint32_t group) {
  return (uint64_t{column / kWeightTileN} * (s.inputSize / kQuantGroup) + group) * kWeightTileN +
         column % kWeightTileN;
}

struct Case {
  Shape shape;
  uint32_t storageRows;
  std::vector<float> input;         // [storageRows][K], bf16 values
  std::vector<float> normWeight;    // [K]
  std::vector<float> auxiliary;     // [storageRows][N], bf16 values (residual or gate)
  std::vector<uint8_t> codes;       // [N][K]
  std::vector<float> scale, bias;   // [N][K/64]
  MetalBuffer inputBuffer, normWeightBuffer, auxiliaryBuffer, weights, scales, biases;
  MetalBuffer quantized, rowScales, sums, output, bf16Output, bf16Sums, bf16OutputSums, normalized;
};

Case makeCase(MetalBackend &backend, const Shape &shape, std::mt19937 &random) {
  Case c;
  c.shape = shape;
  c.storageRows = (shape.rows + kTileRows - 1) / kTileRows * kTileRows;
  const uint32_t K = shape.inputSize, N = shape.outputSize, groups = K / kQuantGroup;
  const uint64_t inputElements = uint64_t{c.storageRows} * K;
  const uint64_t outputElements = uint64_t{c.storageRows} * N;
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::uniform_real_distribution<float> scaleRange(0.004f, 0.05f), biasRange(-0.4f, 0.1f),
      weightRange(0.5f, 1.5f);
  std::uniform_int_distribution<int> code(0, 15);

  c.inputBuffer = shared(backend, inputElements * 2, "int8-input");
  c.normWeightBuffer = shared(backend, uint64_t{K} * 2, "int8-norm-weight");
  c.auxiliaryBuffer = shared(backend, outputElements * 2, "int8-auxiliary");
  c.weights = shared(backend, uint64_t{K} * N / 2, "int8-weights");
  c.scales = shared(backend, uint64_t{N} * groups * 2, "int8-scales");
  c.biases = shared(backend, uint64_t{N} * groups * 2, "int8-biases");
  c.quantized = shared(backend, inputElements, "int8-quantized");
  c.rowScales = shared(backend, uint64_t{c.storageRows} * groups * 4, "int8-activation-scales");
  c.sums = shared(backend, uint64_t{c.storageRows} * groups * 4, "int8-sums");
  c.output = shared(backend, outputElements * 2, "int8-output");
  c.bf16Output = shared(backend, outputElements * 2, "bf16-output");
  c.bf16Sums = shared(backend, uint64_t{c.storageRows} * groups * 4, "bf16-sums");
  c.bf16OutputSums = shared(backend, uint64_t{c.storageRows} * (N / kQuantGroup) * 4, "bf16-output-sums");
  c.normalized = shared(backend, inputElements * 2, "bf16-normalized");

  auto *input = static_cast<uint16_t *>(c.inputBuffer.contents());
  c.input.resize(inputElements);
  for (uint64_t i = 0; i < inputElements; ++i) {
    float value = normal(random);
    if (i % 613 == 0) value *= 9.0f; // outlier channels
    input[i] = toBf16(value);
    c.input[i] = fromBf16(input[i]);
  }
  auto *normWeight = static_cast<uint16_t *>(c.normWeightBuffer.contents());
  c.normWeight.resize(K);
  for (uint32_t k = 0; k < K; ++k) {
    normWeight[k] = toBf16(weightRange(random));
    c.normWeight[k] = fromBf16(normWeight[k]);
  }
  auto *auxiliary = static_cast<uint16_t *>(c.auxiliaryBuffer.contents());
  c.auxiliary.resize(outputElements);
  for (uint64_t i = 0; i < outputElements; ++i) {
    auxiliary[i] = toBf16(normal(random) * 2.0f);
    c.auxiliary[i] = fromBf16(auxiliary[i]);
  }
  auto *weights = static_cast<uint8_t *>(c.weights.contents());
  std::memset(weights, 0, uint64_t{K} * N / 2);
  auto *scales = static_cast<uint16_t *>(c.scales.contents());
  auto *biases = static_cast<uint16_t *>(c.biases.contents());
  c.codes.resize(uint64_t{N} * K);
  c.scale.resize(uint64_t{N} * groups);
  c.bias.resize(uint64_t{N} * groups);
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t g = 0; g < groups; ++g) {
      const uint64_t p = parameterIndex(shape, n, g);
      const float s = roundBf16(scaleRange(random));
      const float b = roundBf16(biasRange(random) * s * 16.0f);
      scales[p] = toBf16(s);
      biases[p] = toBf16(b);
      c.scale[uint64_t{n} * groups + g] = s;
      c.bias[uint64_t{n} * groups + g] = b;
      for (uint32_t k = g * kQuantGroup; k < (g + 1) * kQuantGroup; ++k) {
        const uint8_t q = static_cast<uint8_t>(code(random));
        c.codes[uint64_t{n} * K + k] = q;
        const uint64_t nibble = p * kQuantGroup + (k % kQuantGroup);
        weights[nibble / 2] |= static_cast<uint8_t>(q << ((nibble % 2) * 4));
      }
    }
  return c;
}

ComputeDispatch quantizeDispatch(const Case &c, uint32_t *width) {
  ComputeDispatch d;
  d.pipelineName = "prefill_quantize_int8";
  d.buffers = {{0, c.inputBuffer}, {1, c.quantized}, {2, c.rowScales}, {3, c.sums}};
  d.bytes = {{4, width, sizeof *width}};
  d.threadgroups = {c.shape.rows, 1, 1};
  d.threadsPerThreadgroup = {256, 1, 1};
  return d;
}
ComputeDispatch normDispatch(const Case &c, uint32_t *width) {
  ComputeDispatch d;
  d.pipelineName = "prefill_norm_rms_int8";
  d.buffers = {{0, c.inputBuffer}, {1, c.normWeightBuffer}, {2, c.quantized}, {3, c.rowScales}, {4, c.sums}};
  d.bytes = {{5, width, sizeof *width}};
  d.threadgroups = {c.shape.rows, 1, 1};
  d.threadsPerThreadgroup = {256, 1, 1};
  return d;
}
ComputeDispatch projectionDispatch(const Case &c, const char *kernel, bool auxiliary, Q4PrefillParams *params) {
  ComputeDispatch d;
  d.pipelineName = kernel;
  d.buffers = {{0, c.quantized}, {1, c.weights}, {2, c.scales}, {3, c.biases}};
  uint32_t index = 4;
  if (auxiliary) d.buffers.push_back({index++, c.auxiliaryBuffer});
  d.buffers.push_back({index++, c.output});
  d.buffers.push_back({index++, c.sums});
  d.buffers.push_back({index++, c.rowScales});
  d.bytes = {{index, params, sizeof *params}};
  d.threadgroups = {c.storageRows / kTileRows, c.shape.outputSize / kTileColumns, 1};
  d.threadsPerThreadgroup = {64, 1, 1};
  return d;
}

// The CPU model of the activation producers: the row's bf16 values `value`,
// per 64-group its scale and bf16 sum, and the permuted int8 codes.
struct RowModel {
  std::vector<float> scales;
  std::vector<int> codes;
  std::vector<double> sums;
  std::vector<double> absoluteSums; // sum |v| per group, for the sums' tolerance
};
RowModel modelRow(const std::vector<float> &values) {
  RowModel m;
  const uint32_t K = static_cast<uint32_t>(values.size()), groups = K / kQuantGroup;
  m.scales.resize(groups);
  m.codes.resize(K);
  m.sums.resize(groups);
  m.absoluteSums.resize(groups);
  for (uint32_t g = 0; g < groups; ++g) {
    float peak = 0.0f;
    for (uint32_t k = g * kQuantGroup; k < (g + 1) * kQuantGroup; ++k) peak = std::max(peak, std::fabs(values[k]));
    m.scales[g] = peak > 0.0f ? peak / 127.0f : 1.0f;
    const float inverse = 1.0f / m.scales[g];
    double sum = 0.0, absolute = 0.0;
    for (uint32_t position = 0; position < kQuantGroup; ++position) {
      const float v = values[g * kQuantGroup + permutedInput(position)];
      m.codes[g * kQuantGroup + position] =
          static_cast<int>(std::nearbyint(std::clamp(v * inverse, -127.0f, 127.0f)));
      sum += v;
      absolute += std::fabs(v);
    }
    m.sums[g] = sum;
    m.absoluteSums[g] = absolute;
  }
  return m;
}

// Checks the producer outputs against `models` (one per row). The GPU's
// fast-math reciprocal (and the fused norm's fp32 inverse RMS) can move a
// value that sits on a rounding boundary by one code; such flips must be
// rare, and nothing may differ by more.
void checkProducer(const Case &c, const std::vector<RowModel> &models, const char *what) {
  const uint32_t K = c.shape.inputSize, groups = K / kQuantGroup;
  const auto *codes = static_cast<const int8_t *>(c.quantized.contents());
  const auto *scales = static_cast<const float *>(c.rowScales.contents());
  const auto *sums = static_cast<const float *>(c.sums.contents());
  uint64_t flips = 0;
  for (uint32_t row = 0; row < c.shape.rows; ++row) {
    const RowModel &m = models[row];
    for (uint32_t g = 0; g < groups; ++g) {
      const float got = scales[(uint64_t{row / kTileRows} * groups + g) * kTileRows + row % kTileRows];
      // The fused norm's fp32 inverse RMS may move the group's peak across a
      // bf16 rounding boundary (2^-7 relative at most).
      if (std::fabs(got - m.scales[g]) > (2e-6f + 1.0f / 128.0f) * m.scales[g] + 1e-30f)
        fail(std::string(what) + ": scale [" + std::to_string(row) + "][" + std::to_string(g) + "] is " +
             std::to_string(got) + ", expected " + std::to_string(m.scales[g]));
    }
    for (uint32_t k = 0; k < K; ++k) {
      const int got = codes[uint64_t{row} * K + k];
      const int difference = std::abs(got - m.codes[k]);
      if (difference > 1)
        fail(std::string(what) + ": code [" + std::to_string(row) + "][" + std::to_string(k) + "] is " +
             std::to_string(got) + ", expected " + std::to_string(m.codes[k]));
      flips += difference;
    }
    for (uint32_t g = 0; g < groups; ++g) {
      const float got = sums[(uint64_t{row / kTileRows} * groups + g) * kTileRows + row % kTileRows];
      const double expected = m.sums[g];
      // The fused norm sums bf16-rounded values whose rounding its fp32
      // inverse RMS may flip by one bf16 step each (2^-8 relative).
      const double tolerance = 1e-5 * std::fabs(expected) + m.absoluteSums[g] / 256.0 + 1e-4;
      if (std::fabs(got - expected) > tolerance)
        fail(std::string(what) + ": sum [" + std::to_string(row) + "][" + std::to_string(g) + "] is " +
             std::to_string(got) + ", expected " + std::to_string(expected));
    }
  }
  const uint64_t total = uint64_t{c.shape.rows} * K;
  if (flips > total / 500 + 2)
    fail(std::string(what) + ": " + std::to_string(flips) + " of " + std::to_string(total) +
         " codes differ from the model by one step");
}

enum class Epilogue { None, Residual, UpSilu };

// The projection of the producer's outputs as the kernel computes it, in
// double: per group dot * scale * activationScale + sum * bias, then the
// epilogue and one bf16 rounding.
void checkProjection(const Case &c, Epilogue epilogue, const char *what) {
  const uint32_t K = c.shape.inputSize, N = c.shape.outputSize, groups = K / kQuantGroup;
  const auto *codes = static_cast<const int8_t *>(c.quantized.contents());
  const auto *activationScales = static_cast<const float *>(c.rowScales.contents());
  const auto *sums = static_cast<const float *>(c.sums.contents());
  const auto *output = static_cast<const uint16_t *>(c.output.contents());
  const auto *bf16 = static_cast<const uint16_t *>(c.bf16Output.contents());
  double squares = 0.0, differences = 0.0;
  uint64_t offByOne = 0, elements = 0;
  for (uint32_t row = 0; row < c.shape.rows; ++row)
    for (uint32_t n = 0; n < N; ++n) {
      double acc = 0.0;
      for (uint32_t g = 0; g < groups; ++g) {
        int64_t dot = 0;
        for (uint32_t position = 0; position < kQuantGroup; ++position)
          dot += int64_t{codes[uint64_t{row} * K + g * kQuantGroup + position]} *
                 c.codes[uint64_t{n} * K + g * kQuantGroup + permutedInput(position)];
        const uint64_t slot = (uint64_t{row / kTileRows} * groups + g) * kTileRows + row % kTileRows;
        acc += double(dot) * c.scale[uint64_t{n} * groups + g] * activationScales[slot] +
               double(sums[slot]) * c.bias[uint64_t{n} * groups + g];
      }
      const uint64_t index = uint64_t{row} * N + n;
      // The kernel rounds once, after the epilogue. fp32 accumulation order
      // may still move the result across a bf16 rounding boundary, and the
      // fast exp of the gate adds a little; allow one step at the magnitude
      // of the projection (before a residual cancels it) and of the result.
      const float projection = static_cast<float>(acc);
      float value = projection;
      float slack = bf16Ulp(projection);
      if (epilogue == Epilogue::UpSilu) {
        const float gate = c.auxiliary[index];
        const float silu = gate / (1.0f + std::exp(-gate));
        value = silu * value;
        slack *= std::fabs(silu);
      } else if (epilogue == Epilogue::Residual) {
        value += c.auxiliary[index];
      }
      const float expected = roundBf16(value);
      const float got = fromBf16(output[index]);
      const float tolerance = 1.01f * (slack + bf16Ulp(expected)) + 1e-3f;
      const float error = std::fabs(got - expected);
      if (error > tolerance)
        fail(std::string(what) + ": output [" + std::to_string(row) + "][" + std::to_string(n) + "] is " +
             std::to_string(got) + ", expected " + std::to_string(expected) + " (bf16 kernel " +
             std::to_string(fromBf16(bf16[index])) + ")");
      if (error > 0.0f) ++offByOne;
      ++elements;
      const double reference = fromBf16(bf16[index]);
      squares += reference * reference;
      differences += (got - reference) * (got - reference);
    }
  if (offByOne > elements / 50)
    fail(std::string(what) + ": " + std::to_string(offByOne) + " of " + std::to_string(elements) +
         " outputs differ from the model by a rounding step");
  const double relativeRms = std::sqrt(differences / std::max(squares, 1e-30));
  // The int8 quantization error of a projection on N(0,1) rows with 9x
  // outliers is a few percent (up to ~6% at K=17408); anything larger means
  // a wrong operand.
  if (relativeRms > 0.1)
    fail(std::string(what) + ": relative RMS against the bf16 kernel is " + std::to_string(relativeRms));
  std::cout << "  " << what << ": rows=" << c.shape.rows << " K=" << K << " N=" << N
            << " rounding steps=" << offByOne << "/" << elements
            << " rel_rms_vs_bf16=" << relativeRms << '\n';
}

// The bf16 kernels' output of the same projection, for the sanity bound.
void runBf16(MetalBackend &backend, Case &c, Epilogue epilogue, bool normalized) {
  Q4PrefillParams params{c.shape.outputSize, c.shape.inputSize};
  const MetalBuffer input = normalized ? c.normalized : c.inputBuffer;
  ComputeDispatch sums;
  sums.pipelineName = "prefill_linear_q4_sums32";
  sums.buffers = {{0, input}, {1, c.bf16Sums}};
  sums.bytes = {{2, &params, sizeof params}};
  sums.threadgroups = {c.storageRows / kTileRows, 1, 1};
  sums.threadsPerThreadgroup = {256, 1, 1};
  static_cast<void>(backend.submit(sums));
  ComputeDispatch projection;
  projection.threadgroups = {c.storageRows / kTileRows, c.shape.outputSize / 128, 1};
  projection.threadsPerThreadgroup = {128, 1, 1};
  if (epilogue == Epilogue::None) {
    projection.pipelineName = "prefill_linear_q4_n128_sg4";
    projection.buffers = {{0, input}, {1, c.weights}, {2, c.scales}, {3, c.biases}, {4, c.bf16Output},
                          {5, c.bf16Sums}};
    projection.bytes = {{6, &params, sizeof params}};
  } else {
    projection.pipelineName = epilogue == Epilogue::Residual ? "prefill_linear_q4_n128_residual_sg4"
                                                              : "prefill_linear_q4_n128_up_silu_sums_sg4";
    projection.buffers = {{0, input}, {1, c.weights}, {2, c.scales}, {3, c.biases},
                          {4, c.auxiliaryBuffer}, {5, c.bf16Output}, {6, c.bf16Sums}};
    if (epilogue == Epilogue::UpSilu) {
      // Its output sums are not compared.
      projection.buffers.push_back({7, c.bf16OutputSums});
      projection.bytes = {{8, &params, sizeof params}};
    } else {
      projection.bytes = {{7, &params, sizeof params}};
    }
  }
  static_cast<void>(backend.submit(projection));
}

void runShape(MetalBackend &backend, const Shape &shape, std::mt19937 &random) {
  Case c = makeCase(backend, shape, random);
  const uint32_t K = shape.inputSize;
  uint32_t width = K;
  std::cout << "shape rows=" << shape.rows << " K=" << K << " N=" << shape.outputSize << '\n';

  // Standalone quantization of the bf16 rows, then the three epilogues.
  static_cast<void>(backend.submit(quantizeDispatch(c, &width)));
  std::vector<RowModel> models(shape.rows);
  for (uint32_t row = 0; row < shape.rows; ++row)
    models[row] = modelRow({c.input.begin() + uint64_t{row} * K, c.input.begin() + uint64_t{row + 1} * K});
  checkProducer(c, models, "quantize");
  Q4PrefillParams params{shape.outputSize, shape.inputSize};
  struct Variant {
    const char *kernel;
    Epilogue epilogue;
  };
  for (const Variant v : {Variant{"prefill_linear_q4_int8_m32n64", Epilogue::None},
                          Variant{"prefill_linear_q4_int8_m32n64_residual", Epilogue::Residual},
                          Variant{"prefill_linear_q4_int8_m32n64_up_silu", Epilogue::UpSilu}}) {
    static_cast<void>(backend.submit(projectionDispatch(c, v.kernel, v.epilogue != Epilogue::None, &params)));
    runBf16(backend, c, v.epilogue, false);
    checkProjection(c, v.epilogue, v.kernel);
  }

  // The fused norm: the model normalizes in double and rounds to bf16 as the
  // kernel does. The projection of its rows is then checked as above,
  // against the bf16 norm plus bf16 projection.
  static_cast<void>(backend.submit(normDispatch(c, &width)));
  auto *normalized = static_cast<uint16_t *>(c.normalized.contents());
  for (uint32_t row = 0; row < shape.rows; ++row) {
    double squares = 0.0;
    for (uint32_t k = 0; k < K; ++k) squares += double(c.input[uint64_t{row} * K + k]) * c.input[uint64_t{row} * K + k];
    const float inverse = static_cast<float>(1.0 / std::sqrt(squares / K + 1e-6));
    std::vector<float> values(K);
    for (uint32_t k = 0; k < K; ++k) {
      values[k] = roundBf16(c.input[uint64_t{row} * K + k] * inverse * c.normWeight[k]);
      normalized[uint64_t{row} * K + k] = toBf16(values[k]);
    }
    models[row] = modelRow(values);
  }
  checkProducer(c, models, "norm_rms_int8");
  static_cast<void>(backend.submit(projectionDispatch(c, "prefill_linear_q4_int8_m32n64", false, &params)));
  runBf16(backend, c, Epilogue::None, true);
  checkProjection(c, Epilogue::None, "norm_rms_int8 + prefill_linear_q4_int8_m32n64");
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: q4_prefill_int8_metal_test <metallib>\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      if (backend.capabilities().appleGpuFamily < 10) {
        std::cout << "SKIP: int8 prefill kernels need an Apple10 GPU\n";
        return 0;
      }
      std::mt19937 random(20260928);
      // Output widths are whole 256-column weight tiles (the affine layout).
      for (const Shape shape : {Shape{1, 256, 256}, Shape{31, 512, 512}, Shape{33, 1024, 256},
                                Shape{176, 2048, 768}, Shape{64, 6144, 512}, Shape{96, 5120, 768},
                                Shape{160, 17408, 256}, Shape{2048, 1024, 256}})
        runShape(backend, shape, random);
      std::cout << "PASS\n";
      return 0;
    } catch (const std::exception &error) {
      fail(error.what());
    }
  }
}
