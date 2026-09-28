// Affine Q4 prefill projections on int8 activations (Apple10):
// kernels/prefill/linear_q4_int8.metal. The path is additive: it reads the
// same weight planes as the bf16 tiles and keeps its own activation
// producers, so nothing else in the projection policy changes.
#include "Linear.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Linear.h"

#include <stdexcept>
#include <string>

namespace splash::ops {

namespace {

// A buffer the path does not use needs no bytes and may be absent.
void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes, const char *what) {
  if (bytes && (!buffer || buffer.sizeBytes() < bytes))
    throw std::invalid_argument(std::string("projection ") + what + " buffer holds " +
                                std::to_string(buffer.sizeBytes()) + " bytes, needs " + std::to_string(bytes));
}

constexpr uint32_t kTileRows = 32;
constexpr uint32_t kTileColumns = 64;
constexpr uint32_t kThreads = 64;
constexpr uint32_t kQuantGroup = 64;
// The affine planes are stored in 256-column tiles (WeightLayout.hpp).
constexpr uint32_t kWeightTileColumns = 256;

uint64_t storageRows(uint32_t rows) noexcept {
  return uint64_t{(rows + kTileRows - 1) / kTileRows} * kTileRows;
}

// Every buffer of `rows` int8 rows of `width` inputs, in the int8 layout.
void requireActivations(const Int8Activations &a, uint32_t width, uint32_t rows) {
  if (!rows || rows > SPLASH_PREFILL_TOKEN_BUDGET || !width || width % kQuantGroup)
    throw std::invalid_argument("invalid int8 prefill activation shape");
  const uint64_t storage = storageRows(rows);
  requireBytes(a.quantized, int8ActivationBytes(width, storage), "int8 activation");
  requireBytes(a.scales, storage * (width / kQuantGroup) * sizeof(float), "activation scales");
  requireBytes(a.sums, storage * (width / kQuantGroup) * sizeof(float), "sums");
}

} // namespace

bool Linear::supportsInt8Prefill(const Projection &p) noexcept {
  static_assert(kWeightTileColumns % kTileColumns == 0);
  return p.layout() == WeightLayout::Affine64 && p.destination == FloatOutput::BFloat16 &&
      p.outputSize && p.outputSize % kWeightTileColumns == 0 && p.inputSize && p.inputSize % kQuantGroup == 0;
}

void Linear::addPrefillQuantize(metal::CommandGraph &graph, metal::MetalBuffer input,
                                Int8Activations activations, uint32_t width, uint32_t rows) const {
  requireActivations(activations, width, rows);
  requireBytes(input, uint64_t{rows} * width * 2, "input");
  graph.add("prefill_quantize_int8",
            {std::move(input), std::move(activations.quantized), std::move(activations.scales),
             std::move(activations.sums)},
            width, {rows, 1, 1});
}

void Linear::addPrefillInt8(metal::CommandGraph &graph, std::string_view kernel, Int8Activations input,
                            const Projection &p, metal::MetalBuffer auxiliary, metal::MetalBuffer output,
                            uint32_t rows) const {
  if (!prefillInt8()) throw std::logic_error("int8 prefill projections are not enabled on this device");
  if (!supportsInt8Prefill(p)) throw std::invalid_argument("projection does not take int8 prefill activations");
  const LinearMatrix matrix{p.outputSize, p.inputSize};
  requireAffineProjection(p, matrix);
  requireActivations(input, p.inputSize, rows);
  const uint64_t storage = storageRows(rows);
  requireBytes(output, storage * p.outputSize * 2, "output");
  if (auxiliary) requireBytes(auxiliary, storage * p.outputSize * 2, "auxiliary");
  const AffineWeights &weights = p.affine();
  std::vector<metal::MetalBuffer> bindings{input.quantized, weights.weights, weights.scales, weights.biases};
  if (auxiliary) bindings.push_back(std::move(auxiliary));
  bindings.insert(bindings.end(), {std::move(output), input.sums, input.scales});
  graph.add(std::string(kernel), std::move(bindings), Q4PrefillParams{p.outputSize, p.inputSize},
            {static_cast<uint32_t>(storage / kTileRows), p.outputSize / kTileColumns, 1}, {kThreads, 1, 1});
}

void Linear::addPrefillInt8(metal::CommandGraph &graph, Int8Activations input, const Projection &p,
                            metal::MetalBuffer output, uint32_t rows) const {
  addPrefillInt8(graph, "prefill_linear_q4_int8_m32n64", std::move(input), p, {}, std::move(output), rows);
}
void Linear::addPrefillInt8Residual(metal::CommandGraph &graph, Int8Activations input, const Projection &p,
                                    metal::MetalBuffer residual, metal::MetalBuffer output,
                                    uint32_t rows) const {
  if (!residual) throw std::invalid_argument("int8 residual projection needs residual rows");
  addPrefillInt8(graph, "prefill_linear_q4_int8_m32n64_residual", std::move(input), p, std::move(residual),
                 std::move(output), rows);
}
void Linear::addPrefillInt8UpWithGate(metal::CommandGraph &graph, Int8Activations input, const Projection &up,
                                      metal::MetalBuffer gateScratch, metal::MetalBuffer output,
                                      uint32_t rows) const {
  if (!gateScratch) throw std::invalid_argument("int8 up projection needs gate rows");
  addPrefillInt8(graph, "prefill_linear_q4_int8_m32n64_up_silu", std::move(input), up, std::move(gateScratch),
                 std::move(output), rows);
}

} // namespace splash::ops
