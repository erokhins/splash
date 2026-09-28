#include "Normalization.hpp"

#include "metal/abi/ExecutionGeometry.h"

#include <utility>
#include <stdexcept>

namespace splash::ops {

std::string normKernel(std::string_view name, const NormWeights &weights, uint32_t width) {
  if (!weights.buffer || weights.buffer.sizeBytes() < weights.bytes(width))
    throw std::invalid_argument("norm weights are below the width");
  return std::string(name) + (weights.float32 ? "_f32" : "");
}

PreparedInput Normalization::addRms(metal::CommandGraph &graph,
                                    metal::MetalBuffer input,
                                    const NormWeights &weight,
                                    metal::MetalBuffer output, uint32_t width,
                                    uint32_t rows, LinearScratch scratch,
                                    LinearInput layout) {
  if (layout != LinearInput::Plain && scratch.input && rows && rows % 8 == 0) {
    if (scratch.input.sizeBytes() < tableBytes(width, rows) ||
        scratch.sums.sizeBytes() < tableSumsBytes(layout, width, rows) || width % 64)
      throw std::invalid_argument("Q4 normalization scratch is below requirement");
    graph.add(normKernel(layout == LinearInput::Table16 ? "norm_rms_table16_decode" : "norm_rms_table64_decode",
                         weight, width),
              {input, weight.buffer, output, scratch.input, scratch.sums}, width, {rows, 1, 1});
    return {std::move(output), layout};
  }
  if (rows <= SPLASH_STAGED_NORM_ROWS && width <= SPLASH_STAGED_NORM_WIDTH && width % 4 == 0)
    graph.add(normKernel("norm_rms_staged", weight, width), {std::move(input), weight.buffer, output},
              width, {rows, 1, 1}, {SPLASH_STAGED_NORM_THREADS, 1, 1});
  else
    graph.add(normKernel("norm_rms", weight, width), {std::move(input), weight.buffer, output},
              width, {rows, 1, 1});
  return {};
}

void Normalization::addRmsWithQ4Sums(
    metal::CommandGraph &graph, metal::MetalBuffer input,
    const NormWeights &weight, metal::MetalBuffer output,
    metal::MetalBuffer sums, uint32_t width, uint32_t rows) {
  graph.add(normKernel("prefill_norm_rms_sums32", weight, width),
            {std::move(input), weight.buffer, std::move(output),
             std::move(sums)},
            width, {rows, 1, 1});
}

void Normalization::addRmsWithInt8(metal::CommandGraph &graph, metal::MetalBuffer input,
                                   const NormWeights &weight, Int8Activations activations,
                                   uint32_t width, uint32_t rows) {
  // Only affine (bf16-norm) targets take int8 projections; F32 norms come
  // from GGUF targets, which run their own kernels.
  if (weight.float32) throw std::invalid_argument("int8 normalization takes bf16 norm weights");
  if (!rows || !width || width % 64 || !activations.quantized || !activations.scales || !activations.sums ||
      activations.quantized.sizeBytes() < int8ActivationBytes(width, rows) ||
      activations.scales.sizeBytes() < uint64_t{(rows + 31) / 32 * 32} * (width / 64) * 4 ||
      activations.sums.sizeBytes() < uint64_t{(rows + 31) / 32 * 32} * (width / 64) * 4)
    throw std::invalid_argument("int8 normalization buffers are below requirement");
  graph.add(normKernel("prefill_norm_rms_int8", weight, width),
            {std::move(input), weight.buffer, std::move(activations.quantized), std::move(activations.scales),
             std::move(activations.sums)},
            width, {rows, 1, 1});
}

} // namespace splash::ops
