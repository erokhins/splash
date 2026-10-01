#pragma once

// Plans the affine images (AffinePreparation.hpp) of a safetensors checkpoint:
// a header block, then 16 KiB-aligned sections, each bound to the checkpoint
// tensors it reads. AffineTarget.cpp plans an MLX target with it,
// DraftCheckpoint.cpp a DFlash2 draft.

#include "model/AffinePreparation.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightStore.hpp"

#include <algorithm>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::model::affine {

// An image of a header block; append places each section after it.
[[nodiscard]] inline Image image(std::string name, std::string_view magic, uint32_t layer, uint32_t type) {
  return {std::move(name), std::string(magic), layer, type, kWeightFileAlignment, {}, {}};
}

inline void append(Image &image, Section section) {
  section.offset = image.bytes;
  image.bytes = alignWeightOffset(image.bytes + section.bytes);
  image.sections.push_back(std::move(section));
}

// A tensor copied as stored, BF16 or U32.
inline void copy(Image &image, const std::string &name, std::vector<uint64_t> shape,
                 const std::string &dtype = "BF16") {
  Section section;
  section.bytes = dtype == "U32" ? 4 : kBFloat16Bytes;
  for (uint64_t dimension : shape) section.bytes = checkedWeightMultiply(section.bytes, dimension, "affine tensor");
  section.input = {name, {dtype}, std::move(shape)};
  append(image, std::move(section));
}

// The fields of rows of an MLX Q4 module: its weight codes, scales and biases.
inline ProjectionPart mlxQ4Part(Image &image, const std::string &name, uint32_t rows, uint32_t columns,
                                uint32_t bits = 4, uint32_t experts = 1) {
  image.quantized.emplace_back(name, bits);
  ProjectionPart part{rows, {}};
  for (size_t field = 0; field < 3; ++field) { // weight, scales, biases
    std::vector<uint64_t> shape{rows, field ? columns / kQ4GroupElements : columns * bits / 32};
    if (experts > 1) shape.insert(shape.begin(), experts);
    part.fields.push_back({name + (field == 0 ? ".weight" : field == 1 ? ".scales" : ".biases"),
                           {field ? "BF16" : "U32"}, std::move(shape)});
  }
  return part;
}

// MLX Q4 projection parts, stacked in row order, padded with zero rows to
// `rows`.
inline void projection(Image &image, std::initializer_list<std::pair<std::string, uint32_t>> parts,
                       uint32_t rows, uint32_t columns, uint32_t bits = 4, uint32_t experts = 1) {
  validateQ4Layout(rows, columns);
  Section section;
  section.kind = SectionKind::Projection;
  section.rows = rows;
  section.columns = columns;
  section.bits = bits;
  section.experts = experts;
  section.bytes = checkedWeightMultiply(uint64_t(rows) * columns * bits / 8 +
                                        uint64_t(rows) * columns / 16, experts, "affine projection");
  uint64_t sourceRows = 0;
  for (const auto &[name, count] : parts) {
    section.parts.push_back(mlxQ4Part(image, name, count, columns, bits, experts));
    sourceRows += count;
  }
  if (!experts || sourceRows > rows || rows - sourceRows >= kQ4StorageN)
    throw WeightStoreError("invalid affine projection padding");
  append(image, std::move(section));
}

// An MLX Q4 tensor of [rows, columns] restored to BF16 as stored.
inline void dequantized(Image &image, const std::string &name, uint32_t rows, uint32_t columns) {
  if (!rows || !columns || columns % kQ4GroupElements) throw WeightStoreError("invalid affine Q4 tensor shape");
  Section section;
  section.kind = SectionKind::Dequantize;
  section.rows = rows;
  section.columns = columns;
  section.bytes = checkedWeightMultiply(uint64_t(rows) * columns, kBFloat16Bytes, "dequantized tensor");
  section.parts.push_back(mlxQ4Part(image, name, rows, columns));
  append(image, std::move(section));
}

// Binds every input of image to its checkpoint tensor, which must have one of
// the input's dtypes and its shape, once the checkpoint states the
// quantization of every affine module the image reads.
inline void bind(Image &image, const SafetensorsCheckpoint &source) {
  const auto bindInput = [&](Input &input) {
    const SourceTensor &tensor = source.require(input.name);
    if (std::find(input.dtypes.begin(), input.dtypes.end(), tensor.dtype) == input.dtypes.end() ||
        tensor.shape != input.shape)
      throw WeightStoreError("source tensor type or shape does not match: " + input.name);
    input.tensor = &tensor;
  };
  for (const auto &[module, bits] : image.quantized) source.requireQuantization(module, bits);
  for (Section &section : image.sections) {
    if (section.parts.empty()) bindInput(section.input);
    for (ProjectionPart &part : section.parts)
      for (Input &field : part.fields) bindInput(field);
  }
}

} // namespace splash::model::affine
