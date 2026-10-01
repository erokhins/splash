#pragma once

#include "model/PreparedFiles.hpp"

#include <memory>
#include <span>

namespace splash::model {

struct DFlashDraftLayout;

// A DFlash2 checkpoint, config.json and safetensors as its repository
// releases it (BF16) or as an MLX conversion quantized it (affine Q4 in
// groups of 64) -> the packed draft files of a Splash package
// (DFlashDraft.cpp): every projection in affine Q4, quantized as those drafts
// were or retiled as stored, the selector codebooks in BF16, as stored or
// restored from Q4, every other tensor copied as stored. The checkpoint is
// planned once; each file is prepared when it is opened.
class DraftCheckpointLoader final {
public:
  DraftCheckpointLoader(metal::MetalBackend &backend, const std::filesystem::path &directory,
                        const DFlashDraftLayout &layout, PreparationCheck admitConversion = {});
  ~DraftCheckpointLoader();
  // Every file's cache identity and size, layers first, for the model's disk
  // check before the first file is written.
  [[nodiscard]] std::span<const PreparedWeight> weights() const noexcept;
  // Writes every missing file and maps none.
  void prepare();
  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile model();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] uint64_t preparedDraftBytes(const DFlashDraftLayout &layout);

} // namespace splash::model
