#pragma once

#include "engine/Cache.hpp"
#include "model/SlotFile.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace splash::engine {

// The saved index's bytes: a header naming the cache namespace and the slot
// sizes, the image, and a SHA-256 of everything before it. Parsing rejects
// anything that does not match the header given, so a file of another
// model, build, KV format or page layout, or one damaged on disk, is
// discarded whole rather than read in part.
struct CacheIndexHeader final {
  CacheNamespace cacheNamespace;
  uint64_t kvSlotBytes = 0;
  uint64_t stateSlotBytes = 0;
};
[[nodiscard]] std::vector<std::byte> encodeCacheIndex(const CacheIndexHeader &header,
                                                      const CacheImage &image);
// Empty when the bytes are not an index with this header.
[[nodiscard]] std::optional<CacheImage> decodeCacheIndex(const CacheIndexHeader &header,
                                                         std::span<const std::byte> bytes);
// The image's newest states with their chains, as many as the quota holds.
[[nodiscard]] CacheImage trimCacheImage(const CacheImage &image, uint64_t kvSlotBytes,
                                        uint64_t stateSlotBytes, uint64_t capacityBytes);
// A state of the image whose slot lies beyond its file, or a block whose
// slot does, is a stale index: false then.
[[nodiscard]] bool cacheImageFitsFiles(const CacheImage &image, uint64_t kvSlotBytes,
                                       uint64_t kvFileBytes, uint64_t stateSlotBytes,
                                       uint64_t stateFileBytes);

// One cache namespace's disk tier across runs: a directory under the cache
// root named by the namespace, holding the KV and state slot files, the
// index naming the slots in use, and the lock that keeps one server to it.
// A marker stands while the directory is open; found at the next open, it
// says the previous run never saved, and the index is discarded. Files
// persist across runs; only the lock and the marker are this run's.
class PersistentCacheDirectory final {
public:
  struct Options final {
    std::filesystem::path root;
    std::string namespaceHex;
    CacheNamespace cacheNamespace;
    uint64_t kvSlotBytes = 0;
    uint64_t stateSlotBytes = 0;
    std::shared_ptr<model::DiskBudget> budget;
  };
  // Null when another process holds the directory; throws when it cannot
  // be made or opened.
  [[nodiscard]] static std::unique_ptr<PersistentCacheDirectory> open(const Options &options);
  ~PersistentCacheDirectory();
  PersistentCacheDirectory(const PersistentCacheDirectory &) = delete;
  PersistentCacheDirectory &operator=(const PersistentCacheDirectory &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const noexcept { return directory_; }
  [[nodiscard]] const std::shared_ptr<model::SlotFile> &kvFile() const noexcept { return kv_; }
  [[nodiscard]] const std::shared_ptr<model::SlotFile> &stateFile() const noexcept {
    return state_;
  }
  // The previous run left the marker: its index was discarded.
  [[nodiscard]] bool recoveredFromCrash() const noexcept { return crashed_; }

  // The saved index, validated and trimmed to the quota, or an empty image
  // without one. The note says what was dropped, for the log.
  [[nodiscard]] CacheImage load(std::string &note) const;
  // Flushes both files to the device, writes the index atomically and
  // clears the marker. False, with the error, when the index could not be
  // written; the marker then stands and the next run starts empty.
  [[nodiscard]] bool save(const CacheImage &image, std::string &error);

private:
  PersistentCacheDirectory(std::filesystem::path directory, int lock, bool crashed,
                           std::shared_ptr<model::SlotFile> kv,
                           std::shared_ptr<model::SlotFile> state, CacheIndexHeader header,
                           uint64_t capacityBytes);

  std::filesystem::path directory_;
  int lock_;
  bool crashed_;
  std::shared_ptr<model::SlotFile> kv_;
  std::shared_ptr<model::SlotFile> state_;
  CacheIndexHeader header_;
  uint64_t capacityBytes_;
};

struct CacheRestoreReport final {
  uint32_t blocks = 0;
  uint32_t states = 0;
  std::string note;
};
// Fills the cache from the directory's index: what the quota holds of it,
// or nothing when the index is absent, stale or inconsistent. Files are
// cut back when nothing is taken from them.
[[nodiscard]] CacheRestoreReport restorePersistentCache(const PersistentCacheDirectory &directory,
                                                        Cache &cache,
                                                        model::StateStorage &states);

} // namespace splash::engine
