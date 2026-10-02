#include "engine/CachePersistence.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <unordered_set>

namespace splash::engine {
namespace {

constexpr std::string_view kMagic = "SPLASH-KV-INDEX1";
constexpr uint32_t kVersion = 1;
// A count beyond this comes from a damaged file, not a cache.
constexpr uint64_t kMaximumEntries = uint64_t{1} << 24;
constexpr std::string_view kIndexName = "index";
constexpr std::string_view kIndexDraftName = "index.tmp";
constexpr std::string_view kKvName = "kv.slots";
constexpr std::string_view kStateName = "state.slots";
constexpr std::string_view kLockName = "lock";
constexpr std::string_view kOpenName = "open";

class Writer final {
public:
  template <typename T> void put(T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto *begin = reinterpret_cast<const std::byte *>(&value);
    bytes_.insert(bytes_.end(), begin, begin + sizeof value);
  }
  void put(std::span<const std::byte> bytes) {
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
  }
  [[nodiscard]] std::span<const std::byte> written() const noexcept { return bytes_; }
  [[nodiscard]] std::vector<std::byte> take() noexcept { return std::move(bytes_); }

private:
  std::vector<std::byte> bytes_;
};

class Reader final {
public:
  explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}
  template <typename T> [[nodiscard]] bool get(T &value) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (remaining() < sizeof value)
      return false;
    std::memcpy(&value, bytes_.data() + position_, sizeof value);
    position_ += sizeof value;
    return true;
  }
  [[nodiscard]] bool get(std::span<std::byte> out) {
    if (remaining() < out.size())
      return false;
    std::copy_n(bytes_.data() + position_, out.size(), out.data());
    position_ += out.size();
    return true;
  }
  [[nodiscard]] size_t remaining() const noexcept { return bytes_.size() - position_; }

private:
  std::span<const std::byte> bytes_;
  size_t position_ = 0;
};

std::array<std::byte, CC_SHA256_DIGEST_LENGTH> sha256(std::span<const std::byte> bytes) {
  if (bytes.size() > std::numeric_limits<CC_LONG>::max())
    throw std::overflow_error("cache index is too large to hash");
  std::array<std::byte, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()),
            reinterpret_cast<unsigned char *>(digest.data()));
  return digest;
}

void putHeader(Writer &writer, const CacheIndexHeader &header) {
  writer.put(std::as_bytes(std::span(kMagic)));
  writer.put(kVersion);
  writer.put(std::as_bytes(std::span(header.cacheNamespace.digest)));
  writer.put(header.kvSlotBytes);
  writer.put(header.stateSlotBytes);
  writer.put(uint32_t{KvCache::pageTokens});
}

constexpr size_t kBlockBytes = 8 + 8 + 4 * KvCache::pageTokens + 8 + 8 + 4 + 1;
constexpr size_t kStateBytes = 8 + 4 + 8 * 4 + 1;

class Descriptor final {
public:
  explicit Descriptor(int value) noexcept : value_(value) {}
  ~Descriptor() {
    if (value_ >= 0)
      ::close(value_);
  }
  Descriptor(const Descriptor &) = delete;
  Descriptor &operator=(const Descriptor &) = delete;
  [[nodiscard]] int get() const noexcept { return value_; }
  [[nodiscard]] int release() noexcept { return std::exchange(value_, -1); }

private:
  int value_;
};

bool syncFully(int descriptor) noexcept {
  return ::fcntl(descriptor, F_FULLFSYNC) >= 0 || ::fsync(descriptor) == 0;
}

bool syncDirectory(const std::filesystem::path &directory) noexcept {
  const Descriptor fd(::open(directory.c_str(), O_RDONLY | O_CLOEXEC));
  return fd.get() >= 0 && ::fsync(fd.get()) == 0;
}

uint64_t fileBytes(const std::filesystem::path &file) noexcept {
  std::error_code error;
  const auto size = std::filesystem::file_size(file, error);
  return error ? 0 : size;
}

} // namespace

std::vector<std::byte> encodeCacheIndex(const CacheIndexHeader &header, const CacheImage &image) {
  Writer writer;
  putHeader(writer, header);
  writer.put(uint64_t{image.blocks.size()});
  writer.put(uint64_t{image.states.size()});
  for (const CacheImage::Block &block : image.blocks) {
    writer.put(block.id);
    writer.put(block.parent);
    for (uint32_t token : block.tokens)
      writer.put(token);
    writer.put(block.images.lo);
    writer.put(block.images.hi);
    writer.put(block.slot);
    writer.put(uint8_t{block.hadState});
  }
  for (const CacheImage::State &state : image.states) {
    writer.put(state.block);
    writer.put(state.persisted.slot);
    for (uint64_t value : state.persisted.metadata)
      writer.put(value);
    writer.put(uint8_t{state.checkpoint});
  }
  const auto digest = sha256(writer.written());
  writer.put(std::span<const std::byte>(digest));
  return writer.take();
}

std::optional<CacheImage> decodeCacheIndex(const CacheIndexHeader &header,
                                           std::span<const std::byte> bytes) {
  Writer expected;
  putHeader(expected, header);
  const std::span<const std::byte> headerBytes = expected.written();
  if (bytes.size() < headerBytes.size() + 16 + CC_SHA256_DIGEST_LENGTH)
    return std::nullopt;
  const std::span<const std::byte> body = bytes.first(bytes.size() - CC_SHA256_DIGEST_LENGTH);
  if (!std::equal(headerBytes.begin(), headerBytes.end(), body.begin()))
    return std::nullopt;
  const auto digest = sha256(body);
  if (!std::equal(digest.begin(), digest.end(), bytes.end() - CC_SHA256_DIGEST_LENGTH))
    return std::nullopt;
  Reader reader(body.subspan(headerBytes.size()));
  uint64_t blocks = 0, states = 0;
  if (!reader.get(blocks) || !reader.get(states) || blocks > kMaximumEntries ||
      states > kMaximumEntries ||
      reader.remaining() != blocks * kBlockBytes + states * kStateBytes)
    return std::nullopt;
  CacheImage image;
  image.blocks.resize(blocks);
  image.states.resize(states);
  for (CacheImage::Block &block : image.blocks) {
    uint8_t hadState = 0;
    bool ok = reader.get(block.id) && reader.get(block.parent);
    for (uint32_t &token : block.tokens)
      ok = ok && reader.get(token);
    ok = ok && reader.get(block.images.lo) && reader.get(block.images.hi) &&
         reader.get(block.slot) && reader.get(hadState);
    if (!ok || hadState > 1)
      return std::nullopt;
    block.hadState = hadState;
  }
  for (CacheImage::State &state : image.states) {
    uint8_t checkpoint = 0;
    bool ok = reader.get(state.block) && reader.get(state.persisted.slot);
    for (uint64_t &value : state.persisted.metadata)
      ok = ok && reader.get(value);
    ok = ok && reader.get(checkpoint);
    if (!ok || checkpoint > 1)
      return std::nullopt;
    state.checkpoint = checkpoint;
  }
  return image;
}

CacheImage trimCacheImage(const CacheImage &image, uint64_t kvSlotBytes, uint64_t stateSlotBytes,
                          uint64_t capacityBytes) {
  std::unordered_map<uint64_t, const CacheImage::Block *> byId;
  for (const CacheImage::Block &block : image.blocks)
    byId.emplace(block.id, &block);
  std::unordered_set<uint64_t> keptBlocks;
  std::vector<bool> keptStates(image.states.size(), false);
  uint64_t used = 0;
  // Newest first; the first state that does not fit ends it, so what is
  // kept is the newest states, each with the chain it restores through.
  for (size_t index = image.states.size(); index > 0; --index) {
    const CacheImage::State &state = image.states[index - 1];
    std::vector<uint64_t> chain;
    bool complete = true;
    for (uint64_t id = state.block; id && !keptBlocks.contains(id);) {
      const auto found = byId.find(id);
      if (found == byId.end()) {
        complete = false;
        break;
      }
      chain.push_back(id);
      id = found->second->parent;
    }
    if (!complete)
      continue;
    const uint64_t cost = chain.size() * kvSlotBytes + stateSlotBytes;
    if (cost > capacityBytes - used)
      break;
    used += cost;
    keptBlocks.insert(chain.begin(), chain.end());
    keptStates[index - 1] = true;
  }
  CacheImage trimmed;
  for (const CacheImage::Block &block : image.blocks)
    if (keptBlocks.contains(block.id))
      trimmed.blocks.push_back(block);
  for (size_t index = 0; index < image.states.size(); ++index)
    if (keptStates[index])
      trimmed.states.push_back(image.states[index]);
  return trimmed;
}

bool cacheImageFitsFiles(const CacheImage &image, uint64_t kvSlotBytes, uint64_t kvFileBytes,
                         uint64_t stateSlotBytes, uint64_t stateFileBytes) {
  const auto within = [](uint32_t slot, uint64_t slotBytes, uint64_t fileBytes) {
    return slotBytes && uint64_t{slot} < fileBytes / slotBytes;
  };
  for (const CacheImage::Block &block : image.blocks)
    if (!within(block.slot, kvSlotBytes, kvFileBytes))
      return false;
  for (const CacheImage::State &state : image.states)
    if (!within(state.persisted.slot, stateSlotBytes, stateFileBytes))
      return false;
  return true;
}

std::unique_ptr<PersistentCacheDirectory>
PersistentCacheDirectory::open(const Options &options) {
  if (options.namespaceHex.empty() || !options.budget)
    throw std::invalid_argument("a persistent cache directory needs a namespace and a budget");
  const std::filesystem::path directory = options.root / options.namespaceHex;
  std::filesystem::create_directories(directory);
  Descriptor lock(::open((directory / kLockName).c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600));
  if (lock.get() < 0)
    throw std::system_error(errno, std::generic_category(), "open cache lock");
  if (::flock(lock.get(), LOCK_EX | LOCK_NB) < 0) {
    if (errno == EWOULDBLOCK)
      return nullptr;
    throw std::system_error(errno, std::generic_category(), "lock cache directory");
  }
  const std::filesystem::path marker = directory / kOpenName;
  const bool crashed = std::filesystem::exists(marker);
  if (crashed) {
    // The previous run never saved: its index names slots that were
    // overwritten since.
    std::error_code ignored;
    std::filesystem::remove(directory / kIndexName, ignored);
    std::filesystem::remove(directory / kIndexDraftName, ignored);
  }
  {
    const Descriptor opened(::open(marker.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0600));
    if (opened.get() < 0)
      throw std::system_error(errno, std::generic_category(), "mark cache directory open");
    static_cast<void>(syncFully(opened.get()));
  }
  // The marker reaches the disk before any slot of the files changes.
  static_cast<void>(syncDirectory(directory));
  std::shared_ptr<model::SlotFile> kv =
      model::SlotFile::persistent(options.kvSlotBytes, options.budget, directory / kKvName);
  std::shared_ptr<model::SlotFile> state =
      model::SlotFile::persistent(options.stateSlotBytes, options.budget, directory / kStateName);
  return std::unique_ptr<PersistentCacheDirectory>(new PersistentCacheDirectory(
      directory, lock.release(), crashed, std::move(kv), std::move(state),
      {options.cacheNamespace, options.kvSlotBytes, options.stateSlotBytes},
      options.budget->capacityBytes()));
}

PersistentCacheDirectory::PersistentCacheDirectory(std::filesystem::path directory, int lock,
                                                   bool crashed,
                                                   std::shared_ptr<model::SlotFile> kv,
                                                   std::shared_ptr<model::SlotFile> state,
                                                   CacheIndexHeader header,
                                                   uint64_t capacityBytes)
    : directory_(std::move(directory)), lock_(lock), crashed_(crashed), kv_(std::move(kv)),
      state_(std::move(state)), header_(header), capacityBytes_(capacityBytes) {}

PersistentCacheDirectory::~PersistentCacheDirectory() {
  // The files close with their last owner; the lock goes with this process.
  ::close(lock_);
}

CacheImage PersistentCacheDirectory::load(std::string &note) const {
  const std::filesystem::path index = directory_ / kIndexName;
  std::ifstream input(index, std::ios::binary);
  if (!input)
    return {};
  const std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                                std::istreambuf_iterator<char>());
  std::optional<CacheImage> decoded =
      decodeCacheIndex(header_, std::as_bytes(std::span<const char>(bytes)));
  if (!decoded) {
    note = "the saved index is damaged or of another configuration";
    return {};
  }
  if (!cacheImageFitsFiles(*decoded, header_.kvSlotBytes, fileBytes(directory_ / kKvName),
                           header_.stateSlotBytes, fileBytes(directory_ / kStateName))) {
    note = "the saved index names slots beyond its files";
    return {};
  }
  CacheImage trimmed =
      trimCacheImage(*decoded, header_.kvSlotBytes, header_.stateSlotBytes, capacityBytes_);
  if (trimmed.states.size() != decoded->states.size())
    note = "the quota holds " + std::to_string(trimmed.states.size()) + " of the " +
           std::to_string(decoded->states.size()) + " saved states";
  return trimmed;
}

bool PersistentCacheDirectory::save(const CacheImage &image, std::string &error) {
  const auto kvFlushed = kv_->flush();
  const auto stateFlushed = state_->flush();
  if (!kvFlushed->wait() || !stateFlushed->wait()) {
    error = "the slot files could not be flushed";
    return false;
  }
  const std::vector<std::byte> bytes = encodeCacheIndex(header_, image);
  const std::filesystem::path draft = directory_ / kIndexDraftName;
  {
    const Descriptor output(
        ::open(draft.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600));
    if (output.get() < 0) {
      error = std::string("cannot create the index: ") + std::strerror(errno);
      return false;
    }
    std::span<const std::byte> rest(bytes);
    while (!rest.empty()) {
      const ssize_t count = ::write(output.get(), rest.data(), rest.size());
      if (count < 0 && errno == EINTR)
        continue;
      if (count <= 0) {
        error = std::string("cannot write the index: ") + std::strerror(errno);
        return false;
      }
      rest = rest.subspan(static_cast<size_t>(count));
    }
    if (!syncFully(output.get())) {
      error = std::string("cannot flush the index: ") + std::strerror(errno);
      return false;
    }
  }
  std::error_code failure;
  std::filesystem::rename(draft, directory_ / kIndexName, failure);
  if (failure) {
    error = "cannot publish the index: " + failure.message();
    return false;
  }
  static_cast<void>(syncDirectory(directory_));
  std::filesystem::remove(directory_ / kOpenName, failure);
  if (failure) {
    error = "cannot clear the open marker: " + failure.message();
    return false;
  }
  static_cast<void>(syncDirectory(directory_));
  return true;
}

CacheRestoreReport restorePersistentCache(const PersistentCacheDirectory &directory, Cache &cache,
                                          model::StateStorage &states) {
  CacheRestoreReport report;
  const CacheImage image = directory.load(report.note);
  const auto cut = [&] {
    // Nothing taken: the files start over.
    static_cast<void>(directory.kvFile()->adopt({}));
    static_cast<void>(directory.stateFile()->adopt({}));
  };
  if (image.states.empty()) {
    cut();
    return report;
  }
  try {
    if (!cache.importDisk(image, states)) {
      report.note = "the quota could not take the saved cache";
      return report;
    }
  } catch (const std::exception &error) {
    report.note = std::string("the saved index is inconsistent (") + error.what() + ")";
    return report;
  }
  report.blocks = static_cast<uint32_t>(image.blocks.size());
  report.states = static_cast<uint32_t>(image.states.size());
  return report;
}

} // namespace splash::engine
