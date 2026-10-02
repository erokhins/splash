#include "model/SlotFile.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace splash::model {

struct SlotFile::Backing {
  int descriptor = -1;
  uint64_t slotBytes = 0;
  std::shared_ptr<DiskBudget> budget;
  uint32_t allocated = 0;
  std::atomic<bool> failed{false};
  std::mutex mutex;
  std::vector<uint32_t> free;
  ~Backing() { if (descriptor >= 0) ::close(descriptor); }
};

SlotFile::Slot::Slot(std::shared_ptr<Backing> backing, uint32_t index)
    : backing_(std::move(backing)), index_(index) {}
SlotFile::Slot::~Slot() {
  if (!backing_) return;
  {
    std::lock_guard lock(backing_->mutex);
    backing_->free.push_back(index_);
  }
  backing_->budget->release(backing_->slotBytes);
}

bool SlotFile::Operation::ready() const noexcept {
  return done_.load(std::memory_order_acquire);
}
bool SlotFile::Operation::wait() {
  std::unique_lock lock(mutex_);
  wake_.wait(lock, [&] { return ready(); });
  return success_;
}

namespace {
int createUnlinked(const std::filesystem::path &directory) {
  std::string name = (directory / "splash-cache-XXXXXX").string();
  const int descriptor = ::mkstemp(name.data());
  if (descriptor < 0)
    throw std::system_error(errno, std::generic_category(), "create slot file");
  if (::unlink(name.c_str()) < 0) {
    const int error = errno;
    ::close(descriptor);
    throw std::system_error(error, std::generic_category(), "unlink slot file");
  }
  return descriptor;
}
} // namespace

SlotFile::SlotFile(uint64_t slotBytes, uint64_t capacityBytes,
                   const std::filesystem::path &directory)
    : SlotFile(slotBytes, std::make_shared<DiskBudget>(capacityBytes), directory) {}

SlotFile::SlotFile(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget,
                   const std::filesystem::path &directory)
    : SlotFile(slotBytes, std::move(budget), createUnlinked(directory)) {}

std::unique_ptr<SlotFile> SlotFile::persistent(uint64_t slotBytes,
                                               std::shared_ptr<DiskBudget> budget,
                                               const std::filesystem::path &file) {
  const int descriptor = ::open(file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
  if (descriptor < 0)
    throw std::system_error(errno, std::generic_category(), "open slot file");
  return std::unique_ptr<SlotFile>(new SlotFile(slotBytes, std::move(budget), descriptor));
}

SlotFile::SlotFile(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget, int descriptor)
    : backing_(std::make_shared<Backing>()) {
  // The backing closes the descriptor whatever happens below.
  backing_->descriptor = descriptor;
  if (!budget)
    throw std::invalid_argument("slot file needs a disk budget");
  const uint64_t capacityBytes = budget->capacityBytes();
  if (!slotBytes || capacityBytes > uint64_t{std::numeric_limits<off_t>::max()} ||
      capacityBytes / slotBytes > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("invalid slot file capacity");
  if (capacityBytes < slotBytes)
    throw std::invalid_argument("slot file quota holds no slot");
  if (slotBytes % kAlignmentBytes)
    throw std::invalid_argument("slot size is not aligned for uncached IO");
  backing_->slotBytes = slotBytes;
  backing_->budget = std::move(budget);
  // A write past the file-size limit (ulimit -f, a launchd FileSize) raises
  // SIGXFSZ, whose default action kills the process. Ignored, the write fails
  // with EFBIG and stops the file like any other storage error. The change is
  // process-wide, which is safe because every other engine writer checks its
  // errors; a handler someone installed is left alone.
  struct sigaction fileSize {};
  if (::sigaction(SIGXFSZ, nullptr, &fileSize) == 0 && fileSize.sa_handler == SIG_DFL)
    std::signal(SIGXFSZ, SIG_IGN);
  if (::fcntl(backing_->descriptor, F_SETFD, FD_CLOEXEC) < 0)
    throw std::system_error(errno, std::generic_category(), "close-on-exec slot file");
  // Avoid turning the cold tier into another long-lived RAM copy. Host VM
  // pressure accounting still covers any transient kernel IO memory.
  if (::fcntl(backing_->descriptor, F_NOCACHE, 1) < 0)
    throw std::system_error(errno, std::generic_category(), "uncached slot file");
  worker_ = std::thread([this] { run(); });
}

SlotFile::~SlotFile() {
  {
    std::lock_guard lock(mutex_);
    stopping_ = true;
    for (auto &work : work_) work.operation->cancel();
  }
  wake_.notify_one();
  worker_.join();
}

std::shared_ptr<SlotFile::Slot> SlotFile::acquire() {
  auto slot = std::shared_ptr<Slot>(new Slot({}, 0));
  std::lock_guard lock(backing_->mutex);
  // Prepare the free list before reserving bytes so allocation failure cannot
  // strand quota, and returning a slot never needs to allocate.
  if (backing_->free.empty() && backing_->free.capacity() == backing_->allocated)
    backing_->free.reserve(std::max<size_t>(1, 2 * backing_->free.capacity()));
  if (!backing_->budget->reserve(backing_->slotBytes)) return {};
  uint32_t index;
  if (!backing_->free.empty()) {
    index = backing_->free.back();
    backing_->free.pop_back();
  } else {
    index = backing_->allocated++;
  }
  slot->backing_ = backing_;
  slot->index_ = index;
  return slot;
}

std::vector<std::shared_ptr<SlotFile::Slot>> SlotFile::adopt(std::span<const uint32_t> indices) {
  for (size_t i = 1; i < indices.size(); ++i)
    if (indices[i] <= indices[i - 1])
      throw std::invalid_argument("adopted slots must ascend without repeats");
  const uint64_t slotBytes = backing_->slotBytes;
  if (!indices.empty() &&
      (indices.back() == std::numeric_limits<uint32_t>::max() ||
       (uint64_t{indices.back()} + 1) > uint64_t{std::numeric_limits<off_t>::max()} / slotBytes))
    throw std::invalid_argument("adopted slot lies beyond any file");
  // Everything that can fail to allocate happens before the file changes. A
  // slot without backing is inert: its destructor returns nothing.
  std::vector<std::shared_ptr<Slot>> slots;
  slots.reserve(indices.size());
  for (size_t i = 0; i < indices.size(); ++i)
    slots.push_back(std::shared_ptr<Slot>(new Slot({}, 0)));
  std::vector<uint32_t> gaps;
  uint32_t next = 0;
  for (uint32_t index : indices) {
    for (; next < index; ++next)
      gaps.push_back(next);
    next = index + 1;
  }
  std::lock_guard lock(backing_->mutex);
  if (backing_->allocated || !backing_->free.empty())
    throw std::logic_error("slots are adopted before any is handed out");
  if (!backing_->budget->reserve(indices.size() * slotBytes))
    return {};
  for (size_t i = 0; i < indices.size(); ++i) {
    slots[i]->backing_ = backing_;
    slots[i]->index_ = indices[i];
    slots[i]->written_ = true;
  }
  backing_->free = std::move(gaps);
  backing_->allocated = next;
  // A shorter run leaves no stale tail behind; a failure here costs only
  // disk space.
  static_cast<void>(::ftruncate(backing_->descriptor, static_cast<off_t>(next * slotBytes)));
  return slots;
}

std::shared_ptr<SlotFile::Operation> SlotFile::flush() {
  return submit([backing = backing_](const std::atomic<bool> &) {
    return ::fcntl(backing->descriptor, F_FULLFSYNC) >= 0 ||
           ::fsync(backing->descriptor) == 0;
  }, {});
}

uint64_t SlotFile::slotBytes() const noexcept { return backing_->slotBytes; }

uint64_t SlotFile::capacityBytes() const noexcept {
  return backing_->budget->capacityBytes();
}

uint64_t SlotFile::usedBytes() const noexcept { return backing_->budget->usedBytes(); }
uint64_t SlotFile::readBytes() const noexcept { return backing_->budget->readBytes(); }
uint64_t SlotFile::writtenBytes() const noexcept { return backing_->budget->writtenBytes(); }

bool SlotFile::writable() const noexcept {
  return !backing_->failed.load(std::memory_order_relaxed);
}

bool SlotFile::idle() const {
  std::lock_guard lock(mutex_);
  return work_.empty() && !running_;
}

std::shared_ptr<SlotFile::Operation> SlotFile::submit(
    std::function<bool(const std::atomic<bool> &)> run,
    std::function<void()> completion) {
  auto operation = std::make_shared<Operation>();
  {
    std::lock_guard lock(mutex_);
    if (stopping_)
      throw std::logic_error("slot file is shutting down");
    work_.push_back({operation, std::move(run), std::move(completion)});
  }
  wake_.notify_one();
  return operation;
}

void SlotFile::run() {
  for (;;) {
    Work work;
    {
      std::unique_lock lock(mutex_);
      wake_.wait(lock, [&] { return stopping_ || !work_.empty(); });
      if (work_.empty()) return;
      work = std::move(work_.front());
      work_.pop_front();
      running_ = true;
    }
    bool success = false;
    try { success = work.run(work.operation->cancelled_); }
    catch (...) { success = false; }
    work.run = {};
    // Idle before the operation reports, so whoever wakes on it sees a worker
    // that has let go of the memory it moved.
    {
      std::lock_guard lock(mutex_);
      running_ = false;
    }
    {
      std::lock_guard lock(work.operation->mutex_);
      work.operation->success_ = success;
      work.operation->done_.store(true, std::memory_order_release);
    }
    work.operation->wake_.notify_all();
    // A completion is a wake-up, not part of the transfer: one that throws
    // must not take the worker with it.
    if (work.completion) {
      try { work.completion(); } catch (...) {}
    }
  }
}

namespace {
constexpr size_t kChunkBytes = 1 << 20;

template <typename Span>
uint64_t totalBytes(const std::vector<Span> &spans) {
  uint64_t total = 0;
  for (auto span : spans) {
    if (span.size() > std::numeric_limits<uint64_t>::max() - total)
      throw std::invalid_argument("slot transfer size overflowed");
    total += span.size();
  }
  return total;
}

// Moves the spans in order through io(data, bytes, offset), one chunk at a
// time so that cancellation and short transfers are noticed promptly.
template <typename Span, typename Io>
bool transfer(const std::vector<Span> &spans, off_t offset,
              const std::atomic<bool> &cancelled, Io io) {
  for (auto span : spans) {
    while (!span.empty()) {
      if (cancelled.load(std::memory_order_relaxed)) return false;
      const ssize_t count = io(span.data(), std::min(span.size(), kChunkBytes), offset);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) return false;
      span = span.subspan(static_cast<size_t>(count));
      offset += count;
    }
  }
  return true;
}

off_t slotOffset(uint64_t index, uint64_t slotBytes) {
  return static_cast<off_t>(index * slotBytes);
}
} // namespace

std::shared_ptr<SlotFile::Operation> SlotFile::write(
    std::shared_ptr<Slot> slot, std::vector<std::span<const std::byte>> source,
    std::function<void()> completion) {
  if (!slot || slot->backing_ != backing_ || totalBytes(source) != backing_->slotBytes)
    throw std::invalid_argument("slot write does not match this file's slots");
  if (!writable())
    throw std::logic_error("slot file no longer takes writes");
  return submit([slot, source = std::move(source)](const std::atomic<bool> &cancelled) {
    Backing &backing = *slot->backing_;
    slot->written_ = false;
    if (transfer(source, slotOffset(slot->index_, backing.slotBytes), cancelled,
                 [&backing](const std::byte *data, size_t bytes, off_t offset) {
                   const auto count = ::pwrite(backing.descriptor, data, bytes, offset);
                   if (count > 0)
                     backing.budget->written_.fetch_add(count, std::memory_order_relaxed);
                   return count;
                 })) {
      slot->written_ = true;
      return true;
    }
    if (!cancelled.load(std::memory_order_relaxed))
      backing.failed.store(true, std::memory_order_relaxed);
    return false;
  }, std::move(completion));
}

std::shared_ptr<SlotFile::Operation> SlotFile::read(
    std::shared_ptr<Slot> slot, std::vector<std::span<std::byte>> destination,
    std::function<void()> completion) {
  if (!slot || slot->backing_ != backing_ || totalBytes(destination) != backing_->slotBytes)
    throw std::invalid_argument("slot read does not match this file's slots");
  return submit([slot, destination = std::move(destination)](const std::atomic<bool> &cancelled) {
    if (!slot->written_) return false;
    Backing &backing = *slot->backing_;
    return transfer(destination, slotOffset(slot->index_, slot->backing_->slotBytes), cancelled,
                    [&backing](std::byte *data, size_t bytes, off_t offset) {
                      const auto count = ::pread(backing.descriptor, data, bytes, offset);
                      if (count > 0)
                        backing.budget->read_.fetch_add(count, std::memory_order_relaxed);
                      return count;
                    });
  }, std::move(completion));
}

} // namespace splash::model
