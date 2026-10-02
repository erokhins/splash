#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace splash::model {

// Bytes one disk quota may hold, shared by every slot file of the cache
// tier. Reservations and releases come from the engine thread and from
// whoever drops the last handle of a slot.
class DiskBudget final {
public:
  explicit DiskBudget(uint64_t capacityBytes) noexcept : capacity_(capacityBytes) {}
  [[nodiscard]] uint64_t capacityBytes() const noexcept { return capacity_; }
  [[nodiscard]] uint64_t usedBytes() const noexcept {
    return used_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] bool reserve(uint64_t bytes) noexcept {
    uint64_t used = used_.load(std::memory_order_relaxed);
    do {
      if (bytes > capacity_ - used)
        return false;
    } while (!used_.compare_exchange_weak(used, used + bytes, std::memory_order_relaxed));
    return true;
  }
  void release(uint64_t bytes) noexcept { used_.fetch_sub(bytes, std::memory_order_relaxed); }
  // Cumulative bytes accepted by file IO, including partial/cancelled work.
  // This is application IO, not physical SSD traffic or filesystem overhead.
  [[nodiscard]] uint64_t readBytes() const noexcept {
    return read_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint64_t writtenBytes() const noexcept {
    return written_.load(std::memory_order_relaxed);
  }

private:
  friend class SlotFile;
  uint64_t capacity_;
  std::atomic<uint64_t> used_{0};
  std::atomic<uint64_t> read_{0};
  std::atomic<uint64_t> written_{0};
};

// Storage of fixed-size slots in a file, served by one IO worker in
// submission order and bounded by a disk budget. The file is an unlinked
// temporary one, gone with the process, or a named one kept across runs
// (persistent()), whose slots a saved index takes back with adopt(). Callers
// own the memory an operation moves and keep it alive until the operation is
// ready. A slot is readable only after one complete write; a failed or
// cancelled write leaves it unreadable, and after a failed write the file
// accepts no further writes. So that a file-size limit fails a write rather
// than killing the process, a file ignores SIGXFSZ from its construction on.
class SlotFile final {
  struct Backing;

public:
  // Slot offsets stay aligned to this for uncached IO.
  static constexpr uint64_t kAlignmentBytes = 16384;

  class Slot final {
  public:
    ~Slot();
    Slot(const Slot &) = delete;
    Slot &operator=(const Slot &) = delete;
    // The slot's position in the file, what a saved index records.
    [[nodiscard]] uint32_t index() const noexcept { return index_; }

  private:
    friend class SlotFile;
    Slot(std::shared_ptr<Backing> backing, uint32_t index);
    std::shared_ptr<Backing> backing_;
    uint32_t index_;
    // Owned by the worker: operations on one file run in submission order.
    bool written_ = false;
  };

  class Operation final {
  public:
    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] bool wait();
    void cancel() noexcept { cancelled_.store(true, std::memory_order_relaxed); }
    // Stops the operation before its next chunk, or before it starts, and
    // waits until the worker has let go of the memory it moves; the owner may
    // free that memory afterwards.
    void drain() {
      cancel();
      static_cast<void>(wait());
    }

  private:
    friend class SlotFile;
    std::atomic<bool> cancelled_{false};
    std::atomic<bool> done_{false};
    bool success_ = false;
    std::mutex mutex_;
    std::condition_variable wake_;
  };

  // The budget holds at least one slot: a smaller quota is a configuration
  // error. Files sharing a budget compete for its bytes.
  SlotFile(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget,
           const std::filesystem::path &directory = std::filesystem::temp_directory_path());
  // A file with a budget of its own.
  SlotFile(uint64_t slotBytes, uint64_t capacityBytes,
           const std::filesystem::path &directory = std::filesystem::temp_directory_path());
  // The named file, created when missing and kept when the object goes.
  // Every slot starts free: adopt() takes back the ones a saved index names.
  [[nodiscard]] static std::unique_ptr<SlotFile>
  persistent(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget,
             const std::filesystem::path &file);
  ~SlotFile();
  SlotFile(const SlotFile &) = delete;
  SlotFile &operator=(const SlotFile &) = delete;
  // Null when the budget is exhausted.
  [[nodiscard]] std::shared_ptr<Slot> acquire();
  // Takes back the slots a saved index names, in ascending order without
  // repeats, as written slots, on a file no slot of which has been handed
  // out. Every lower index not named is free; the file is cut after the
  // last one. Empty, leaving the file as it was, when the budget cannot hold
  // them all: the caller trims what it adopts to the quota first.
  [[nodiscard]] std::vector<std::shared_ptr<Slot>> adopt(std::span<const uint32_t> indices);
  // Pushes the slots written so far through to the device (F_FULLFSYNC),
  // after the operations submitted before it.
  [[nodiscard]] std::shared_ptr<Operation> flush();
  [[nodiscard]] uint64_t slotBytes() const noexcept;
  // The shared budget's capacity and use.
  [[nodiscard]] uint64_t capacityBytes() const noexcept;
  [[nodiscard]] uint64_t usedBytes() const noexcept;
  [[nodiscard]] uint64_t readBytes() const noexcept;
  [[nodiscard]] uint64_t writtenBytes() const noexcept;
  // False once a write has failed; complete slots stay readable.
  [[nodiscard]] bool writable() const noexcept;
  // True when the worker holds nothing: no operation queued and none running.
  // Owners drain their own operations; tests use this to check that none is
  // left.
  [[nodiscard]] bool idle() const;
  // The spans total one slot and stay valid until the operation is ready.
  [[nodiscard]] std::shared_ptr<Operation> write(
      std::shared_ptr<Slot> slot, std::vector<std::span<const std::byte>> source,
      std::function<void()> completion);
  [[nodiscard]] std::shared_ptr<Operation> read(
      std::shared_ptr<Slot> slot, std::vector<std::span<std::byte>> destination,
      std::function<void()> completion);

private:
  struct Work {
    std::shared_ptr<Operation> operation;
    std::function<bool(const std::atomic<bool> &)> run;
    std::function<void()> completion;
  };
  // Takes over an open descriptor.
  SlotFile(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget, int descriptor);
  [[nodiscard]] std::shared_ptr<Operation> submit(
      std::function<bool(const std::atomic<bool> &)> work,
      std::function<void()> completion);
  void run();
  std::shared_ptr<Backing> backing_;
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<Work> work_;
  bool running_ = false;
  bool stopping_ = false;
  std::thread worker_;
};

} // namespace splash::model
