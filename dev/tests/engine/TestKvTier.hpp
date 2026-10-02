#pragma once

#include "model/Model.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace splash::test {

// A KV disk tier without a disk: slots count against a quota, every
// transfer holds one staging slot until it finishes, and transfers finish
// when the test says so.
class TestKvTier final : public model::KvTier {
public:
  struct Transfer final {
    bool ready = false;
    bool success = true;
    bool finished = false;
  };

  uint64_t slotBytes() const noexcept override { return 100; }
  bool writable() const noexcept override { return writableFile; }
  bool canDemote() const noexcept override { return writableFile && staging < stagingSlots; }
  std::shared_ptr<model::KvDiskSlot> acquireSlot() override {
    if (slots >= capacity) return {};
    return std::make_shared<Slot>(*this, nextIndex++);
  }
  std::unique_ptr<model::KvTransfer>
  demote(uint32_t, std::shared_ptr<model::KvDiskSlot>, std::function<void()>) override {
    if (!canDemote()) return {};
    ++demotions;
    return start();
  }
  std::unique_ptr<model::KvTransfer>
  restore(std::shared_ptr<model::KvDiskSlot>, uint32_t, std::function<void()>) override {
    if (staging >= stagingSlots) return {};
    ++restores;
    return start();
  }
  bool copiesQueued() const noexcept override { return queued; }
  void poll() override {}
  // Persistence: each slot carries the index it was adopted with, or the
  // next free one.
  uint32_t slotIndex(const model::KvDiskSlot &slot) const override {
    return static_cast<const Slot &>(slot).index;
  }
  std::vector<std::shared_ptr<model::KvDiskSlot>>
  adoptSlots(std::span<const uint32_t> indices) override {
    std::vector<std::shared_ptr<model::KvDiskSlot>> result;
    if (slots + indices.size() > capacity) return result;
    for (uint32_t index : indices) {
      result.push_back(std::make_shared<Slot>(*this, index));
      nextIndex = std::max(nextIndex, index + 1);
    }
    return result;
  }

  // Finishes every transfer still in flight.
  void complete(bool success = true) {
    for (auto &transfer : transfers) {
      if (transfer->finished || transfer->ready) continue;
      transfer->ready = true;
      transfer->success = success;
    }
    queued = false;
  }
  [[nodiscard]] uint32_t inFlight() const noexcept {
    uint32_t count = 0;
    for (const auto &transfer : transfers) count += !transfer->finished;
    return count;
  }

  uint32_t slots = 0;
  uint32_t capacity = 4;
  uint32_t staging = 0;
  uint32_t stagingSlots = 2;
  uint32_t demotions = 0;
  uint32_t restores = 0;
  bool writableFile = true;
  bool queued = false;
  uint32_t nextIndex = 0;
  std::vector<std::shared_ptr<Transfer>> transfers;

private:
  struct Slot final : model::KvDiskSlot {
    Slot(TestKvTier &owner, uint32_t position) : tier(owner), index(position) { ++tier.slots; }
    ~Slot() override { --tier.slots; }
    TestKvTier &tier;
    uint32_t index;
  };
  class Ticket final : public model::KvTransfer {
  public:
    Ticket(TestKvTier &owner, std::shared_ptr<Transfer> transfer)
        : tier_(owner), transfer_(std::move(transfer)) {}
    bool ready() const noexcept override { return transfer_->ready; }
    bool finish() override {
      if (!transfer_->finished) {
        transfer_->finished = true;
        --tier_.staging;
      }
      return transfer_->success;
    }

  private:
    TestKvTier &tier_;
    std::shared_ptr<Transfer> transfer_;
  };

  std::unique_ptr<model::KvTransfer> start() {
    auto transfer = std::make_shared<Transfer>();
    transfers.push_back(transfer);
    ++staging;
    queued = true;
    return std::make_unique<Ticket>(*this, std::move(transfer));
  }
};

} // namespace splash::test
