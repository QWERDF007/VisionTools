#pragma once

#include "TestSupport.h"

#include <map>
#include <mutex>

namespace visiontools::tests {

class MatAllocationProbe final : public cv::MatAllocator
{
public:
    struct Snapshot
    {
        std::size_t allocations = 0;
        std::size_t releases    = 0;
        std::size_t liveBytes   = 0;
        std::size_t peakBytes   = 0;
    };

    MatAllocationProbe()
        : delegate_(cv::Mat::getDefaultAllocator())
    {
        cv::Mat::setDefaultAllocator(this);
    }

    ~MatAllocationProbe() override
    {
        cv::Mat::setDefaultAllocator(delegate_);
        REQUIRE(records_.empty(), "Tracked matrices must be released before the allocation probe");
    }

    MatAllocationProbe(const MatAllocationProbe &)            = delete;
    MatAllocationProbe &operator=(const MatAllocationProbe &) = delete;

    Snapshot snapshot(int type) const
    {
        std::lock_guard lock(mutex_);
        const auto      found = counts_.find(type);
        return found == counts_.end() ? Snapshot{} : found->second;
    }

    cv::UMatData *allocate(int dims, const int *sizes, int type, void *data, std::size_t *step, cv::AccessFlag flags,
                           cv::UMatUsageFlags usage) const override
    {
        auto           *allocation = delegate_->allocate(dims, sizes, type, data, step, flags, usage);
        std::lock_guard lock(mutex_);
        REQUIRE(records_.emplace(allocation, Record{type, allocation->size}).second,
                "Allocator must not return a simultaneously live allocation twice");
        auto &count = counts_[type];
        ++count.allocations;
        count.liveBytes += allocation->size;
        count.peakBytes           = std::max(count.peakBytes, count.liveBytes);
        allocation->currAllocator = this;
        return allocation;
    }

    bool allocate(cv::UMatData *data, cv::AccessFlag flags, cv::UMatUsageFlags usage) const override
    {
        return delegate_->allocate(data, flags, usage);
    }

    void deallocate(cv::UMatData *data) const override
    {
        {
            std::lock_guard lock(mutex_);
            const auto      found = records_.find(data);
            REQUIRE(found != records_.end(), "Only tracked allocations may reach the probe destructor");
            auto &count = counts_.at(found->second.type);
            ++count.releases;
            count.liveBytes -= found->second.bytes;
            records_.erase(found);
        }
        data->currAllocator = delegate_;
        delegate_->deallocate(data);
    }

private:
    struct Record
    {
        int         type;
        std::size_t bytes;
    };

    cv::MatAllocator                              *delegate_;
    mutable std::mutex                             mutex_;
    mutable std::map<const cv::UMatData *, Record> records_;
    mutable std::map<int, Snapshot>                counts_;
};

}
