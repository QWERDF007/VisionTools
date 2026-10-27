#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace visiontools::preprocess {

// Pure planning result of one image-read call. The cursor is an explicit file identity plus an
// exhausted flag, so a rescan can never silently restart a finished non-looping sequence and a
// reverse scan keeps its real successor.
struct SequencePlan
{
    std::vector<std::filesystem::path> files;
    std::vector<std::string>           keys;
    std::size_t                        startIndex = 0; // 当前项在目录中的正序下标
    std::string                        nextKey;       // 下一项文件标识；耗尽时为空
    std::string                        lastKey;       // 本次最后交付的文件标识
    bool                               exhausted = false;
};

struct SequenceCursor
{
    std::string nextKey;
    std::string lastKey;
    bool        exhausted = false;
    // At least one batch was delivered for this catalog and direction. Without it an empty cursor
    // is "first call", with it an empty cursor means "everything we knew about vanished", which is
    // an explicit end rather than a silent restart from the first file.
    bool started = false;
};

struct SequenceRequest
{
    bool        reverse = false;
    bool        loop    = false;
    std::size_t count   = 1;
};

// Where the next batch starts. The planner never guesses: a wrapped sequence asks for a restart
// explicitly, and a finished one ends explicitly.
enum class SequenceStart
{
    Found,
    RestartRequested,
    End
};

struct SequencePosition
{
    SequenceStart kind = SequenceStart::Found;
    std::size_t   step = 0; // meaningful for Found and RestartRequested
};

// Resolves the next start from the cursor, the current catalog and the requested direction.
// `catalog` must be sorted in forward order.
SequencePosition startPosition(const std::vector<std::string> &keys, const SequenceCursor &cursor, bool reverse,
                              bool loop);

// Throws visionrt::ContractError when a non-looping sequence has already finished.
SequencePlan planSequence(const std::vector<std::filesystem::path> &catalog, const SequenceCursor &cursor,
                          const SequenceRequest &request);

// Bounded decode cache. It only releases its own references: images handed out through DataRef keep
// their owner alive for as long as a consumer holds them. The retained history never exceeds its
// byte budget; the last successful batch is owned separately by the tool, so it is not part of that
// accounting and is never used to justify exceeding it.
class DecodedImageCache
{
public:
    explicit DecodedImageCache(std::size_t budgetBytes);

    // A hit is a use: it moves the entry to the front so eviction keeps real recency.
    const cv::Mat *find(const std::string &key);
    // Returns false when the image cannot fit inside the budget even after evicting everything that
    // is not protected: the caller may still return it once, but it never enters the history.
    bool insert(const std::string &key, const cv::Mat &image);
    // Protects the keys of the last successful output batch from eviction.
    void protect(const std::vector<std::string> &keys);
    void clear();
    std::size_t bytes() const noexcept { return bytes_; }
    std::size_t budget() const noexcept { return budget_; }
    std::size_t size() const noexcept { return entries_.size(); }

private:
    struct Entry
    {
        cv::Mat                   image;
        std::size_t               bytes = 0;
        std::list<std::string>::iterator order;
    };

    // Drops least recently used entries until `incoming` fits. Returns false when the protected
    // batch alone leaves no room, in which case nothing is retained for this key.
    bool evictUntilWithinBudget(std::size_t incoming);

    std::size_t                            budget_;
    std::size_t                            bytes_ = 0;
    std::list<std::string>                 order_; // front = most recent
    std::unordered_map<std::string, Entry> entries_;
    std::map<std::string, char>            protected_;
};

} // namespace visiontools::preprocess
