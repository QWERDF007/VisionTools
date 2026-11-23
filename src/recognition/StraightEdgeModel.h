#pragma once

#include "StraightEdgeLocator.h"

#include <cstdint>
#include <span>
#include <vector>

namespace visiontools::recognition {

// The two measured reference edges. Runtime search regions and options belong to the node inputs.
class StraightEdgeModel
{
public:
    static StraightEdgeModel train(const cv::Mat &reference, const std::vector<SearchRectangle> &bands,
                                   const EdgeSearchOptions &options);

    std::vector<std::uint8_t> encode() const;
    static StraightEdgeModel  decode(std::span<const std::uint8_t> bytes);

    bool empty() const noexcept
    {
        return segments_.size() < 2;
    }

    const std::vector<ReferenceSegment> &segments() const noexcept
    {
        return segments_;
    }

private:
    static StraightEdgeModel read(std::span<const std::uint8_t> bytes);

    std::vector<ReferenceSegment> segments_;
};

} // namespace visiontools::recognition
