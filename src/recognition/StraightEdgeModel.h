#pragma once

#include "StraightEdgeLocator.h"

#include <cstdint>
#include <span>
#include <vector>

namespace visiontools::recognition {

// The published model of one alignment node: the two reference edges measured on the training image,
// the search bands they were measured through, and the search parameters that produced them. The
// trainer writes it, the alignment tool reads it, and neither keeps a second copy of an endpoint.
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

    const EdgeSearchOptions &search() const noexcept
    {
        return options_;
    }

private:
    static StraightEdgeModel read(std::span<const std::uint8_t> bytes);

    std::vector<ReferenceSegment> segments_;
    EdgeSearchOptions             options_;
};

} // namespace visiontools::recognition
