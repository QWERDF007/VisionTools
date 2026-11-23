#include "StraightEdgeModel.h"

#include "ModelBytes.h"
#include "common/errors/ToolError.h"
#include "errors/RecognitionErrors.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace visiontools::recognition {
namespace {
namespace mb = modelbytes;

constexpr std::uint8_t  kMagic[4]      = {'V', 'T', 'S', 'E'};
constexpr std::uint16_t kFormatVersion = 2;
constexpr std::size_t   kMaxNameBytes  = 64;
constexpr std::uint8_t  kSegmentCount  = 2;

[[noreturn]] void corrupt(const std::string &detail)
{
    fail(RecognitionError::StraightEdgeModelIncompatible, "直边模型", detail);
}

std::string bandName(std::size_t index)
{
    return "第 " + std::to_string(index + 1) + " 条参考边";
}

void checkOptions(const EdgeSearchOptions &options, const std::string &context)
{
    if (!std::isfinite(options.contrast) || options.contrast < 0.0 || options.contrast > 1024.0)
        corrupt(context + "的边缘对比度不合法");
    if (options.width < 1 || options.width > 63)
        corrupt(context + "的边缘宽度不合法");
    if (!std::isfinite(options.minCompleteness) || options.minCompleteness < 0.0 || options.minCompleteness > 1.0)
        corrupt(context + "的最低边完整度不合法");
}
} // namespace

StraightEdgeModel StraightEdgeModel::train(const cv::Mat &reference, const std::vector<SearchRectangle> &bands,
                                           const EdgeSearchOptions &options)
{
    if (reference.empty() || reference.type() != CV_8UC1)
        fail(RecognitionError::UnsupportedColorImage, "直边训练需要单通道 8 位灰度图");
    if (bands.size() != kSegmentCount)
        fail(RecognitionError::StraightEdgeRegionsInvalid, "参考边",
             "需要恰好两个搜索矩形，实际 " + std::to_string(bands.size()) + " 个");
    checkOptions(options, "训练参数");

    StraightEdgeModel model;

    for (std::size_t index = 0; index < bands.size(); ++index)
    {
        const auto &band = bands[index];
        if (!(band.width > 1.0) || !(band.height > 0.0))
            fail(RecognitionError::StraightEdgeRegionsInvalid, bandName(index), "搜索矩形的宽度或高度不合法");
        ReferenceSegment segment;
        segment.name     = bandName(index);
        const auto found = locateSegment(reference, band, options);
        if (!found)
            fail(RecognitionError::EdgeNotFound, segment.name, "搜索框内没有满足对比度与完整度的直边");
        // The published endpoints are the measured ones: a node never re-derives them from the
        // search band, so the reference pose is exactly what training saw.
        segment.from = found->start;
        segment.to   = found->end;
        model.segments_.push_back(std::move(segment));
    }

    const cv::Point2d first  = model.segments_[0].to - model.segments_[0].from;
    const cv::Point2d second = model.segments_[1].to - model.segments_[1].from;
    const double      cross  = first.x * second.y - first.y * second.x;
    const double      scale  = cv::norm(first) * cv::norm(second);
    if (!(scale > 0.0) || std::abs(cross) < kMinEdgeAngleSine * scale)
        fail(RecognitionError::ReferenceSegmentsParallel, model.segments_[0].name + " 与 " + model.segments_[1].name);
    return model;
}

std::vector<std::uint8_t> StraightEdgeModel::encode() const
{
    if (empty())
        fail(RecognitionError::StraightEdgeModelIncompatible, "直边模型", "模型没有两条参考边");

    std::vector<std::uint8_t> bytes;
    bytes.reserve(64 + segments_.size() * 96);
    bytes.insert(bytes.end(), std::begin(kMagic), std::end(kMagic));
    mb::putU16(bytes, kFormatVersion);
    bytes.push_back(kSegmentCount);
    for (const auto &segment : segments_)
    {
        mb::putText(bytes, segment.name, kMaxNameBytes);
        mb::putDouble(bytes, segment.from.x);
        mb::putDouble(bytes, segment.from.y);
        mb::putDouble(bytes, segment.to.x);
        mb::putDouble(bytes, segment.to.y);
    }
    return bytes;
}

StraightEdgeModel StraightEdgeModel::decode(std::span<const std::uint8_t> bytes)
{
    try
    {
        return read(bytes);
    }
    catch (const mb::Corrupt &error)
    {
        corrupt(error.detail);
    }
}

StraightEdgeModel StraightEdgeModel::read(std::span<const std::uint8_t> bytes)
{
    mb::Reader  reader{bytes};
    const auto *magic = reader.take(sizeof(kMagic));
    if (!std::equal(std::begin(kMagic), std::end(kMagic), magic))
        corrupt("文件标识不正确");
    if (reader.u16() != kFormatVersion)
        corrupt("版本不受支持");
    if (reader.take(1)[0] != kSegmentCount)
        corrupt("参考边数量不正确");

    StraightEdgeModel model;
    model.segments_.reserve(kSegmentCount);
    for (std::uint8_t index = 0; index < kSegmentCount; ++index)
    {
        ReferenceSegment segment;
        segment.name          = reader.text(kMaxNameBytes);
        segment.from          = {reader.f64(), reader.f64()};
        segment.to            = {reader.f64(), reader.f64()};
        const double values[] = {segment.from.x, segment.from.y, segment.to.x, segment.to.y};
        for (const double value : values)
            if (!std::isfinite(value))
                corrupt("参考边 " + std::to_string(index + 1) + " 的几何量不合法");
        if (!(cv::norm(segment.to - segment.from) > 1.0))
            corrupt("参考边 " + std::to_string(index + 1) + " 长度不足");
        model.segments_.push_back(std::move(segment));
    }

    if (reader.remaining() != 0)
        corrupt("文件长度与参考边数量不符");
    return model;
}

} // namespace visiontools::recognition
