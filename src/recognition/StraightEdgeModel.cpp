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
constexpr std::uint16_t kFormatVersion = 1;
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

cv::Point2d bandDirection(const SearchRectangle &band)
{
    const double radians = band.angle * 3.14159265358979323846 / 180.0;
    return {std::cos(radians), std::sin(radians)};
}

std::uint8_t polarityCode(edge::EdgePolarity polarity)
{
    return static_cast<std::uint8_t>(polarity);
}

edge::EdgePolarity polarityOf(std::uint8_t code, const std::string &context)
{
    if (code > static_cast<std::uint8_t>(edge::EdgePolarity::LightToDark))
        corrupt(context + "的跃迁方向不合法");
    return static_cast<edge::EdgePolarity>(code);
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
    model.options_ = options;
    for (std::size_t index = 0; index < bands.size(); ++index)
    {
        const auto &band = bands[index];
        if (!(band.width > 1.0) || !(band.height > 0.0))
            fail(RecognitionError::StraightEdgeRegionsInvalid, bandName(index), "搜索矩形的宽度或高度不合法");
        // The long side of the drawn band is the edge; the short side is how far across it the
        // locator may look. The nominal segment is that long side through the band centre.
        const cv::Point2d direction = bandDirection(band);
        ReferenceSegment  segment;
        segment.name   = bandName(index);
        segment.from   = band.center - direction * (band.width / 2.0);
        segment.to     = band.center + direction * (band.width / 2.0);
        segment.search = band;

        const auto found = locateSegment(reference, segment, options);
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
    if (!(scale > 0.0) || std::abs(cross) < 1e-3 * scale)
        fail(RecognitionError::ReferenceSegmentsParallel, model.segments_[0].name + " 与 " + model.segments_[1].name);
    return model;
}

std::vector<std::uint8_t> StraightEdgeModel::encode() const
{
    if (empty())
        fail(RecognitionError::StraightEdgeModelIncompatible, "直边模型", "模型没有两条参考边");
    checkOptions(options_, "直边模型");
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
        mb::putDouble(bytes, segment.search.center.x);
        mb::putDouble(bytes, segment.search.center.y);
        mb::putDouble(bytes, segment.search.width);
        mb::putDouble(bytes, segment.search.height);
        mb::putDouble(bytes, segment.search.angle);
    }
    mb::putDouble(bytes, options_.contrast);
    bytes.push_back(static_cast<std::uint8_t>(std::clamp(options_.width, 1, 63)));
    bytes.push_back(polarityCode(options_.polarity));
    mb::putDouble(bytes, options_.minCompleteness);
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
        segment.search.center = {reader.f64(), reader.f64()};
        segment.search.width  = reader.f64();
        segment.search.height = reader.f64();
        segment.search.angle  = reader.f64();
        const double values[] = {segment.from.x,       segment.from.y,          segment.to.x,
                                 segment.to.y,         segment.search.center.x, segment.search.center.y,
                                 segment.search.width, segment.search.height,   segment.search.angle};
        for (const double value : values)
            if (!std::isfinite(value))
                corrupt("参考边 " + std::to_string(index + 1) + " 的几何量不合法");
        if (!(cv::norm(segment.to - segment.from) > 1.0))
            corrupt("参考边 " + std::to_string(index + 1) + " 长度不足");
        if (!(segment.search.width > 1.0) || !(segment.search.height > 0.0))
            corrupt("参考边 " + std::to_string(index + 1) + " 的搜索矩形不合法");
        model.segments_.push_back(std::move(segment));
    }

    model.options_.contrast        = reader.f64();
    model.options_.width           = reader.take(1)[0];
    model.options_.polarity        = polarityOf(reader.take(1)[0], "直边模型");
    model.options_.minCompleteness = reader.f64();
    checkOptions(model.options_, "直边模型");
    if (reader.remaining() != 0)
        corrupt("文件长度与参考边数量不符");
    return model;
}

} // namespace visiontools::recognition
