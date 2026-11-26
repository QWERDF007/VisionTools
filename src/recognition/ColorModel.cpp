#include "ColorModel.h"

#include "ModelBytes.h"
#include "common/GeometryTransform.h"
#include "common/RoiSupport.h"
#include "common/errors/ToolError.h"
#include "errors/RecognitionErrors.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace visiontools::recognition {
namespace {
// Ridge added to both chroma variances: a class trained on a perfectly flat patch must not reject
// its own pixels because the covariance is singular.
constexpr double kCovarianceRidge = 4.0;
// A class below this many pixels has no usable statistics at all.
constexpr std::size_t kMinClassPixels = 64;
// The file is data, not code: a corrupt header must fail before any large allocation.
constexpr std::uint32_t kMaxClasses    = 256;
constexpr std::uint16_t kMaxLabelBytes = 1024;

struct Accumulator
{
    std::size_t pixels = 0;
    double      u      = 0;
    double      v      = 0;
    double      uu     = 0;
    double      uv     = 0;
    double      vv     = 0;
};

[[noreturn]] void corrupt(const std::string &detail)
{
    fail(RecognitionError::ColorModelIncompatible, "颜色模型", detail);
}

using modelbytes::putDouble;
using modelbytes::putU16;
using modelbytes::putU32;
using modelbytes::putU64;
using Reader = modelbytes::Reader;

void accumulate(const cv::Mat &yuv, const cv::Mat &mask, Accumulator &accumulator)
{
    for (int row = 0; row < yuv.rows; ++row)
    {
        const auto *pixels = yuv.ptr<cv::Vec3b>(row);
        const auto *valid  = mask.ptr<std::uint8_t>(row);
        for (int column = 0; column < yuv.cols; ++column)
        {
            if (valid[column] == 0)
                continue;
            const double u = pixels[column][1];
            const double v = pixels[column][2];
            accumulator.pixels += 1;
            accumulator.u += u;
            accumulator.v += v;
            accumulator.uu += u * u;
            accumulator.uv += u * v;
            accumulator.vv += v * v;
        }
    }
}

constexpr std::uint8_t  kMagic[4]      = {'V', 'T', 'C', 'M'};
constexpr std::uint16_t kFormatVersion = 1;
// 0 = OpenCV YUV 4:4:4, channels 1 and 2 are U and V. The value is part of the model so a future
// preprocessing change cannot be read with today's convention.
constexpr std::uint8_t  kChromaSpaceYuv = 0;
constexpr std::uint8_t  kChromaChannels = 2;
} // namespace

ChromaImage::ChromaImage(const cv::Mat &bgr)
{
    if (bgr.type() != CV_8UC3)
        fail(RecognitionError::UnsupportedColorImage, "图像");
    cv::cvtColor(bgr, planes_, cv::COLOR_BGR2YUV);
}

ColorModel ColorModel::train(const std::vector<Sample> &samples)
{
    std::map<std::string, Accumulator> accumulators;
    for (const auto &sample : samples)
    {
        if (sample.label.empty())
            fail(RecognitionError::ColorClassMissing, "颜色样本");
        if (sample.regions.empty())
            fail(RecognitionError::ColorSamplesMissing, sample.label);
        if (sample.pixels.type() != CV_8UC3)
            fail(RecognitionError::UnsupportedColorImage, sample.label);
        const auto bounded = roi::boundedRegionMask(sample.regions, sample.pixels.size(), geometry::Alignment::identity());
        if (bounded.empty())
            continue;
        cv::Mat patchYuv;
        cv::cvtColor(sample.pixels(bounded.bounds), patchYuv, cv::COLOR_BGR2YUV);
        accumulate(patchYuv, bounded.mask, accumulators[sample.label]);
    }

    ColorModel model;
    for (const auto &[label, accumulator] : accumulators)
    {
        if (accumulator.pixels < kMinClassPixels)
            fail(RecognitionError::ColorSamplesTooSmall, label,
                 std::to_string(accumulator.pixels) + " 个像素，至少需要 " + std::to_string(kMinClassPixels));
        const double count = static_cast<double>(accumulator.pixels);
        const double meanU = accumulator.u / count;
        const double meanV = accumulator.v / count;
        const double varU  = accumulator.uu / count - meanU * meanU + kCovarianceRidge;
        const double varV  = accumulator.vv / count - meanV * meanV + kCovarianceRidge;
        const double covUV = accumulator.uv / count - meanU * meanV;
        const double det   = varU * varV - covUV * covUV;
        if (!(det > 1e-9))
            fail(RecognitionError::ColorSamplesTooSmall, label, "协方差矩阵不可逆");

        model.classes_.push_back(ColorClass{label, {meanU, meanV},
                                            cv::Matx22d{varV / det, -covUV / det, -covUV / det, varU / det},
                                            std::log(det), accumulator.pixels});
    }
    if (model.classes_.empty())
        fail(RecognitionError::ColorSamplesMissing, "训练数据");
    model.buildLookupTable();
    return model;
}

std::vector<std::uint8_t> ColorModel::encode() const
{
    if (classes_.empty())
        fail(RecognitionError::ColorSamplesMissing, "颜色模型");
    std::vector<std::uint8_t> bytes;
    bytes.reserve(28 + classes_.size() * 72);
    bytes.insert(bytes.end(), std::begin(kMagic), std::end(kMagic));
    putU16(bytes, kFormatVersion);
    bytes.push_back(kChromaSpaceYuv);
    bytes.push_back(kChromaChannels);
    putU16(bytes, 0);
    putDouble(bytes, kCovarianceRidge);
    putU32(bytes, static_cast<std::uint32_t>(classes_.size()));
    for (const auto &color : classes_)
    {
        if (color.label.size() > kMaxLabelBytes)
            fail(RecognitionError::ColorClassMissing, color.label, "类别名过长");
        putU16(bytes, static_cast<std::uint16_t>(color.label.size()));
        bytes.insert(bytes.end(), color.label.begin(), color.label.end());
        putDouble(bytes, color.mean[0]);
        putDouble(bytes, color.mean[1]);
        putDouble(bytes, color.inverse(0, 0));
        putDouble(bytes, color.inverse(0, 1));
        putDouble(bytes, color.inverse(1, 0));
        putDouble(bytes, color.inverse(1, 1));
        putDouble(bytes, color.logDeterminant);
        putU64(bytes, static_cast<std::uint64_t>(color.pixels));
    }
    return bytes;
}

ColorModel ColorModel::decode(std::span<const std::uint8_t> bytes)
{
    try
    {
        return read(bytes);
    }
    catch (const modelbytes::Corrupt &error)
    {
        corrupt(error.detail);
    }
}

ColorModel ColorModel::read(std::span<const std::uint8_t> bytes)
{
    Reader      reader{bytes};
    const auto *magic = reader.take(sizeof(kMagic));
    if (!std::equal(std::begin(kMagic), std::end(kMagic), magic))
        corrupt("文件标识不正确");
    if (reader.u16() != kFormatVersion)
        corrupt("版本不受支持");
    const auto space    = reader.take(1)[0];
    const auto channels = reader.take(1)[0];
    if (space != kChromaSpaceYuv || channels != kChromaChannels)
        corrupt("色彩预处理约定不受支持");
    if (reader.u16() != 0)
        corrupt("保留字段必须为零");
    const double ridge = reader.f64();
    if (!std::isfinite(ridge) || !(ridge > 0))
        corrupt("正则参数不合法");
    const auto count = reader.u32();
    if (count == 0 || count > kMaxClasses)
        corrupt("类别数量不合法");

    ColorModel model;
    model.classes_.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index)
    {
        const auto labelBytes = reader.u16();
        if (labelBytes == 0 || labelBytes > kMaxLabelBytes)
            corrupt("类别名长度不合法");
        const auto *label = reader.take(labelBytes);
        ColorClass  color;
        color.label          = std::string(reinterpret_cast<const char *>(label), labelBytes);
        color.mean           = {reader.f64(), reader.f64()};
        color.inverse        = cv::Matx22d{reader.f64(), reader.f64(), reader.f64(), reader.f64()};
        color.logDeterminant = reader.f64();
        color.pixels         = static_cast<std::size_t>(reader.u64());
        if (!std::isfinite(color.mean[0]) || !std::isfinite(color.mean[1]) || !std::isfinite(color.inverse(0, 0))
            || !std::isfinite(color.inverse(0, 1)) || !std::isfinite(color.inverse(1, 0))
            || !std::isfinite(color.inverse(1, 1)) || !std::isfinite(color.logDeterminant) || color.pixels == 0)
            corrupt("类别统计量不合法");
        model.classes_.push_back(std::move(color));
    }
    if (reader.remaining() != 0)
        corrupt("文件长度与类别数量不符");
    model.buildLookupTable();
    return model;
}

void ColorModel::buildLookupTable()
{
    uvLookup_.assign(65536, LookupEntry{});
    for (int u = 0; u < 256; ++u)
    {
        const double ud = static_cast<double>(u);
        for (int v = 0; v < 256; ++v)
        {
            const double  vd           = static_cast<double>(v);
            std::uint16_t best         = 0;
            double        bestDensity  = -std::numeric_limits<double>::infinity();
            double        bestDistance = 0.0;
            for (std::size_t index = 0; index < classes_.size(); ++index)
            {
                const auto  &color = classes_[index];
                const double du    = ud - color.mean[0];
                const double dv    = vd - color.mean[1];
                const double distance = du * (color.inverse(0, 0) * du + color.inverse(0, 1) * dv)
                                        + dv * (color.inverse(1, 0) * du + color.inverse(1, 1) * dv);
                const double density = -0.5 * distance - 0.5 * color.logDeterminant;
                if (density > bestDensity)
                {
                    bestDensity  = density;
                    bestDistance = distance;
                    best         = static_cast<std::uint16_t>(index);
                }
            }
            uvLookup_[static_cast<std::size_t>((u << 8) | v)]
                = {std::exp(-0.5 * std::max(0.0, bestDistance)), best};
        }
    }
}

ColorModel::Classification ColorModel::finalizeClassification(const std::vector<std::size_t> &histogram,
                                                              const std::vector<double> &scoreSums, std::size_t total,
                                                              std::size_t accepted, double minRatio) const
{
    Classification result;
    if (total == 0)
        return result;
    result.ratio = static_cast<double>(accepted) / static_cast<double>(total);
    if (result.ratio < minRatio)
        return result;

    std::size_t winner = 0;
    for (std::size_t index = 1; index < histogram.size(); ++index)
        if (histogram[index] > histogram[winner])
            winner = index;
    if (histogram[winner] == 0)
        return result;
    result.label = classes_[winner].label;
    result.score = scoreSums[winner] / static_cast<double>(histogram[winner]);
    return result;
}

ColorModel::Classification ColorModel::classify(const ChromaImage &image, const cv::Mat &mask, double minScore,
                                                double minRatio) const
{
    if (classes_.empty())
        fail(RecognitionError::ColorSamplesMissing, "颜色模型");
    const cv::Mat &yuv = image.planes();
    if (mask.type() != CV_8UC1 || mask.size() != yuv.size())
        fail(CommonToolError::InputOutOfRange, "检测区域掩码与图像尺寸不一致");

    std::vector<std::size_t> histogram(classes_.size(), 0);
    std::vector<double>      scoreSums(classes_.size(), 0.0);
    std::size_t              total    = 0;
    std::size_t              accepted = 0;
    const auto              *lut      = uvLookup_.data();

    for (int row = 0; row < yuv.rows; ++row)
    {
        const auto *pixels = yuv.ptr<cv::Vec3b>(row);
        const auto *valid  = mask.ptr<std::uint8_t>(row);
        for (int column = 0; column < yuv.cols; ++column)
        {
            if (valid[column] == 0)
                continue;
            ++total;
            const auto &entry = lut[(static_cast<std::size_t>(pixels[column][1]) << 8) | pixels[column][2]];
            if (entry.score < minScore)
                continue;
            ++accepted;
            histogram[entry.classIndex] += 1;
            scoreSums[entry.classIndex] += entry.score;
        }
    }
    return finalizeClassification(histogram, scoreSums, total, accepted, minRatio);
}

ColorModel::Classification ColorModel::classifyCircle(const cv::Mat &bgr, cv::Point2d center, double radius,
                                                      double minScore, double minRatio, cv::Mat &yuvScratch) const
{
    if (classes_.empty())
        fail(RecognitionError::ColorSamplesMissing, "颜色模型");
    if (bgr.type() != CV_8UC3)
        fail(RecognitionError::UnsupportedColorImage, "图像");
    if (!(radius > 0.0) || !std::isfinite(radius) || !std::isfinite(center.x) || !std::isfinite(center.y))
        return {};

    constexpr double kBoundaryTolerance = 0.5;
    const double     radiusLimit        = radius + kBoundaryTolerance;
    const double     radiusLimitSq      = radiusLimit * radiusLimit;
    const int        x0 = std::clamp(static_cast<int>(std::floor(center.x - radiusLimit)), 0, bgr.cols);
    const int        y0 = std::clamp(static_cast<int>(std::floor(center.y - radiusLimit)), 0, bgr.rows);
    const int        x1 = std::clamp(static_cast<int>(std::ceil(center.x + radiusLimit)) + 1, 0, bgr.cols);
    const int        y1 = std::clamp(static_cast<int>(std::ceil(center.y + radiusLimit)) + 1, 0, bgr.rows);
    if (x1 <= x0 || y1 <= y0)
        return {};

    cv::cvtColor(bgr(cv::Rect(x0, y0, x1 - x0, y1 - y0)), yuvScratch, cv::COLOR_BGR2YUV);

    std::vector<std::size_t> histogram(classes_.size(), 0);
    std::vector<double>      scoreSums(classes_.size(), 0.0);
    std::size_t              total    = 0;
    std::size_t              accepted = 0;
    const auto              *lut      = uvLookup_.data();

    for (int y = y0; y < y1; ++y)
    {
        const double dy  = static_cast<double>(y) - center.y;
        const double dy2 = dy * dy;
        if (dy2 > radiusLimitSq)
            continue;
        const double dxMax = std::sqrt(radiusLimitSq - dy2);
        const int    lx0   = std::max(x0, static_cast<int>(std::ceil(center.x - dxMax)) - 1);
        const int    lx1   = std::min(x1 - 1, static_cast<int>(std::floor(center.x + dxMax)) + 1);
        const auto  *row   = yuvScratch.ptr<cv::Vec3b>(y - y0) - x0;
        for (int x = lx0; x <= lx1; ++x)
        {
            const double dx = static_cast<double>(x) - center.x;
            if (dx * dx + dy2 > radiusLimitSq)
                continue;
            ++total;
            const auto &px    = row[x];
            const auto &entry = lut[(static_cast<std::size_t>(px[1]) << 8) | px[2]];
            if (entry.score < minScore)
                continue;
            ++accepted;
            histogram[entry.classIndex] += 1;
            scoreSums[entry.classIndex] += entry.score;
        }
    }
    return finalizeClassification(histogram, scoreSums, total, accepted, minRatio);
}

} // namespace visiontools::recognition
