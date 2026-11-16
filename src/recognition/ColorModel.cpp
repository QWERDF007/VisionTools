#include "ColorModel.h"

#include "common/RoiSupport.h"
#include "common/errors/ToolError.h"

#include "errors/RecognitionErrors.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
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

struct Accumulator
{
    std::size_t pixels = 0;
    double      u      = 0;
    double      v      = 0;
    double      uu     = 0;
    double      uv     = 0;
    double      vv     = 0;
};

std::string utf8(const std::filesystem::path &path)
{
    const auto text = path.u8string();
    return {reinterpret_cast<const char *>(text.data()), text.size()};
}

// cv::imread takes a narrow path; reading the bytes first keeps Unicode directories working.
cv::Mat loadColorImage(const std::filesystem::path &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        fail(RecognitionError::AssetUnreadable, utf8(path), "训练图不存在或无法打开");
    const std::vector<char> bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    if (bytes.empty())
        fail(RecognitionError::AssetUnreadable, utf8(path), "训练图为空");
    const cv::Mat image = cv::imdecode(bytes, cv::IMREAD_COLOR);
    if (image.empty())
        fail(RecognitionError::AssetUnreadable, utf8(path), "训练图无法解码");
    return image;
}

cv::Mat regionMask(const ColorRegion &region, cv::Size size)
{
    const auto record = visionrt::makeRectangle(region.center.x - region.width / 2.0,
                                                region.center.y - region.height / 2.0,
                                                region.width, region.height, region.angle);
    return roi::regionMask({record}, size, geometry::Alignment::identity());
}

} // namespace

ChromaImage::ChromaImage(const cv::Mat &bgr)
{
    if (bgr.type() != CV_8UC3)
        fail(RecognitionError::UnsupportedColorImage, "图像");
    cv::cvtColor(bgr, planes_, cv::COLOR_BGR2YUV);
}

ColorModel ColorModel::train(const ColorTraining &training)
{
    std::map<std::string, Accumulator> accumulators;
    std::map<std::filesystem::path, cv::Mat>     chromaImages;

    for (const auto &sample : training.samples)
    {
        const auto &key    = sample.image;
        auto        cached = chromaImages.find(key);
        if (cached == chromaImages.end())
        {
            // Each training image is decoded and converted once even when several records share it.
            auto inserted = chromaImages.emplace(key, ChromaImage(loadColorImage(sample.image)).planes());
            cached        = inserted.first;
        }
        const cv::Mat &yuv = cached->second;

        auto &accumulator = accumulators[sample.label];
        for (const auto &region : sample.regions)
        {
            const cv::Mat mask = regionMask(region, yuv.size());
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
    return model;
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
    std::vector<double>      scoreSums(classes_.size(), 0);
    std::size_t              total    = 0;
    std::size_t              accepted = 0;

    for (int row = 0; row < yuv.rows; ++row)
    {
        const auto *pixels = yuv.ptr<cv::Vec3b>(row);
        const auto *valid  = mask.ptr<std::uint8_t>(row);
        for (int column = 0; column < yuv.cols; ++column)
        {
            if (valid[column] == 0)
                continue;
            ++total;
            const double u = pixels[column][1];
            const double v = pixels[column][2];
            std::size_t  best         = 0;
            double       bestDensity  = -std::numeric_limits<double>::infinity();
            double       bestDistance = 0;
            for (std::size_t index = 0; index < classes_.size(); ++index)
            {
                const auto &color = classes_[index];
                const double du    = u - color.mean[0];
                const double dv    = v - color.mean[1];
                const double distance = du * (color.inverse(0, 0) * du + color.inverse(0, 1) * dv)
                                        + dv * (color.inverse(1, 0) * du + color.inverse(1, 1) * dv);
                const double density = -0.5 * distance - 0.5 * color.logDeterminant;
                if (density > bestDensity)
                {
                    bestDensity  = density;
                    bestDistance = distance;
                    best         = index;
                }
            }
            const double score = std::exp(-0.5 * std::max(0.0, bestDistance));
            if (score < minScore)
                continue;
            ++accepted;
            histogram[best] += 1;
            scoreSums[best] += score;
        }
    }

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

} // namespace visiontools::recognition
