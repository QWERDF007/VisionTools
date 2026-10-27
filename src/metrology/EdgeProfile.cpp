#include "EdgeProfile.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace visiontools::metrology {
namespace {
constexpr double kPi = 3.14159265358979323846;

double sampleBilinear(const cv::Mat &gray, double x, double y, bool &inside)
{
    if (!(x >= 0.0 && y >= 0.0 && x <= gray.cols - 1.0 && y <= gray.rows - 1.0))
    {
        inside = false;
        return std::numeric_limits<double>::quiet_NaN();
    }
    const int    x0 = static_cast<int>(std::floor(x));
    const int    y0 = static_cast<int>(std::floor(y));
    const int    x1 = std::min(x0 + 1, gray.cols - 1);
    const int    y1 = std::min(y0 + 1, gray.rows - 1);
    const double fx = x - x0, fy = y - y0;
    const auto & row0 = gray.ptr<std::uint8_t>(y0);
    const auto & row1 = gray.ptr<std::uint8_t>(y1);
    const double top    = row0[x0] * (1.0 - fx) + row0[x1] * fx;
    const double bottom = row1[x0] * (1.0 - fx) + row1[x1] * fx;
    inside              = true;
    return top * (1.0 - fy) + bottom * fy;
}

std::vector<double> gaussianKernel(double sigma, double step)
{
    const double sigmaSamples = sigma / step;
    const int    radius       = std::max(1, static_cast<int>(std::ceil(3.0 * sigmaSamples)));
    std::vector<double> kernel(static_cast<std::size_t>(radius) * 2 + 1);
    double              sum = 0;
    for (int i = -radius; i <= radius; ++i)
    {
        const double weight = std::exp(-0.5 * (i * i) / (sigmaSamples * sigmaSamples));
        kernel[static_cast<std::size_t>(i + radius)] = weight;
        sum += weight;
    }
    for (auto &weight : kernel)
        weight /= sum;
    return kernel;
}
} // namespace

IntensityProfile buildProfile(const cv::Mat &gray, cv::Point2d center, cv::Point2d direction, double halfLength,
                              cv::Point2d averageDirection, double averageHalfWidth, double step, std::size_t budget)
{
    if (gray.empty() || gray.type() != CV_8UC1)
        throw std::runtime_error("边缘采样需要单通道 8 位灰度图");
    if (!(step > 0.0) || !(halfLength > 0.0) || !std::isfinite(halfLength))
        throw std::runtime_error("采样步长与搜索半宽必须为正数");
    const double length = std::hypot(direction.x, direction.y);
    if (!(length > 1e-9))
        throw std::runtime_error("搜索方向不能为零向量");
    direction.x /= length;
    direction.y /= length;
    const double averageLength = std::hypot(averageDirection.x, averageDirection.y);
    if (!(averageLength > 1e-9))
        throw std::runtime_error("平均方向不能为零向量");
    averageDirection.x /= averageLength;
    averageDirection.y /= averageLength;

    const std::size_t count    = static_cast<std::size_t>(std::floor(2.0 * halfLength / step)) + 1;
    const std::size_t tangential = static_cast<std::size_t>(std::max(1.0, std::round(averageHalfWidth * 2.0 + 1.0)));
    if (count > budget || tangential > budget / std::max<std::size_t>(1, count))
        throw std::runtime_error("边缘采样预算超限，请减小搜索宽度或平均宽度");

    IntensityProfile profile;
    profile.step       = step;
    profile.halfLength = halfLength;
    profile.values.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
    {
        const double offset = -halfLength + static_cast<double>(index) * step;
        const double baseX  = center.x + direction.x * offset;
        const double baseY  = center.y + direction.y * offset;
        double       sum    = 0;
        std::size_t  used   = 0;
        for (std::size_t sample = 0; sample < tangential; ++sample)
        {
            const double tangentialOffset
                = (static_cast<double>(sample) - static_cast<double>(tangential - 1) / 2.0);
            bool  inside = false;
            const double value = sampleBilinear(gray, baseX + averageDirection.x * tangentialOffset,
                                                baseY + averageDirection.y * tangentialOffset, inside);
            if (inside)
            {
                sum += value;
                ++used;
            }
        }
        profile.values.push_back(used == tangential ? sum / static_cast<double>(used)
                                                    : std::numeric_limits<double>::quiet_NaN());
    }
    return profile;
}

std::vector<double> gradientProfile(const IntensityProfile &profile, double sigma)
{
    const std::size_t count = profile.values.size();
    std::vector<double> smoothed(count, std::numeric_limits<double>::quiet_NaN());
    if (count == 0)
        return smoothed;
    if (!(sigma > 0.0) || !std::isfinite(sigma))
        throw std::runtime_error("平滑标准差必须为正数");
    const auto   kernel = gaussianKernel(sigma, profile.step);
    const int    radius = static_cast<int>(kernel.size() / 2);
    for (std::size_t index = 0; index < count; ++index)
    {
        if (index < static_cast<std::size_t>(radius) || index + static_cast<std::size_t>(radius) >= count)
            continue;
        bool         valid = true;
        double       sum   = 0;
        for (int k = -radius; k <= radius; ++k)
        {
            const double value = profile.values[static_cast<std::ptrdiff_t>(index) + k];
            if (!std::isfinite(value))
            {
                valid = false;
                break;
            }
            sum += value * kernel[static_cast<std::size_t>(k + radius)];
        }
        if (valid)
            smoothed[index] = sum;
    }
    std::vector<double> gradient(count, std::numeric_limits<double>::quiet_NaN());
    if (count < 3)
        return gradient;
    for (std::size_t index = 1; index + 1 < count; ++index)
    {
        const double left = smoothed[index - 1], right = smoothed[index + 1];
        if (std::isfinite(left) && std::isfinite(right))
            gradient[index] = (right - left) / (2.0 * profile.step);
    }
    return gradient;
}

std::vector<EdgeCandidate> detectEdges(const IntensityProfile &profile, const ProfileConfig &config)
{
    const auto gradient = gradientProfile(profile, config.sigma);
    std::vector<EdgeCandidate> result;
    if (gradient.size() < 3)
        return result;
    const auto accepts = [&](double value)
    {
        if (!std::isfinite(value) || !(std::abs(value) > 0.0) || std::abs(value) < config.minGradient)
            return false;
        switch (config.polarity)
        {
        case EdgePolarity::DarkToLight:
            return value > 0.0;
        case EdgePolarity::LightToDark:
            return value < 0.0;
        case EdgePolarity::Any:
            return true;
        }
        return false;
    };
    for (std::size_t index = 1; index + 1 < gradient.size(); ++index)
    {
        const double previous = gradient[index - 1], current = gradient[index], next = gradient[index + 1];
        if (!accepts(current))
            continue;
        const bool maximum = current > previous && current >= next;
        const bool minimum = current < previous && current <= next;
        if (!maximum && !minimum)
            continue;
        const double denominator = previous - 2.0 * current + next;
        double       subpixel    = 0.0;
        if (std::abs(denominator) > 1e-12)
            subpixel = std::clamp(0.5 * (previous - next) / denominator, -0.5, 0.5);
        const double offset = -profile.halfLength + static_cast<double>(index) * profile.step + subpixel * profile.step;
        result.push_back({offset, current, {-1, -1}});
    }
    return result;
}

std::optional<EdgeCandidate> nearestEdge(std::vector<EdgeCandidate> candidates)
{
    if (candidates.empty())
        return std::nullopt;
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const EdgeCandidate &a, const EdgeCandidate &b)
                     {
                         if (a.offset != b.offset)
                             return std::abs(a.offset) < std::abs(b.offset);
                         return std::abs(a.gradient) > std::abs(b.gradient);
                     });
    return candidates.front();
}

cv::Mat grayscale(const cv::Mat &image)
{
    if (image.empty())
        throw std::runtime_error("图像为空");
    if (image.depth() != CV_8U)
        throw std::runtime_error("测量工具首版只支持 8 位图像");
    switch (image.channels())
    {
    case 1:
        return image;
    case 3:
    {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    case 4:
    {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
        return gray;
    }
    default:
        throw std::runtime_error("测量工具只支持 1、3 或 4 通道图像");
    }
}
} // namespace visiontools::metrology
