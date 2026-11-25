#include "TemplateMatcher.h"

#include "common/errors/ToolError.h"
#include "errors/RecognitionErrors.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <string>

namespace visiontools::recognition {
namespace {

constexpr char kMagic[4] = {'V', 'T', 'M', '2'};
constexpr std::uint32_t kFormatVersion = 2;
constexpr double kDegToRad = std::numbers::pi / 180.0;

void writeU32(std::vector<std::uint8_t> &out, std::uint32_t v)
{
    for (int i = 0; i < 4; ++i)
        out.push_back(static_cast<std::uint8_t>((v >> (i * 8)) & 0xFFu));
}

void writeI32(std::vector<std::uint8_t> &out, std::int32_t v)
{
    writeU32(out, static_cast<std::uint32_t>(v));
}

void writeF64(std::vector<std::uint8_t> &out, double v)
{
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(v));
    std::memcpy(&bits, &v, sizeof(bits));
    for (int i = 0; i < 8; ++i)
        out.push_back(static_cast<std::uint8_t>((bits >> (i * 8)) & 0xFFu));
}

void writeBlob(std::vector<std::uint8_t> &out, const std::vector<std::uint8_t> &blob)
{
    writeU32(out, static_cast<std::uint32_t>(blob.size()));
    out.insert(out.end(), blob.begin(), blob.end());
}

struct Reader
{
    std::span<const std::uint8_t> data;
    std::size_t pos = 0;

    void require(std::size_t n) const
    {
        if (pos + n > data.size())
            fail(RecognitionError::MatchModelIncompatible, "匹配模型字节不完整");
    }

    std::uint8_t u8()
    {
        require(1);
        return data[pos++];
    }

    std::uint32_t u32()
    {
        require(4);
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i)
            v |= static_cast<std::uint32_t>(data[pos++]) << (i * 8);
        return v;
    }

    std::int32_t i32() { return static_cast<std::int32_t>(u32()); }

    double f64()
    {
        require(8);
        std::uint64_t bits = 0;
        for (int i = 0; i < 8; ++i)
            bits |= static_cast<std::uint64_t>(data[pos++]) << (i * 8);
        double v = 0.0;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }

    std::vector<std::uint8_t> blob()
    {
        const std::uint32_t len = u32();
        require(len);
        std::vector<std::uint8_t> out(data.begin() + static_cast<std::ptrdiff_t>(pos),
                                      data.begin() + static_cast<std::ptrdiff_t>(pos + len));
        pos += len;
        return out;
    }
};

cv::Mat ensureGray8U(const cv::Mat &input)
{
    if (input.empty())
        fail(CommonToolError::UnsupportedImage, "模板匹配收到空图像");
    cv::Mat gray;
    if (input.channels() == 1)
        gray = input;
    else if (input.channels() == 3)
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else if (input.channels() == 4)
        cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
    else
        fail(CommonToolError::UnsupportedImage, "模板匹配仅支持 1/3/4 通道图像");
    if (gray.depth() != CV_8U)
        fail(CommonToolError::UnsupportedImage, "模板匹配仅支持 8 位图像");
    return gray;
}

cv::Mat rotatePatchAndMask(const cv::Mat &src, const cv::Mat &mask, double angleCcwDeg, cv::Mat &rotatedMask)
{
    if (std::abs(angleCcwDeg) < 1e-6)
    {
        rotatedMask = mask;
        return src;
    }
    const cv::Point2f center((src.cols - 1) * 0.5f, (src.rows - 1) * 0.5f);
    cv::Mat rot = cv::getRotationMatrix2D(center, angleCcwDeg, 1.0);
    const double rad = std::abs(angleCcwDeg) * kDegToRad;
    const double c = std::cos(rad);
    const double s = std::sin(rad);
    const int newW = std::max(src.cols, static_cast<int>(std::ceil(src.cols * c + src.rows * s)));
    const int newH = std::max(src.rows, static_cast<int>(std::ceil(src.cols * s + src.rows * c)));
    rot.at<double>(0, 2) += (newW - 1) * 0.5 - center.x;
    rot.at<double>(1, 2) += (newH - 1) * 0.5 - center.y;

    cv::Mat rotated;
    cv::warpAffine(src, rotated, rot, cv::Size(newW, newH), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    cv::warpAffine(mask, rotatedMask, rot, cv::Size(newW, newH), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
    return rotated;
}

bool maskIsFull(const cv::Mat &mask)
{
    return mask.empty() || cv::countNonZero(mask) == mask.cols * mask.rows;
}

cv::Mat computeNccMap(const cv::Mat &search, const cv::Mat &templ, const cv::Mat &mask, bool ignorePolarity)
{
    cv::Mat result;
    if (maskIsFull(mask))
        cv::matchTemplate(search, templ, result, cv::TM_CCOEFF_NORMED);
    else
        cv::matchTemplate(search, templ, result, cv::TM_CCOEFF_NORMED, mask);

    for (int y = 0; y < result.rows; ++y)
    {
        float *row = result.ptr<float>(y);
        for (int x = 0; x < result.cols; ++x)
        {
            if (!std::isfinite(row[x]))
                row[x] = -1.0f;
            else if (ignorePolarity)
                row[x] = std::abs(row[x]);
        }
    }
    return result;
}

// Unit-normalized edge orientation field: (gx, gy) for use_polarity, (cos 2theta, sin 2theta) for ignore_polarity.
struct GradientField
{
    cv::Mat u;
    cv::Mat v;
    int edgeCount = 0;
};

GradientField buildGradientField(const cv::Mat &gray, const cv::Mat &mask, double contrast, bool ignorePolarity)
{
    cv::Mat blurred, gx, gy;
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0.8);
    cv::Sobel(blurred, gx, CV_32F, 1, 0, 3);
    cv::Sobel(blurred, gy, CV_32F, 0, 1, 3);

    // Sobel 3x3 magnitude for a step of size `contrast` is ~4 * contrast.
    const float magThresh = static_cast<float>(std::max(1.0, contrast * 3.0));
    const float magThreshSq = magThresh * magThresh;

    GradientField field;
    field.u = cv::Mat::zeros(gray.size(), CV_32F);
    field.v = cv::Mat::zeros(gray.size(), CV_32F);
    const bool hasMask = !mask.empty();

    for (int y = 1; y + 1 < gray.rows; ++y)
    {
        const float *rx = gx.ptr<float>(y);
        const float *ry = gy.ptr<float>(y);
        const std::uint8_t *rm = hasMask ? mask.ptr<std::uint8_t>(y) : nullptr;
        float *ru = field.u.ptr<float>(y);
        float *rv = field.v.ptr<float>(y);
        for (int x = 1; x + 1 < gray.cols; ++x)
        {
            if (rm && rm[x] == 0)
                continue;
            const float dx = rx[x];
            const float dy = ry[x];
            const float m2 = dx * dx + dy * dy;
            if (m2 < magThreshSq)
                continue;
            const float invM = 1.0f / std::sqrt(m2);
            const float ux = dx * invM;
            const float uy = dy * invM;
            if (ignorePolarity)
            {
                ru[x] = ux * ux - uy * uy;
                rv[x] = 2.0f * ux * uy;
            }
            else
            {
                ru[x] = ux;
                rv[x] = uy;
            }
            ++field.edgeCount;
        }
    }
    return field;
}

cv::Mat computeShapeScoreMap(const GradientField &search, const GradientField &templ, bool ignorePolarity)
{
    if (templ.edgeCount <= 0 || templ.u.cols > search.u.cols || templ.u.rows > search.u.rows)
        return {};
    cv::Mat corrU, corrV;
    cv::matchTemplate(search.u, templ.u, corrU, cv::TM_CCORR);
    cv::matchTemplate(search.v, templ.v, corrV, cv::TM_CCORR);
    cv::Mat score = (corrU + corrV) * (1.0f / static_cast<float>(templ.edgeCount));
    if (ignorePolarity)
        score = (score + 1.0f) * 0.5f;
    return score;
}

double parabolicOffset(double yPrev, double yCurr, double yNext)
{
    const double denom = yPrev - 2.0 * yCurr + yNext;
    if (std::abs(denom) < 1e-9 || denom >= 0.0)
        return 0.0;
    return std::clamp(0.5 * (yPrev - yNext) / denom, -0.5, 0.5);
}

struct RefinedPeak
{
    cv::Point2d center;
    double angleDeg = 0.0;
    double score = -1.0;
    bool valid = false;
};

RefinedPeak evaluateAngleLocal(const cv::Mat &gray,
                               const cv::Mat &grayTemplate,
                               const cv::Mat &mask,
                               cv::Point2d approxCenter,
                               double angleCcwDeg,
                               int margin,
                               bool ignorePolarity)
{
    cv::Mat rotMask;
    const cv::Mat rotTempl = rotatePatchAndMask(grayTemplate, mask, angleCcwDeg, rotMask);
    const int x0 = std::max(0, static_cast<int>(std::floor(approxCenter.x - rotTempl.cols * 0.5)) - margin);
    const int y0 = std::max(0, static_cast<int>(std::floor(approxCenter.y - rotTempl.rows * 0.5)) - margin);
    const int x1 = std::min(gray.cols, static_cast<int>(std::ceil(approxCenter.x + rotTempl.cols * 0.5)) + margin);
    const int y1 = std::min(gray.rows, static_cast<int>(std::ceil(approxCenter.y + rotTempl.rows * 0.5)) + margin);
    if (x1 - x0 < rotTempl.cols || y1 - y0 < rotTempl.rows)
        return {};

    const cv::Mat patch = gray(cv::Rect(x0, y0, x1 - x0, y1 - y0));
    const cv::Mat scoreMap = computeNccMap(patch, rotTempl, rotMask, ignorePolarity);
    double maxVal = -1.0;
    cv::Point maxLoc;
    cv::minMaxLoc(scoreMap, nullptr, &maxVal, nullptr, &maxLoc);
    if (!std::isfinite(maxVal))
        return {};

    double dx = 0.0;
    double dy = 0.0;
    if (maxLoc.x > 0 && maxLoc.x + 1 < scoreMap.cols)
        dx = parabolicOffset(scoreMap.at<float>(maxLoc.y, maxLoc.x - 1),
                             scoreMap.at<float>(maxLoc.y, maxLoc.x),
                             scoreMap.at<float>(maxLoc.y, maxLoc.x + 1));
    if (maxLoc.y > 0 && maxLoc.y + 1 < scoreMap.rows)
        dy = parabolicOffset(scoreMap.at<float>(maxLoc.y - 1, maxLoc.x),
                             scoreMap.at<float>(maxLoc.y, maxLoc.x),
                             scoreMap.at<float>(maxLoc.y + 1, maxLoc.x));

    const double cx = x0 + maxLoc.x + dx + (rotTempl.cols - 1) * 0.5;
    const double cy = y0 + maxLoc.y + dy + (rotTempl.rows - 1) * 0.5;
    return {{cx, cy}, angleCcwDeg, maxVal, true};
}

RefinedPeak refineCandidate(const cv::Mat &gray,
                            const cv::Mat &grayTemplate,
                            const cv::Mat &mask,
                            cv::Point2d coarseCenter,
                            double coarseAngleDeg,
                            double angleCenterDeg,
                            double angleTolDeg,
                            double coarseStepDeg,
                            int initialMargin,
                            bool ignorePolarity)
{
    const double minAngle = angleCenterDeg - angleTolDeg;
    const double maxAngle = angleCenterDeg + angleTolDeg;
    auto clampAngle = [&](double a) { return std::clamp(a, minAngle, maxAngle); };

    RefinedPeak best = evaluateAngleLocal(gray, grayTemplate, mask, coarseCenter,
                                          clampAngle(coarseAngleDeg), initialMargin, ignorePolarity);
    if (!best.valid)
        return {};

    if (angleTolDeg <= 1e-6)
        return best;

    const double step1 = std::max(0.25, std::min(0.5, coarseStepDeg * 0.5));
    const double span1 = std::max(step1 * 2.0, std::min(angleTolDeg, coarseStepDeg * 1.5));
    for (double da = -span1; da <= span1 + 1e-6; da += step1)
    {
        const double a = clampAngle(coarseAngleDeg + da);
        const auto trial = evaluateAngleLocal(gray, grayTemplate, mask, best.center, a, initialMargin, ignorePolarity);
        if (trial.valid && trial.score > best.score)
            best = trial;
    }

    const double step2 = 0.15;
    const double baseAngle = best.angleDeg;
    const cv::Point2d baseCenter = best.center;
    const auto pMinus = evaluateAngleLocal(gray, grayTemplate, mask, baseCenter, clampAngle(baseAngle - step2), 3, ignorePolarity);
    const auto pZero = evaluateAngleLocal(gray, grayTemplate, mask, baseCenter, baseAngle, 3, ignorePolarity);
    const auto pPlus = evaluateAngleLocal(gray, grayTemplate, mask, baseCenter, clampAngle(baseAngle + step2), 3, ignorePolarity);
    if (pZero.valid)
        best = pZero;
    if (pMinus.valid && pMinus.score > best.score)
        best = pMinus;
    if (pPlus.valid && pPlus.score > best.score)
        best = pPlus;

    if (pMinus.valid && pZero.valid && pPlus.valid && pZero.score >= pMinus.score && pZero.score >= pPlus.score
        && std::abs((baseAngle - step2) - clampAngle(baseAngle - step2)) < 1e-6
        && std::abs((baseAngle + step2) - clampAngle(baseAngle + step2)) < 1e-6)
    {
        const double offset = parabolicOffset(pMinus.score, pZero.score, pPlus.score);
        const double interpAngle = clampAngle(baseAngle + offset * step2);
        const auto interp = evaluateAngleLocal(gray, grayTemplate, mask, baseCenter, interpAngle, 3, ignorePolarity);
        if (interp.valid && interp.score >= best.score)
            best = interp;
    }
    return best;
}

struct CoarseCandidate
{
    cv::Point2d center;
    double angleDeg = 0.0;
    double score = 0.0;
};

std::vector<double> buildAngleList(double centerDeg, double tolDeg, double stepDeg)
{
    std::vector<double> angles;
    if (tolDeg <= 1e-6)
    {
        angles.push_back(centerDeg);
        return angles;
    }
    const double step = std::max(0.1, stepDeg);
    const int steps = std::max(1, static_cast<int>(std::ceil(tolDeg / step)));
    const double actualStep = tolDeg / steps;
    angles.reserve(static_cast<std::size_t>(2 * steps + 1));
    for (int i = -steps; i <= steps; ++i)
        angles.push_back(centerDeg + i * actualStep);
    return angles;
}

std::vector<CoarseCandidate> collectLocalPeaks(const cv::Mat &scoreMap,
                                               double angleDeg,
                                               double threshold,
                                               cv::Point2d originOffset,
                                               cv::Size templSize,
                                               double scaleFactor)
{
    std::vector<CoarseCandidate> peaks;
    const double halfW = (templSize.width - 1) * 0.5;
    const double halfH = (templSize.height - 1) * 0.5;

    for (int y = 0; y < scoreMap.rows; ++y)
    {
        const float *row = scoreMap.ptr<float>(y);
        for (int x = 0; x < scoreMap.cols; ++x)
        {
            const float val = row[x];
            if (!(val >= threshold))
                continue;
            bool isMax = true;
            const int y0 = std::max(0, y - 1);
            const int y1 = std::min(scoreMap.rows - 1, y + 1);
            const int x0 = std::max(0, x - 1);
            const int x1 = std::min(scoreMap.cols - 1, x + 1);
            for (int ny = y0; ny <= y1 && isMax; ++ny)
            {
                const float *nrow = scoreMap.ptr<float>(ny);
                for (int nx = x0; nx <= x1; ++nx)
                {
                    if ((ny != y || nx != x) && nrow[nx] > val)
                    {
                        isMax = false;
                        break;
                    }
                }
            }
            if (!isMax)
                continue;

            double dx = 0.0;
            double dy = 0.0;
            if (x > 0 && x + 1 < scoreMap.cols)
                dx = parabolicOffset(row[x - 1], val, row[x + 1]);
            if (y > 0 && y + 1 < scoreMap.rows)
                dy = parabolicOffset(scoreMap.at<float>(y - 1, x), val, scoreMap.at<float>(y + 1, x));

            peaks.push_back({{originOffset.x + (x + dx + halfW) * scaleFactor,
                              originOffset.y + (y + dy + halfH) * scaleFactor},
                             angleDeg,
                             static_cast<double>(val)});
        }
    }
    return peaks;
}

template<class T>
void applySpatialNms(std::vector<T> &items, double minDist, int maxResults)
{
    std::sort(items.begin(), items.end(), [](const T &a, const T &b) { return a.score > b.score; });
    std::vector<T> kept;
    const double minDistSq = minDist * minDist;
    for (const auto &item : items)
    {
        bool overlaps = false;
        for (const auto &existing : kept)
        {
            const cv::Point2d d = item.center - existing.center;
            if (d.dot(d) < minDistSq)
            {
                overlaps = true;
                break;
            }
        }
        if (overlaps)
            continue;
        kept.push_back(item);
        if (maxResults > 0 && static_cast<int>(kept.size()) >= maxResults)
            break;
    }
    items = std::move(kept);
}

int pyramidLevelsFor(cv::Size templSize, cv::Size searchSize)
{
    int levels = 0;
    int tw = templSize.width;
    int th = templSize.height;
    int sw = searchSize.width;
    int sh = searchSize.height;
    while (levels < 2 && std::min(tw, th) >= 48 && std::max(tw, th) >= 120 && sw >= tw + 24 && sh >= th + 24)
    {
        tw /= 2;
        th /= 2;
        sw /= 2;
        sh /= 2;
        ++levels;
    }
    return levels;
}

cv::Mat downsampleGray(const cv::Mat &src, int levels)
{
    cv::Mat cur = src;
    for (int i = 0; i < levels; ++i)
        cv::pyrDown(cur, cur);
    return cur;
}

cv::Mat downsampleMask(const cv::Mat &src, cv::Size targetSize)
{
    if (src.size() == targetSize)
        return src;
    cv::Mat dst;
    cv::resize(src, dst, targetSize, 0.0, 0.0, cv::INTER_NEAREST);
    return dst;
}

} // namespace

MatchingMethod parseMatchingMethod(std::string_view text)
{
    if (text == "shape" || text == "shape-based")
        return MatchingMethod::Shape;
    if (text == "correlation" || text == "correlation-based")
        return MatchingMethod::Correlation;
    fail(CommonToolError::InputOutOfRange, "匹配方法只接受 shape 或 correlation");
}

struct TemplateModel::Impl
{
    TemplateTrainOptions options;
    cv::Point2d refCenter{0.0, 0.0};
    cv::Size2d refSize{0.0, 0.0};
    double refAngleDeg = 0.0;
    bool isCircle = false;
    double radius = 0.0;
    cv::Mat grayTemplate;
    cv::Mat mask;
};

TemplateModel::TemplateModel() = default;
TemplateModel::~TemplateModel() = default;
TemplateModel::TemplateModel(TemplateModel &&) noexcept = default;
TemplateModel &TemplateModel::operator=(TemplateModel &&) noexcept = default;
TemplateModel::TemplateModel(const TemplateModel &) = default;
TemplateModel &TemplateModel::operator=(const TemplateModel &) = default;

bool TemplateModel::empty() const noexcept { return !impl_ || impl_->grayTemplate.empty(); }
MatchingMethod TemplateModel::method() const noexcept { return impl_ ? impl_->options.method : MatchingMethod::Shape; }
cv::Point2d TemplateModel::referenceCenter() const noexcept { return impl_ ? impl_->refCenter : cv::Point2d{}; }
cv::Size2d TemplateModel::referenceSize() const noexcept { return impl_ ? impl_->refSize : cv::Size2d{}; }
double TemplateModel::referenceAngleDeg() const noexcept { return impl_ ? impl_->refAngleDeg : 0.0; }
bool TemplateModel::isCircle() const noexcept { return impl_ && impl_->isCircle; }
double TemplateModel::radius() const noexcept { return impl_ ? impl_->radius : 0.0; }
const TemplateTrainOptions &TemplateModel::trainOptions() const noexcept
{
    static const TemplateTrainOptions kDefault{};
    return impl_ ? impl_->options : kDefault;
}

geometry::Alignment TemplateModel::alignmentFor(cv::Point2d currentCenter, double angleCcwDeg) const noexcept
{
    if (!impl_)
        return geometry::Alignment::identity();
    const double theta = -angleCcwDeg * kDegToRad;
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const cv::Point2d r = impl_->refCenter;
    return geometry::Alignment{{
        c, -s, currentCenter.x - (c * r.x - s * r.y),
        s,  c, currentCenter.y - (s * r.x + c * r.y)
    }};
}

TemplateModel TemplateModel::trainRectangle(const cv::Mat &image,
                                            cv::Point2d center,
                                            double width,
                                            double height,
                                            double angleDeg,
                                            const TemplateTrainOptions &options)
{
    const cv::Mat gray = ensureGray8U(image);
    if (!std::isfinite(width) || !std::isfinite(height) || width < 5.0 || height < 5.0)
        fail(RecognitionError::MatchTrainingInvalid, "训练模板矩形尺寸必须至少为 5x5 像素");

    const int patchW = std::max(5, static_cast<int>(std::round(width)));
    const int patchH = std::max(5, static_cast<int>(std::round(height)));
    if (patchW >= gray.cols || patchH >= gray.rows)
        fail(RecognitionError::MatchTrainingInvalid, "训练模板区域不能大于或等于整张图像");

    const double rad = angleDeg * kDegToRad;
    const double c = std::cos(rad);
    const double s = std::sin(rad);
    const double halfW = (patchW - 1) * 0.5;
    const double halfH = (patchH - 1) * 0.5;
    cv::Mat mapX(patchH, patchW, CV_32FC1);
    cv::Mat mapY(patchH, patchW, CV_32FC1);
    for (int py = 0; py < patchH; ++py)
    {
        float *rx = mapX.ptr<float>(py);
        float *ry = mapY.ptr<float>(py);
        const double dy = py - halfH;
        for (int px = 0; px < patchW; ++px)
        {
            const double dx = px - halfW;
            rx[px] = static_cast<float>(center.x + c * dx - s * dy);
            ry[px] = static_cast<float>(center.y + s * dx + c * dy);
        }
    }
    cv::Mat patch;
    cv::remap(gray, patch, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    cv::Mat mask(patchH, patchW, CV_8UC1, cv::Scalar(255));
    if (options.method == MatchingMethod::Shape)
    {
        const auto checkField = buildGradientField(patch, mask, options.modelContrast, options.ignorePolarity);
        if (checkField.edgeCount < 8)
            fail(RecognitionError::MatchTrainingInvalid, "模板区域边缘特征不足，无法建立形状匹配模型");
    }

    auto impl = std::make_shared<Impl>();
    impl->options = options;
    impl->refCenter = center;
    impl->refSize = {width, height};
    impl->refAngleDeg = angleDeg;
    impl->isCircle = false;
    impl->radius = 0.0;
    impl->grayTemplate = std::move(patch);
    impl->mask = std::move(mask);

    TemplateModel model;
    model.impl_ = std::move(impl);
    return model;
}

TemplateModel TemplateModel::trainCircle(const cv::Mat &image,
                                         cv::Point2d center,
                                         double radius,
                                         const TemplateTrainOptions &options)
{
    const cv::Mat gray = ensureGray8U(image);
    if (!std::isfinite(radius) || radius < 3.0)
        fail(RecognitionError::MatchTrainingInvalid, "训练模板圆半径必须至少为 3 像素");

    const int half = std::max(3, static_cast<int>(std::ceil(radius)));
    const int diameter = 2 * half + 1;
    if (diameter >= gray.cols || diameter >= gray.rows)
        fail(RecognitionError::MatchTrainingInvalid, "训练模板圆超出图像尺寸");

    cv::Mat patch;
    cv::getRectSubPix(gray, cv::Size(diameter, diameter),
                      cv::Point2f(static_cast<float>(center.x), static_cast<float>(center.y)), patch);
    cv::Mat mask(diameter, diameter, CV_8UC1, cv::Scalar(0));
    const double r2 = radius * radius;
    for (int y = 0; y < diameter; ++y)
    {
        auto *row = mask.ptr<std::uint8_t>(y);
        const double dy = y - half;
        for (int x = 0; x < diameter; ++x)
        {
            const double dx = x - half;
            if (dx * dx + dy * dy <= r2)
                row[x] = 255;
        }
    }

    if (options.method == MatchingMethod::Shape)
    {
        const auto checkField = buildGradientField(patch, mask, options.modelContrast, options.ignorePolarity);
        if (checkField.edgeCount < 8)
            fail(RecognitionError::MatchTrainingInvalid, "模板圆区域边缘特征不足，无法建立形状匹配模型");
    }

    auto impl = std::make_shared<Impl>();
    impl->options = options;
    impl->refCenter = center;
    impl->refSize = {radius * 2.0, radius * 2.0};
    impl->refAngleDeg = 0.0;
    impl->isCircle = true;
    impl->radius = radius;
    impl->grayTemplate = std::move(patch);
    impl->mask = std::move(mask);

    TemplateModel model;
    model.impl_ = std::move(impl);
    return model;
}

std::vector<TemplateHit> TemplateModel::match(const cv::Mat &image,
                                              const cv::Rect &searchRoi,
                                              const TemplateMatchOptions &options) const
{
    if (empty())
        fail(RecognitionError::MatchModelPathMissing, "匹配模型未加载");

    const cv::Mat gray = ensureGray8U(image);
    const cv::Rect fullRect(0, 0, gray.cols, gray.rows);
    const cv::Rect roi = searchRoi.empty() ? fullRect : (searchRoi & fullRect);
    if (roi.width < impl_->grayTemplate.cols || roi.height < impl_->grayTemplate.rows)
        return {};

    const cv::Mat searchPatch = gray(roi);
    const double minScore = std::clamp(options.minScore, 0.0, 1.0);
    const double maxOverlap = std::clamp(options.maxOverlap, 0.0, 0.99);
    const double minCenterDist = impl_->isCircle
        ? std::max(2.0, 2.0 * impl_->radius * (1.0 - maxOverlap))
        : std::max(2.0, std::min(impl_->refSize.width, impl_->refSize.height) * (1.0 - maxOverlap));

    const int pyrLevels = pyramidLevelsFor(impl_->grayTemplate.size(), searchPatch.size());
    const double scaleFactor = static_cast<double>(1 << pyrLevels);
    const cv::Mat sImg = downsampleGray(searchPatch, pyrLevels);
    const cv::Mat tImg = downsampleGray(impl_->grayTemplate, pyrLevels);
    const cv::Mat tMask = downsampleMask(impl_->mask, tImg.size());

    const double coarseStep = std::max(0.5, impl_->options.angleStepDeg * (pyrLevels >= 2 ? 1.5 : 1.0));
    const auto angles = buildAngleList(options.angleCenterDeg, options.angleToleranceDeg, coarseStep);
    std::vector<CoarseCandidate> coarse;

    if (impl_->options.method == MatchingMethod::Shape)
    {
        const auto searchField = buildGradientField(sImg, cv::Mat(), impl_->options.searchContrast, impl_->options.ignorePolarity);
        const double coarseThresh = std::max(0.20, minScore * 0.45);
        for (double a : angles)
        {
            cv::Mat rotMask;
            const cv::Mat rotTempl = rotatePatchAndMask(tImg, tMask, a, rotMask);
            if (rotTempl.cols > sImg.cols || rotTempl.rows > sImg.rows)
                continue;
            const auto templField = buildGradientField(rotTempl, rotMask, impl_->options.modelContrast, impl_->options.ignorePolarity);
            if (templField.edgeCount < 8)
                continue;
            const cv::Mat scoreMap = computeShapeScoreMap(searchField, templField, impl_->options.ignorePolarity);
            if (scoreMap.empty())
                continue;
            auto peaks = collectLocalPeaks(scoreMap, a, coarseThresh,
                                           {static_cast<double>(roi.x), static_cast<double>(roi.y)},
                                           rotTempl.size(), scaleFactor);
            coarse.insert(coarse.end(), peaks.begin(), peaks.end());
        }
    }
    else
    {
        const double coarseThresh = (pyrLevels > 0) ? std::max(0.25, minScore * 0.80) : minScore;
        for (double a : angles)
        {
            cv::Mat rotMask;
            const cv::Mat rotTempl = rotatePatchAndMask(tImg, tMask, a, rotMask);
            if (rotTempl.cols > sImg.cols || rotTempl.rows > sImg.rows)
                continue;
            const cv::Mat scoreMap = computeNccMap(sImg, rotTempl, rotMask, impl_->options.ignorePolarity);
            auto peaks = collectLocalPeaks(scoreMap, a, coarseThresh,
                                           {static_cast<double>(roi.x), static_cast<double>(roi.y)},
                                           rotTempl.size(), scaleFactor);
            coarse.insert(coarse.end(), peaks.begin(), peaks.end());
        }
    }

    if (coarse.empty())
        return {};

    const int maxRefineCandidates = options.maxResults > 0 ? std::max(6, options.maxResults * 4) : 0;
    applySpatialNms(coarse, minCenterDist * 0.75, maxRefineCandidates);

    std::vector<TemplateHit> hits;
    hits.reserve(coarse.size());
    const int margin = std::max(4, static_cast<int>(scaleFactor) * 3);
    for (const auto &cand : coarse)
    {
        if (options.refineSubpixel)
        {
            const auto refined = refineCandidate(gray, impl_->grayTemplate, impl_->mask,
                                                 cand.center, cand.angleDeg,
                                                 options.angleCenterDeg, options.angleToleranceDeg,
                                                 coarseStep, margin, impl_->options.ignorePolarity);
            if (!refined.valid || refined.score < minScore)
                continue;
            hits.push_back({refined.center, refined.angleDeg, 1.0, refined.score,
                            alignmentFor(refined.center, refined.angleDeg)});
        }
        else
        {
            if (cand.score < minScore)
                continue;
            hits.push_back({cand.center, cand.angleDeg, 1.0, cand.score,
                            alignmentFor(cand.center, cand.angleDeg)});
        }
    }

    applySpatialNms(hits, minCenterDist, options.maxResults);
    return hits;
}

std::vector<std::uint8_t> TemplateModel::encode() const
{
    if (empty())
        fail(RecognitionError::MatchModelIncompatible, "无法序列化空匹配模型");

    std::vector<std::uint8_t> templPng;
    std::vector<std::uint8_t> maskPng;
    if (!cv::imencode(".png", impl_->grayTemplate, templPng) || !cv::imencode(".png", impl_->mask, maskPng))
        fail(RecognitionError::MatchModelIncompatible, "模板图像编码失败");

    std::vector<std::uint8_t> out;
    out.insert(out.end(), std::begin(kMagic), std::end(kMagic));
    writeU32(out, kFormatVersion);
    out.push_back(static_cast<std::uint8_t>(impl_->options.method));
    out.push_back(impl_->options.ignorePolarity ? 1u : 0u);
    out.push_back(impl_->isCircle ? 1u : 0u);
    out.push_back(0u);
    writeI32(out, impl_->options.numFeatures);
    writeF64(out, impl_->options.modelContrast);
    writeF64(out, impl_->options.searchContrast);
    writeF64(out, impl_->options.angleToleranceDeg);
    writeF64(out, impl_->options.angleStepDeg);
    writeF64(out, impl_->refCenter.x);
    writeF64(out, impl_->refCenter.y);
    writeF64(out, impl_->refSize.width);
    writeF64(out, impl_->refSize.height);
    writeF64(out, impl_->refAngleDeg);
    writeF64(out, impl_->radius);
    writeBlob(out, templPng);
    writeBlob(out, maskPng);
    return out;
}

TemplateModel TemplateModel::decode(std::span<const std::uint8_t> bytes)
{
    Reader r{bytes, 0};
    r.require(4);
    if (std::memcmp(bytes.data(), kMagic, 4) != 0)
        fail(RecognitionError::MatchModelIncompatible, "匹配模型魔数无效");
    r.pos = 4;
    if (r.u32() != kFormatVersion)
        fail(RecognitionError::MatchModelIncompatible, "匹配模型版本不受支持");

    auto impl = std::make_shared<Impl>();
    const auto methodCode = r.u8();
    if (methodCode > 1)
        fail(RecognitionError::MatchModelIncompatible, "匹配方法编码无效");
    impl->options.method = static_cast<MatchingMethod>(methodCode);
    impl->options.ignorePolarity = r.u8() != 0;
    impl->isCircle = r.u8() != 0;
    (void)r.u8();
    impl->options.numFeatures = r.i32();
    impl->options.modelContrast = r.f64();
    impl->options.searchContrast = r.f64();
    impl->options.angleToleranceDeg = r.f64();
    impl->options.angleStepDeg = r.f64();
    impl->refCenter.x = r.f64();
    impl->refCenter.y = r.f64();
    impl->refSize.width = r.f64();
    impl->refSize.height = r.f64();
    impl->refAngleDeg = r.f64();
    impl->radius = r.f64();

    const auto templPng = r.blob();
    const auto maskPng = r.blob();

    impl->grayTemplate = cv::imdecode(templPng, cv::IMREAD_GRAYSCALE);
    impl->mask = cv::imdecode(maskPng, cv::IMREAD_GRAYSCALE);
    if (impl->grayTemplate.empty() || impl->mask.empty() || impl->grayTemplate.size() != impl->mask.size())
        fail(RecognitionError::MatchModelIncompatible, "匹配模型模板图像解码失败");

    TemplateModel model;
    model.impl_ = std::move(impl);
    return model;
}

} // namespace visiontools::recognition
