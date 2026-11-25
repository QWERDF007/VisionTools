#pragma once

#include "common/GeometryTransform.h"

#include <opencv2/core.hpp>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace visiontools::recognition {

enum class MatchingMethod : std::uint8_t
{
    Shape = 0,
    Correlation = 1
};

MatchingMethod parseMatchingMethod(std::string_view text);

struct TemplateTrainOptions
{
    MatchingMethod method = MatchingMethod::Shape;
    double modelContrast = 10.0;
    double searchContrast = 5.0;
    double angleToleranceDeg = 20.0;
    double angleStepDeg = 1.0;
    bool ignorePolarity = false;
    int numFeatures = 128;
};

struct TemplateMatchOptions
{
    double minScore = 0.5;
    double angleToleranceDeg = 20.0;
    double angleCenterDeg = 0.0;
    double maxOverlap = 0.3;
    int maxResults = 1;
    bool refineSubpixel = true;
};

struct TemplateHit
{
    cv::Point2d center;
    double angleDeg = 0.0; // CCW rotation in image coordinates (same convention as OpenCV / InferRT)
    double scale = 1.0;
    double score = 0.0;
    geometry::Alignment alignment = geometry::Alignment::identity();
};

class TemplateModel
{
public:
    TemplateModel();
    ~TemplateModel();
    TemplateModel(TemplateModel &&) noexcept;
    TemplateModel &operator=(TemplateModel &&) noexcept;
    TemplateModel(const TemplateModel &);
    TemplateModel &operator=(const TemplateModel &);

    static TemplateModel trainRectangle(const cv::Mat &gray,
                                        cv::Point2d center,
                                        double width,
                                        double height,
                                        double angleDeg,
                                        const TemplateTrainOptions &options);

    static TemplateModel trainCircle(const cv::Mat &gray,
                                     cv::Point2d center,
                                     double radius,
                                     const TemplateTrainOptions &options);

    bool empty() const noexcept;
    MatchingMethod method() const noexcept;
    cv::Point2d referenceCenter() const noexcept;
    cv::Size2d referenceSize() const noexcept;
    double referenceAngleDeg() const noexcept;
    bool isCircle() const noexcept;
    double radius() const noexcept;
    const TemplateTrainOptions &trainOptions() const noexcept;

    std::vector<TemplateHit> match(const cv::Mat &gray,
                                   const cv::Rect &searchRoi,
                                   const TemplateMatchOptions &options) const;

    std::vector<std::uint8_t> encode() const;
    static TemplateModel decode(std::span<const std::uint8_t> bytes);

    // Rigid transform mapping reference image coordinates -> current image coordinates.
    geometry::Alignment alignmentFor(cv::Point2d currentCenter, double angleCcwDeg) const noexcept;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace visiontools::recognition
