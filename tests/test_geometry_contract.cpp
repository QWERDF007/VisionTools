// Geometry, image transform and metrology contract for the seven-tool delivery.
// Synthetic fixtures provide analytic ground truth; no model or GPU is involved.
#include "preprocess/BrightnessTool.h"
#include "ToolResultAssertions.h"
#include "preprocess/RotateImageTool.h"
#include "metrology/CircleFit.h"
#include "metrology/CircleMeasureKernel.h"
#include "metrology/CircleTool.h"
#include "metrology/RowSpacingTool.h"
#include "metrology/AssistTools.h"

#include "common/GeometryTransform.h"
#include "common/RoiSupport.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define REQUIRE(cond, message)                                                                                  \
    do                                                                                                          \
    {                                                                                                           \
        if (!(cond))                                                                                            \
        {                                                                                                       \
            std::cerr << "Assertion failed: " << #cond << " (" << message << ") at line " << __LINE__           \
                      << std::endl;                                                                             \
            std::exit(1);                                                                                       \
        }                                                                                                       \
    }                                                                                                           \
    while (0)

using namespace visionrt;
using namespace visiontools;

namespace {
// Each block below names the tool it exercises, so a port number is never a bare literal.
namespace rot    = preprocess::ports::image_rotate;
namespace bright = preprocess::ports::image_brightness;
namespace circle = metrology::ports::measure_circle;
namespace rows   = metrology::ports::measure_row_spacing;
namespace assist = metrology::ports::assist;

// Every image tool publishes its image input as port 1; this helper is shared by all of them.
Values withImages(const cv::Mat &image)
{
    Values values;
    values.emplace(1, DataRef<Image>(Image{image}));
    return values;
}

cv::Mat rotateReference()
{
    cv::Mat image(2, 3, CV_8UC1);
    image.at<std::uint8_t>(0, 0) = 1;
    image.at<std::uint8_t>(0, 1) = 2;
    image.at<std::uint8_t>(0, 2) = 3;
    image.at<std::uint8_t>(1, 0) = 4;
    image.at<std::uint8_t>(1, 1) = 5;
    image.at<std::uint8_t>(1, 2) = 6;
    return image;
}

void geometryAndRoi()
{
    const auto rotation = geometry::rotationAround(10.0, 20.0, 90.0);
    double     x = 0, y = 0;
    rotation.apply(10.0, 20.0, x, y);
    REQUIRE(std::abs(x - 10.0) < 1e-9 && std::abs(y - 20.0) < 1e-9, "rotation must keep its pivot fixed");
    rotation.apply(11.0, 20.0, x, y);
    REQUIRE(std::abs(x - 10.0) < 1e-9 && std::abs(y - 21.0) < 1e-9, "positive angle rotates clockwise");
    const auto inverse = rotation.inverse();
    REQUIRE(inverse.has_value(), "rotation must be invertible");
    inverse->apply(x, y, x, y);
    REQUIRE(std::abs(x - 11.0) < 1e-9 && std::abs(y - 20.0) < 1e-9, "inverse transform must round-trip");
    REQUIRE(rotation.isSimilarity(), "a rotation is a similarity");
    geometry::Alignment scaled;
    scaled.m = {2, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE(!scaled.isSimilarity(), "non-uniform scaling must not be accepted by metrology");

    const roi::Record rectangle = roi::rectangle(30, 30, 20, 10, 0);
    const auto       mask       = roi::regionMask({rectangle}, cv::Size(60, 60), geometry::Alignment::identity());
    int              area       = cv::countNonZero(mask);
    // Pixel centres on the geometric boundary are included, so a 20x10 rectangle
    // selects 21x11 centres.
    REQUIRE(area == 21 * 11, "rectangle mask must include boundary pixel centres");
    REQUIRE(mask.at<std::uint8_t>(30, 30) == 255 && mask.at<std::uint8_t>(2, 2) == 0,
            "mask must mark only the selected region");
}

void rotateTool()
{
    preprocess::RotateImageTool tool(ToolServices{});
    Values                    inputs = withImages(rotateReference());
    inputs.emplace(rot::in::ANGLE_DEG, 90.0);
    inputs.emplace(rot::in::CENTER, Array::fromNested(NestedVector<double, 1>{}));
    inputs.emplace(rot::in::CANVAS, std::string("expand"));
    inputs.emplace(rot::in::INTERPOLATION, std::string("nearest"));
    inputs.emplace(rot::in::BORDER_VALUE, Array::fromNested(NestedVector<double, 1>{0.0}));
    inputs.emplace(rot::in::ALIGNMENT, Array::fromNested(NestedVector<double, 3>{}));
    const auto result = tool.execute(inputs);
    REQUIRE(testview::status(result).code == 0, testview::status(result).message);
    const auto &images = std::get<DataRef<Image>>(result.outputs.at(rot::out::IMAGES)).get();
    REQUIRE(images.size() == 1 && images[0].cols == 2 && images[0].rows == 3,
            "expanded canvas must swap the non-square size");
    REQUIRE(images[0].at<std::uint8_t>(0, 1) == 1 && images[0].at<std::uint8_t>(0, 0) == 4
                && images[0].at<std::uint8_t>(2, 0) == 6,
            "90 degree clockwise pixel mapping is wrong");
    const auto matrices = nestedOf<NestedVector<double, 3>>(std::get<Array>(result.outputs.at(rot::out::ALIGNMENT)));
    REQUIRE(matrices.size() == 1 && std::abs(matrices[0][0][1] + 1.0) < 1e-9
                && std::abs(matrices[0][1][0] - 1.0) < 1e-9 && std::abs(matrices[0][0][2] - 1.0) < 1e-9,
            "rotation alignment must include the expansion translation");
}

void brightnessTool()
{
    cv::Mat image(40, 50, CV_8UC3, cv::Scalar(10, 20, 30));
    Values  inputs = withImages(image);
    inputs.emplace(bright::in::GAIN, 2.0);
    inputs.emplace(bright::in::OFFSET, 10.0);
    inputs.emplace(bright::in::ROIS, DataRef<ROI>(ROI{{{2, 10, 10, 20, 10, 0}}}));
    inputs.emplace(bright::in::ALIGNMENT, Array::fromNested(NestedVector<double, 3>{}));
    const auto result = preprocess::brightnessTool(inputs, ToolServices{});
    REQUIRE(testview::status(result).code == 0, testview::status(result).message);
    const auto &output = std::get<DataRef<Image>>(result.outputs.at(bright::out::IMAGES)).get().front();
    const auto  inside = output.at<cv::Vec3b>(15, 15);
    const auto  outside = output.at<cv::Vec3b>(5, 5);
    REQUIRE(inside[0] == 30 && inside[1] == 50 && inside[2] == 70, "gain and offset must apply inside the ROI");
    REQUIRE(outside[0] == 10 && outside[1] == 20 && outside[2] == 30, "pixels outside the ROI must stay unchanged");
    const auto mask = std::get<DataRef<Image>>(result.outputs.at(bright::out::APPLIED_MASK)).get().front();
    REQUIRE(cv::countNonZero(mask) > 0 && mask.type() == CV_8UC1, "the applied region is a declared output");

    // Floating point images keep values above 255 instead of saturating.
    cv::Mat floatImage(8, 8, CV_32FC1, cv::Scalar(200.0F));
    Values  floatInputs = withImages(floatImage);
    floatInputs.emplace(bright::in::GAIN, 2.0);
    floatInputs.emplace(bright::in::OFFSET, 0.0);
    floatInputs.emplace(bright::in::ROIS, DataRef<ROI>(ROI{}));
    floatInputs.emplace(bright::in::ALIGNMENT, Array::fromNested(NestedVector<double, 3>{}));
    const auto floatResult = preprocess::brightnessTool(floatInputs, ToolServices{});
    REQUIRE(testview::status(floatResult).code == 0, testview::status(floatResult).message);
    const auto &floats = std::get<DataRef<Image>>(floatResult.outputs.at(bright::out::IMAGES)).get().front();
    REQUIRE(std::abs(floats.at<float>(0, 0) - 400.0F) < 1e-3, "float images must not be clipped to 255");
    const auto &floatMask = std::get<DataRef<Image>>(floatResult.outputs.at(bright::out::APPLIED_MASK)).get().front();
    REQUIRE(cv::countNonZero(floatMask) == floatMask.total(), "an empty ROI list covers the whole image");
}

void circleMeasurement()
{
    // Analytic anti-aliased disk: intensity is a linear ramp across the exact
    // edge at distance 40, so the sub-pixel ground truth is unambiguous.
    cv::Mat gray(240, 260, CV_8UC1);
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x)
        {
            const double distance = std::hypot(x - 130.0, y - 120.0);
            const double coverage = std::clamp(0.5 + (40.0 - distance), 0.0, 1.0);
            gray.at<std::uint8_t>(y, x) = static_cast<std::uint8_t>(std::lround(20.0 + 200.0 * coverage));
        }
    metrology::CircleTool tool(ToolServices{});
    Values                inputs = withImages(gray);
    inputs.emplace(circle::in::ROIS, DataRef<ROI>(ROI{{{3, 130.0, 120.0, 40.0}}}));
    inputs.emplace(circle::in::ALIGNMENT, Array::fromNested(NestedVector<double, 3>{}));
    inputs.emplace(circle::in::NUM_CALIPERS, std::int64_t{64});
    inputs.emplace(circle::in::SEARCH_HALF_WIDTH, 8.0);
    inputs.emplace(circle::in::AVERAGE_WIDTH, 3.0);
    inputs.emplace(circle::in::SIGMA, 1.0);
    inputs.emplace(circle::in::MIN_GRADIENT, 8.0);
    inputs.emplace(circle::in::POLARITY, std::string("any"));
    inputs.emplace(circle::in::MIN_POINTS, std::int64_t{12});
    inputs.emplace(circle::in::MIN_COMPLETENESS, 0.6);
    inputs.emplace(circle::in::MIN_COVERAGE_DEG, 180.0);
    inputs.emplace(circle::in::RADIUS_MIN, 0.5);
    inputs.emplace(circle::in::RADIUS_MAX, 1000.0);
    inputs.emplace(circle::in::MAX_RMS, 1.0);
    inputs.emplace(circle::in::UNITS_PER_PIXEL, 1.0);
    inputs.emplace(circle::in::UNIT, std::string("px"));
    const auto result = tool.execute(inputs);
    REQUIRE(testview::status(result).code == 0, testview::status(result).message);
    const auto centers = nestedOf<NestedVector<double, 3>>(std::get<Array>(result.outputs.at(circle::out::CENTERS)));
    const auto radii   = nestedOf<NestedVector<double, 2>>(std::get<Array>(result.outputs.at(circle::out::RADIUS_PX)));
    const auto valid   = nestedOf<NestedVector<bool, 2>>(std::get<Array>(result.outputs.at(circle::out::VALID)));
    REQUIRE(valid.size() == 1 && valid[0].size() == 1 && valid[0][0], "standard circle must be valid");
    REQUIRE(radii[0].size() == 1 && std::abs(radii[0][0] - 40.0) <= 0.6,
            "synthetic circle radius must be recovered within 0.6 px");
    REQUIRE(std::abs(centers[0][0][0] - 130.0) <= 0.6 && std::abs(centers[0][0][1] - 120.0) <= 0.6,
            "synthetic circle centre must be recovered within 0.6 px");
    const auto sources = nestedOf<NestedVector<std::int64_t, 2>>(std::get<Array>(result.outputs.at(circle::out::SOURCE_ROI_INDICES)));
    REQUIRE(sources[0][0] == 0, "valid circles must map back to their nominal ROI index");

    // A missing circle ROI is a configuration error, not a silent empty success.
    Values empty = withImages(gray);
    empty.emplace(circle::in::ROIS, DataRef<ROI>(ROI{}));
    empty.emplace(circle::in::ALIGNMENT, Array::fromNested(NestedVector<double, 3>{}));
    REQUIRE(testview::status(tool.execute(empty)).code != 0, "unconfigured ROI must fail");
}

void rowSpacingMeasurement()
{
    cv::Mat gray(80, 220, CV_8UC1, cv::Scalar(20));
    for (int index = 0; index < 5; ++index)
        cv::rectangle(gray, cv::Rect(20 + index * 30, 10, 8, 60), cv::Scalar(230), cv::FILLED);
    metrology::RowSpacingTool tool(ToolServices{});
    Values                    inputs = withImages(gray);
    inputs.emplace(rows::in::ROIS, DataRef<ROI>(ROI{{{2, 10, 20, 170, 40, 0}}}));
    inputs.emplace(rows::in::ALIGNMENT, Array::fromNested(NestedVector<double, 3>{}));
    inputs.emplace(rows::in::SIGMA, 1.0);
    inputs.emplace(rows::in::MIN_GRADIENT, 8.0);
    inputs.emplace(rows::in::OBJECT_POLARITY, std::string("bright"));
    inputs.emplace(rows::in::WIDTH_MIN, 4.0);
    inputs.emplace(rows::in::WIDTH_MAX, 12.0);
    inputs.emplace(rows::in::UNITS_PER_PIXEL, 1.0);
    inputs.emplace(rows::in::UNIT, std::string("px"));
    const auto result = tool.execute(inputs);
    REQUIRE(testview::status(result).code == 0, testview::status(result).message);
    const auto widths   = nestedOf<NestedVector<double, 3>>(std::get<Array>(result.outputs.at(rows::out::WIDTH_PX)));
    const auto gaps     = nestedOf<NestedVector<double, 3>>(std::get<Array>(result.outputs.at(rows::out::GAP_PX)));
    const auto pitches  = nestedOf<NestedVector<double, 3>>(std::get<Array>(result.outputs.at(rows::out::PITCH_PX)));
    const auto accepted = nestedOf<NestedVector<bool, 3>>(std::get<Array>(result.outputs.at(rows::out::ACCEPTED)));
    REQUIRE(widths.size() == 1 && widths[0].size() == 1 && widths[0][0].size() == 5,
            "five bright bands must produce five pairs");
    REQUIRE(std::abs(widths[0][0].front() - 8.0) <= 0.6, "band width must be recovered within 0.6 px");
    REQUIRE(gaps[0][0].size() == 4, "gap count must be pair count minus one");
    REQUIRE(std::abs(gaps[0][0].front() - 22.0) <= 0.8, "gap must be recovered within 0.8 px");
    REQUIRE(std::abs(pitches[0][0].front() - 30.0) <= 0.8, "pitch must be recovered within 0.8 px");
    REQUIRE(accepted[0][0].front(), "declared width range must accept the measured band");

    // Incompatible width range keeps the raw pairing but marks it unaccepted.
    inputs[rows::in::WIDTH_MIN] = 20.0;
    inputs[rows::in::WIDTH_MAX] = 40.0;
    const auto rejected = tool.execute(inputs);
    REQUIRE(testview::status(rejected).code == 0, testview::status(rejected).message);
    const auto rejectedAccepted = nestedOf<NestedVector<bool, 3>>(std::get<Array>(rejected.outputs.at(rows::out::ACCEPTED)));
    const auto rejectedWidths   = nestedOf<NestedVector<double, 3>>(std::get<Array>(rejected.outputs.at(rows::out::WIDTH_PX)));
    REQUIRE(rejectedWidths[0][0].size() == 5, "unaccepted pairs must stay in the raw result");
    REQUIRE(!rejectedAccepted[0][0].front(), "width range must mark pairs as unaccepted");
}

void assistCandidate()
{
    cv::Mat gray(200, 200, CV_8UC1, cv::Scalar(15));
    cv::circle(gray, cv::Point(100, 100), 30, cv::Scalar(220), cv::FILLED, cv::LINE_AA);
    metrology::AssistCircleTool tool(ToolServices{});
    Values inputs;
    inputs.emplace(assist::in::IMAGES, DataRef<Image>(Image{gray}));
    inputs.emplace(assist::in::CURSOR, Array::fromNested(NestedVector<double, 1>{130.0, 100.0}));
    inputs.emplace(assist::in::PHASE, std::string("hover"));
    inputs.emplace(assist::in::SEED_ROIS, DataRef<ROI>(ROI{}));
    inputs.emplace(assist::in::ALIGNMENT, Array::fromNested(NestedVector<double, 3>{}));
    inputs.emplace(assist::in::SEARCH_EXTENT, std::int64_t{128});
    const auto result = tool.execute(inputs);
    REQUIRE(testview::status(result).code == 0, testview::status(result).message);
    REQUIRE(std::get<bool>(result.outputs.at(assist::out::CANDIDATE_VALID)), "assist must find the synthetic circle");
    const auto proposal = std::get<DataRef<ROI>>(result.outputs.at(assist::out::PROPOSAL_ROIS)).get();
    REQUIRE(proposal.size() == 1 && proposal[0].size() == 1 && proposal[0][0].front() == 3.0,
            "proposal must be a circle in reference coordinates");
}

// Synthetic anti-aliased annulus: the only real edge lives on `radius`, and the ramp makes the
// sub-pixel position unambiguous.
cv::Mat diskImage(int size, double centerX, double centerY, double radius)
{
    cv::Mat gray(size, size, CV_8UC1);
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x)
        {
            const double distance = std::hypot(x - centerX, y - centerY);
            const double coverage = std::clamp(0.5 + (radius - distance), 0.0, 1.0);
            gray.at<std::uint8_t>(y, x) = static_cast<std::uint8_t>(std::lround(20.0 + 200.0 * coverage));
        }
    return gray;
}

Values circleInputs(const cv::Mat &gray, double centerX, double centerY, double radius)
{
    Values inputs;
    inputs.emplace(circle::in::IMAGES, DataRef<Image>(Image{gray}));
    inputs.emplace(circle::in::ROIS, DataRef<ROI>(ROI{{{3, centerX, centerY, radius}}}));
    inputs.emplace(circle::in::ALIGNMENT, Array::fromNested(NestedVector<double, 3>{}));
    inputs.emplace(circle::in::NUM_CALIPERS, std::int64_t{64});
    inputs.emplace(circle::in::SEARCH_HALF_WIDTH, 8.0);
    inputs.emplace(circle::in::AVERAGE_WIDTH, 3.0);
    inputs.emplace(circle::in::SIGMA, 1.0);
    inputs.emplace(circle::in::MIN_GRADIENT, 8.0);
    inputs.emplace(circle::in::POLARITY, std::string("any"));
    inputs.emplace(circle::in::MIN_POINTS, std::int64_t{12});
    inputs.emplace(circle::in::MIN_COMPLETENESS, 0.6);
    inputs.emplace(circle::in::MIN_COVERAGE_DEG, 180.0);
    inputs.emplace(circle::in::RADIUS_MIN, 0.5);
    inputs.emplace(circle::in::RADIUS_MAX, 1000.0);
    inputs.emplace(circle::in::MAX_RMS, 1.0);
    inputs.emplace(circle::in::UNITS_PER_PIXEL, 1.0);
    inputs.emplace(circle::in::UNIT, std::string("px"));
    return inputs;
}

// The quality gate must run on the *final* inlier set. A short true arc surrounded by scattered
// outliers used to pass because minPoints/completeness/coverage were measured before rejection.
void circleQualityGate()
{
    cv::Mat gray = diskImage(240, 130.0, 120.0, 40.0);
    // Keep only a 70 degree wedge of the true edge, then add isolated specks all around the ring.
    cv::Mat arc(240, 240, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x)
        {
            const double distance = std::hypot(x - 130.0, y - 120.0);
            const double angle     = std::atan2(y - 120.0, x - 130.0) * 180.0 / 3.14159265358979323846;
            const bool   inWedge   = angle > -35.0 && angle < 35.0;
            if (inWedge)
                arc.at<std::uint8_t>(y, x) = gray.at<std::uint8_t>(y, x);
            else
                arc.at<std::uint8_t>(y, x) = 20;
        }
    std::mt19937 random(20261025u);
    std::uniform_real_distribution<double> angleDistribution(-180.0, 180.0);
    // Specks sit 4-8 px off the ring: close enough for every caliper to report an edge (so the
    // pre-rejection counts stay high) yet far enough to be rejected as final inliers.
    std::uniform_real_distribution<double> offsetDistribution(4.0, 8.0);
    std::uniform_int_distribution<int>     signDistribution(0, 1);
    for (int index = 0; index < 90; ++index)
    {
        const double angle  = angleDistribution(random) * 3.14159265358979323846 / 180.0;
        const double offset = offsetDistribution(random) * (signDistribution(random) == 0 ? -1.0 : 1.0);
        const double radius = 40.0 + offset;
        const int    x      = static_cast<int>(std::lround(130.0 + std::cos(angle) * radius));
        const int    y      = static_cast<int>(std::lround(120.0 + std::sin(angle) * radius));
        if (x >= 1 && y >= 1 && x + 1 < arc.cols && y + 1 < arc.rows)
            arc.at<std::uint8_t>(y, x) = 200;
    }
    metrology::CircleTool tool(ToolServices{});
    const auto            shortArc = tool.execute(circleInputs(arc, 130.0, 120.0, 40.0));
    REQUIRE(testview::status(shortArc).code == 0, testview::status(shortArc).message);
    const auto arcValid = nestedOf<NestedVector<bool, 2>>(std::get<Array>(shortArc.outputs.at(circle::out::VALID)));
    REQUIRE(arcValid.size() == 1 && !arcValid[0][0],
            "short arc with ring-wide outliers must not pass the final inlier gate");
    const auto arcMessages = nestedOf<NestedVector<std::string, 2>>(std::get<Array>(shortArc.outputs.at(circle::out::REASONS)));
    REQUIRE(!arcMessages[0][0].empty(), "a rejected circle must explain its reason");

    // Too few final inliers: a very short clean arc can not describe a circle either.
    cv::Mat shortClean(240, 240, CV_8UC1, cv::Scalar(20));
    cv::ellipse(shortClean, cv::Point(130, 120), cv::Size(40, 40), 0, -25, 25, cv::Scalar(220), 1, cv::LINE_AA);
    const auto shortCleanResult = tool.execute(circleInputs(shortClean, 130.0, 120.0, 40.0));
    REQUIRE(testview::status(shortCleanResult).code == 0, testview::status(shortCleanResult).message);
    const auto cleanValid = nestedOf<NestedVector<bool, 2>>(std::get<Array>(shortCleanResult.outputs.at(circle::out::VALID)));
    REQUIRE(cleanValid.size() == 1 && !cleanValid[0][0], "a 50 degree arc must not count as a circle");

    // Geometry and visuals must come from the same final inlier set, and a fit that keeps no
    // point may not report a zero-RMS success.
    std::vector<cv::Point2d> scattered{{0, 0}, {0, 1}, {100, 100}, {100, 101}, {50, 50}};
    metrology::CircleFitOptions options;
    options.maxRms = 0.1;
    const auto fit = metrology::fitCircle(scattered, options);
    REQUIRE(!fit.has_value(), "zero inliers must fail instead of reporting a perfect fit");
    cv::Mat valid{diskImage(200, 100.0, 100.0, 30.0)};
    metrology::CircleOptions measureOptions;
    measureOptions.calipers        = 64;
    measureOptions.searchHalfWidth = 6.0;
    measureOptions.minPoints       = 12;
    measureOptions.minCompleteness = 0.6;
    measureOptions.minCoverageDeg  = 180.0;
    const auto measurement = metrology::measureCircleKernel(valid, {100.0, 100.0}, 30.0, measureOptions);
    REQUIRE(measurement.valid, measurement.reason);
    REQUIRE(measurement.rms > 0.0, "a measured circle must report a real residual");
    REQUIRE(!measurement.edges.empty() && measurement.edges.size() == measurement.angles.size()
                && measurement.edges.size() == measurement.edgeOffsets.size(),
            "rendered points must mirror the final inlier set");
    REQUIRE(std::abs(measurement.completeness * 64.0 - static_cast<double>(measurement.edges.size())) < 1e-9,
            "completeness must be derived from the same final inliers");
}

// Candidate search is bounded by the cursor window and by a hard distance limit.
void assistLocalWindow()
{
    cv::Mat gray(400, 400, CV_8UC1, cv::Scalar(15));
    cv::circle(gray, cv::Point(320, 320), 30, cv::Scalar(220), cv::FILLED, cv::LINE_AA);
    metrology::AssistCircleTool tool(ToolServices{});
    Values                      inputs;
    inputs.emplace(assist::in::IMAGES, DataRef<Image>(Image{gray}));
    inputs.emplace(assist::in::CURSOR, Array::fromNested(NestedVector<double, 1>{60.0, 60.0}));
    inputs.emplace(assist::in::PHASE, std::string("hover"));
    inputs.emplace(assist::in::SEED_ROIS, DataRef<ROI>(ROI{}));
    inputs.emplace(assist::in::ALIGNMENT, Array::fromNested(NestedVector<double, 3>{}));
    inputs.emplace(assist::in::SEARCH_EXTENT, std::int64_t{80});
    const auto outside = tool.execute(inputs);
    REQUIRE(testview::status(outside).code == 0, testview::status(outside).message);
    REQUIRE(!std::get<bool>(outside.outputs.at(assist::out::CANDIDATE_VALID)),
            "a circle outside the cursor window must not become a candidate");

    // Non-integer resize factors must still map back through pixel centres.
    cv::Mat odd = diskImage(401, 200.5, 200.5, 40.0);
    inputs[assist::in::IMAGES] = DataRef<Image>(Image{odd});
    inputs[assist::in::CURSOR] = Array::fromNested(NestedVector<double, 1>{240.0, 200.5});
    inputs[assist::in::SEARCH_EXTENT] = std::int64_t{160};
    const auto scaled = tool.execute(inputs);
    REQUIRE(testview::status(scaled).code == 0, testview::status(scaled).message);
    REQUIRE(std::get<bool>(scaled.outputs.at(assist::out::CANDIDATE_VALID)), "odd-sized images must still be searched");
    const auto scaledProposal = std::get<DataRef<ROI>>(scaled.outputs.at(assist::out::PROPOSAL_ROIS)).get();
    REQUIRE(std::abs(scaledProposal[0][0][1] - 200.5) <= 0.8 && std::abs(scaledProposal[0][0][2] - 200.5) <= 0.8,
            "non-integer scaling must preserve the circle centre");

    // A seed states the nominal geometry: the contour search is skipped, so a radius outside the
    // search extent still resolves as long as the user pointed at the target.
    inputs[assist::in::CURSOR] = Array::fromNested(NestedVector<double, 1>{240.5, 200.5});
    inputs[assist::in::SEARCH_EXTENT] = std::int64_t{50};
    inputs[assist::in::SEED_ROIS]     = DataRef<ROI>(ROI{{{3, 200.5, 200.5, 40.0}}});
    const auto seeded = tool.execute(inputs);
    REQUIRE(testview::status(seeded).code == 0, testview::status(seeded).message);
    REQUIRE(std::get<bool>(seeded.outputs.at(assist::out::CANDIDATE_VALID)), "seed geometry must not depend on contour search");
    const auto seededProposal = std::get<DataRef<ROI>>(seeded.outputs.at(assist::out::PROPOSAL_ROIS)).get();
    REQUIRE(std::abs(seededProposal[0][0][3] - 40.0) <= 0.8, "seeded radius must survive refinement");
}

} // namespace

int main()
{
    geometryAndRoi();
    rotateTool();
    brightnessTool();
    circleMeasurement();
    rowSpacingMeasurement();
    assistCandidate();
    circleQualityGate();
    assistLocalWindow();
    std::cout << "GEOMETRY_METROLOGY_PASS rotate, brightness, ROI, circle, row spacing and assist verified"
              << std::endl;
    return 0;
}
