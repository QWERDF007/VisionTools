#include "metrology/CalibrateCameraTool.h"
#include "metrology/RectifyImageTool.h"
#include "metrology/RowSpacingTool.h"
#include "common/GeometryTransform.h"
#include "common/PlanarCalibration.h"
#include "ToolResultAssertions.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace visionrt;
using namespace visiontools;
namespace cal = metrology::ports::calibrate_camera;
namespace rec = metrology::ports::rectify_image;
namespace row = metrology::ports::measure_row_spacing;

namespace {
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

cv::Mat loadImage(const std::filesystem::path &path)
{
    std::ifstream stream(path, std::ios::binary);
    std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    auto image = cv::imdecode(bytes, cv::IMREAD_UNCHANGED);
    require(!image.empty(), "rectify test image unavailable");
    return image;
}

void verifySyntheticDistortedGrid()
{
    const cv::Size size(640, 480);
    geometry::PlanarCalibration plane;
    plane.imageSize = size;
    plane.center = {320.0, 240.0};
    plane.division = -0.35;
    plane.imageToMillimeters = cv::Matx33d(
        0.05,  0.0, -16.0,
        0.0,  -0.05, 12.0,
        0.0,   0.0,  1.0);

    // Draw dark vertical bars spaced every 2.0 mm in world coordinates into the distorted image.
    cv::Mat distorted(size, CV_8UC1, cv::Scalar(220));
    for (int y = 0; y < size.height; ++y)
    {
        auto *ptr = distorted.ptr<std::uint8_t>(y);
        for (int x = 0; x < size.width; ++x)
        {
            const cv::Point2d mm = plane.map({static_cast<double>(x), static_cast<double>(y)});
            const double phase = std::fmod(mm.x + 100.0, 2.0);
            if (phase >= 0.7 && phase <= 1.3 && mm.y >= -4.0 && mm.y <= 4.0)
                ptr[x] = 30;
        }
    }

    metrology::RectifyImageTool tool;
    const auto res = tool.execute({
        {rec::in::IMAGES, DataRef<Image>(Image{distorted})},
        {rec::in::CALIBRATION, geometry::calibrationArray(plane)},
        {rec::in::UNIT, std::string("mm")}
    });
    require(static_cast<bool>(testview::status(res)), "synthetic rectification succeeded");

    const auto &rectifiedImages = std::get<DataRef<Image>>(res.outputs.at(rec::out::RECTIFIED_IMAGES)).get();
    require(rectifiedImages.size() == 1, "one rectified image returned");
    const cv::Mat &rectified = rectifiedImages.front();
    require(rectified.cols > 0 && rectified.rows > 0 && rectified.type() == distorted.type(), "rectified image shape and type valid");

    const auto &adaptedArray = std::get<Array>(res.outputs.at(rec::out::ADAPTED_CALIBRATION));
    const auto adapted = geometry::calibrationsFromArray(adaptedArray, 1).front();
    require(adapted.division == 0.0, "adapted calibration has zero lens distortion");
    require(adapted.imageSize == rectified.size(), "adapted calibration matches rectified image size");

    const auto pixelSizes = std::get<Array>(res.outputs.at(rec::out::PIXEL_SIZE)).elements<double>();
    require(pixelSizes.size() == 1 && std::abs(pixelSizes[0] - 0.05) < 1e-4, "center pixel size matches 0.05 mm/px");

    const auto &frame = std::get<CoordinateFrame2D>(res.outputs.at(rec::out::COORDINATE_FRAME));
    require(frame.count(0) == 1, "rectified coordinate frame published");
    const auto f0 = frame.at(0, 0);
    require(f0.xAxis().x > 0.0 && f0.yAxis().y < 0.0, "rectified frame keeps world +X right and +Y up");

    // Measure bar spacing on rectified image using adapted calibration.
    const cv::Point2d leftPx = adapted.project({-10.0, 0.0});
    const cv::Point2d rightPx = adapted.project({10.0, 0.0});
    const double roiW = std::abs(rightPx.x - leftPx.x);
    const double roiH = 60.0;
    const cv::Point2d roiCenter = 0.5 * (leftPx + rightPx);

    metrology::RowSpacingTool rowTool;
    const auto measured = rowTool.execute({
        {row::in::IMAGES, DataRef<Image>(Image{rectified})},
        {row::in::ROIS, Rectangle::fromEncoded(ROI{{makeRectangle(roiCenter.x - roiW / 2.0, roiCenter.y - roiH / 2.0, roiW, roiH, 0.0)}})},
        {row::in::ALIGNMENT, geometry::alignmentArray(geometry::Alignment::identity())},
        {row::in::CALIBRATION, adaptedArray},
        {row::in::SIGMA, 1.0},
        {row::in::MIN_GRADIENT, 10.0},
        {row::in::OBJECT_POLARITY, std::string("dark")},
        {row::in::WIDTH_MIN, 1.0},
        {row::in::WIDTH_MAX, 1000.0},
        {row::in::UNITS_PER_PIXEL, 1.0},
        {row::in::UNIT, std::string("mm")}
    });
    require(static_cast<bool>(testview::status(measured)), "row spacing on rectified image succeeded");
    const auto pitches = nestedOf<NestedVector<double, 3>>(std::get<Array>(measured.outputs.at(row::out::PITCH))).front().front();
    require(pitches.size() >= 8, "detected expected number of bars");
    const double mean = std::accumulate(pitches.begin(), pitches.end(), 0.0) / static_cast<double>(pitches.size());
    require(std::abs(mean - 2.0) < 0.02, "rectified bar pitch matches 2.0 mm");

    // Error paths: empty calibration, mismatched size, invalid unit, upstream skip.
    const auto emptyCal = tool.execute({
        {rec::in::IMAGES, DataRef<Image>(Image{distorted})},
        {rec::in::CALIBRATION, Array::fromNested(NestedVector<double, 3>{})},
        {rec::in::UNIT, std::string("mm")}
    });
    require(!testview::status(emptyCal), "empty calibration must fail");

    cv::Mat wrongSize(100, 100, CV_8UC1, cv::Scalar(0));
    const auto mismatch = tool.execute({
        {rec::in::IMAGES, DataRef<Image>(Image{wrongSize})},
        {rec::in::CALIBRATION, geometry::calibrationArray(plane)},
        {rec::in::UNIT, std::string("mm")}
    });
    require(!testview::status(mismatch), "image size mismatch must fail");

    const auto badUnit = tool.execute({
        {rec::in::IMAGES, DataRef<Image>(Image{distorted})},
        {rec::in::CALIBRATION, geometry::calibrationArray(plane)},
        {rec::in::UNIT, std::string("inch")}
    });
    require(!testview::status(badUnit), "unsupported unit must fail");

    const auto skipped = tool.execute({
        {kUpstreamStatusPort, std::int64_t{7}},
        {rec::in::IMAGES, DataRef<Image>(Image{distorted})},
        {rec::in::CALIBRATION, geometry::calibrationArray(plane)},
        {rec::in::UNIT, std::string("mm")}
    });
    require(testview::status(skipped).code == 7 && isMissing(skipped.outputs.at(rec::out::RECTIFIED_IMAGES)),
            "upstream failure skips rectification");
}

void verifyRealDistortedRuler(const std::filesystem::path &platePath,
                              const std::filesystem::path &rulerPath,
                              const std::filesystem::path &descriptorPath,
                              const std::filesystem::path &saveModelPath)
{
    const cv::Mat plateImg = loadImage(platePath);
    const cv::Mat rulerImg = loadImage(rulerPath);

    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto modelPath = saveModelPath.empty()
        ? (std::filesystem::temp_directory_path() / ("visiontools-rectify-" + stamp + ".zip"))
        : saveModelPath;
    if (modelPath.has_parent_path())
        std::filesystem::create_directories(modelPath.parent_path());

    TrainingSet set;
    set.images.emplace(1, trainingFrame(1, plateImg, ColorOrder::Automatic));
    set.records.emplace(1, TrainingRecord{1, 1, {}});

    const auto descU8 = descriptorPath.generic_u8string();
    const auto modelU8 = modelPath.generic_u8string();

    metrology::CalibrateCameraTool calTool;
    const auto trained = calTool.train({
        {cal::train::DATASET, DataRef<TrainingSet>(set)},
        {cal::train::PLATE_DESCRIPTION, std::string(reinterpret_cast<const char *>(descU8.data()), descU8.size())},
        {cal::train::MODEL_PATH, std::string(reinterpret_cast<const char *>(modelU8.data()), modelU8.size())},
        {cal::train::MAX_REPROJECTION_ERROR, 0.5}
    });
    if (!testview::status(trained))
        std::cerr << testview::status(trained).message << ": " << testview::status(trained).detail << '\n';
    require(static_cast<bool>(testview::status(trained)), "distorted plate calibration training succeeded");
    const double rms = std::get<double>(trained.outputs.at(cal::out::REPROJECTION_ERROR));
    require(rms < 0.3, "distorted plate calibration reprojection error < 0.3 px");

    ToolConfiguration cfg;
    cfg.models.emplace(cal::MODEL_ARTIFACT, modelPath);
    require(static_cast<bool>(calTool.initialize(cfg)), "calibration model initialized");

    const auto calResult = calTool.execute({
        {cal::in::IMAGES, DataRef<Image>(Image{rulerImg})},
        {cal::in::REFERENCE_SEGMENTS, Line{}},
        {cal::in::REFERENCE_LENGTH, 1.0},
        {cal::in::UNIT, std::string("mm")}
    });
    require(static_cast<bool>(testview::status(calResult)), "calibration execute succeeded");
    const auto &calArray = std::get<Array>(calResult.outputs.at(cal::out::CALIBRATION));

    metrology::RectifyImageTool rectTool;
    const auto rectResult = rectTool.execute({
        {rec::in::IMAGES, DataRef<Image>(Image{rulerImg})},
        {rec::in::CALIBRATION, calArray},
        {rec::in::UNIT, std::string("mm")}
    });
    if (!testview::status(rectResult))
        std::cerr << testview::status(rectResult).message << ": " << testview::status(rectResult).detail << '\n';
    require(static_cast<bool>(testview::status(rectResult)), "real ruler rectification succeeded");

    const cv::Mat &rectified = std::get<DataRef<Image>>(rectResult.outputs.at(rec::out::RECTIFIED_IMAGES)).get().front();
    const auto &adaptedArray = std::get<Array>(rectResult.outputs.at(rec::out::ADAPTED_CALIBRATION));
    const auto adapted = geometry::calibrationsFromArray(adaptedArray, 1).front();
    const double pixelSize = std::get<Array>(rectResult.outputs.at(rec::out::PIXEL_SIZE)).elements<double>().front();

    std::cout << "Distorted RMS=" << rms << " px; rectified=" << rectified.cols << "x" << rectified.rows
              << "; pixelSize=" << pixelSize << " mm/px\n";
    require(pixelSize > 0.04 && pixelSize < 0.06, "pixel size near 0.0474 mm/px");

    // Measure ruler ticks near top of rectified ruler using a horizontal band across the ruler scale.
    // Find row band with the dark ticks by locating the vertical projection of the ruler scale.
    const cv::Point2d leftRef = adapted.project(geometry::calibrationsFromArray(calArray, 1).front().map({60.0, 200.0}));
    const cv::Point2d rightRef = adapted.project(geometry::calibrationsFromArray(calArray, 1).front().map({1236.0, 200.0}));
    std::cout << "Mapped sample ticks band: left=(" << leftRef.x << "," << leftRef.y << ") right=("
              << rightRef.x << "," << rightRef.y << ")\n";

    // Also evaluate the top tick band in rectified coordinates directly around the detected tick strip.
    double bestMean = 0.0;
    double bestStdUm = 1e9;
    std::size_t bestCount = 0;
    double bestY = 0.0;
    for (int cy = 120; cy <= 420; cy += 10)
    {
        const double roiX = rectified.cols * 0.06;
        const double roiW = rectified.cols * 0.88;
        const double roiH = 36.0;
        metrology::RowSpacingTool rowTool;
        const auto m = rowTool.execute({
            {row::in::IMAGES, DataRef<Image>(Image{rectified})},
            {row::in::ROIS, Rectangle::fromEncoded(ROI{{makeRectangle(roiX, cy - roiH * 0.5, roiW, roiH, 0.0)}})},
            {row::in::ALIGNMENT, geometry::alignmentArray(geometry::Alignment::identity())},
            {row::in::CALIBRATION, adaptedArray},
            {row::in::SIGMA, 1.0},
            {row::in::MIN_GRADIENT, 8.0},
            {row::in::OBJECT_POLARITY, std::string("dark")},
            {row::in::WIDTH_MIN, 1.0},
            {row::in::WIDTH_MAX, 1000.0},
            {row::in::UNITS_PER_PIXEL, 1.0},
            {row::in::UNIT, std::string("mm")}
        });
        if (!testview::status(m)) continue;
        const auto pitches = nestedOf<NestedVector<double, 3>>(std::get<Array>(m.outputs.at(row::out::PITCH))).front().front();
        if (pitches.size() < 40) continue;
        const double mean = std::accumulate(pitches.begin(), pitches.end(), 0.0) / static_cast<double>(pitches.size());
        double sq = 0.0;
        for (double v : pitches) sq += (v - mean) * (v - mean);
        const double stdUm = std::sqrt(sq / static_cast<double>(pitches.size() - 1)) * 1000.0;
        if (std::abs(mean - 1.0) < 0.05 && pitches.size() >= bestCount && stdUm < bestStdUm)
        {
            bestCount = pitches.size();
            bestMean = mean;
            bestStdUm = stdUm;
            bestY = cy;
        }
    }
    std::cout << "Best rectified tick row cy=" << bestY << "; pitches=" << bestCount
              << "; mean=" << bestMean << " mm; std=" << bestStdUm << " um\n";
    require(bestCount >= 40, "rectified ruler detected at least 40 pitches");
    require(bestMean >= 0.99 && bestMean <= 1.01, "rectified ruler mean pitch within 1% of 1 mm");
    require(bestStdUm < 20.0, "rectified ruler sample std < 20 um");

    if (saveModelPath.empty())
    {
        std::error_code ignored;
        std::filesystem::remove(modelPath, ignored);
    }
    calTool.release();
}
} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t **argv)
#else
int main(int argc, char **argv)
#endif
try
{
    verifySyntheticDistortedGrid();
    if (argc >= 4)
    {
        const std::filesystem::path savePath = argc >= 5 ? std::filesystem::path(argv[4]) : std::filesystem::path{};
        verifyRealDistortedRuler(std::filesystem::path(argv[1]), std::filesystem::path(argv[2]),
                                 std::filesystem::path(argv[3]), savePath);
    }
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
