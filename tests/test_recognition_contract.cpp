// Recognition contract: the straight-edge alignment must recover a known reference-to-current pose,
// and the colour tool must return the trained class per detection circle while rejecting colours it
// never saw. Fixtures are synthetic so every expectation has an analytic answer.
#include "recognition/AlignStraightBordersTool.h"
#include "recognition/PickStraightEdgeTool.h"
#include "recognition/StraightEdgeProbe.h"
#include "recognition/RecognizeColorTool.h"
#include "recognition/errors/RecognitionErrors.h"

#include "common/EdgeProfile.h"
#include "common/GeometryTransform.h"
#include "ToolResultAssertions.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
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
namespace align = recognition::ports::align_straight_borders;
namespace color = recognition::ports::recognize_color;

std::filesystem::path fixtureDirectory()
{
    const auto directory = std::filesystem::temp_directory_path() / "visiontools_recognition_contract";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    return directory;
}

void writeText(const std::filesystem::path &path, const std::string &content)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    REQUIRE(file.good(), "fixture file must open");
    file << content;
}

std::string generic(const std::filesystem::path &path)
{
    return path.generic_string();
}

// A white square on black: the bottom and right borders are the two trained edges.
cv::Mat connectorImage()
{
    cv::Mat image(400, 400, CV_8UC3, cv::Scalar{0, 0, 0});
    cv::rectangle(image, cv::Rect{100, 100, 200, 200}, cv::Scalar{255, 255, 255}, cv::FILLED);
    return image;
}

geometry::Alignment pose(double translationX, double translationY, double angleDegrees)
{
    auto alignment = geometry::rotationAround(200.0, 200.0, angleDegrees);
    alignment.m[2] += translationX;
    alignment.m[5] += translationY;
    return alignment;
}

std::filesystem::path straightEdgeAsset(const std::filesystem::path &directory)
{
    const auto path = directory / "alignment.json";
    writeText(path, std::string{"{\n"
                                "  \"schema_version\": 1,\n"
                                "  \"kind\": \"straight_edges\",\n"
                                "  \"reference_image\": \""}
                        + generic(directory / "reference.png")
                        + "\",\n"
                          "  \"segments\": [\n"
                          "    {\"name\": \"底边\", \"from\": [120, 300], \"to\": [280, 300], \"search\": [200, 300, 160, 40, 0]},\n"
                          "    {\"name\": \"右边\", \"from\": [300, 120], \"to\": [300, 280], \"search\": [300, 200, 160, 40, 0]}\n"
                          "  ]\n"
                          "}\n");
    return path;
}

Values alignmentInputs(const cv::Mat &image, const std::string &polarity)
{
    Values inputs;
    inputs.emplace(align::in::IMAGES, DataRef<Image>(Image{image}));
    inputs.emplace(align::in::CONTRAST, 30.0);
    inputs.emplace(align::in::EDGE_WIDTH, std::int64_t{3});
    inputs.emplace(align::in::POLARITY, polarity);
    inputs.emplace(align::in::MIN_COMPLETENESS, 0.4);
    return inputs;
}

void straightEdgeAlignment(const std::filesystem::path &directory)
{
    const auto asset = straightEdgeAsset(directory);
    const auto reference = connectorImage();

    recognition::AlignStraightBordersTool tool;
    Values                                init;
    init.emplace(align::init::TRAINING, generic(asset));
    REQUIRE(tool.initialize(init).code == 0, "training asset must load");

    const auto identity = tool.execute(alignmentInputs(reference, "any"));
    REQUIRE(testview::status(identity).code == 0, testview::status(identity).message);
    const auto identityMatrices = visionrt::nestedOf<NestedVector<double, 3>>(
        std::get<Array>(identity.outputs.at(align::out::ALIGNMENTS)));
    REQUIRE(identityMatrices.size() == 1 && identityMatrices.front().size() == 3,
            "one 3x3 matrix per image");
    const auto &matrix = identityMatrices.front();
    // A filled rectangle covers whole pixels, so its intensity step sits half a pixel inside the
    // geometric border; the pose must reproduce the image up to that quantisation.
    const auto identityError = [&](int row, int column) {
        return std::abs(matrix[row][column] - (row == column ? 1.0 : 0.0));
    };
    REQUIRE(identityError(0, 0) < 1e-3 && identityError(1, 1) < 1e-3, "reference image keeps its axes");
    REQUIRE(identityError(0, 1) < 1e-3 && identityError(1, 0) < 1e-3, "reference image keeps its rotation");
    REQUIRE(std::hypot(matrix[0][2], matrix[1][2]) < 0.75, "reference image keeps its position");

    // The part moves: rotation about the centre plus a shift. The tool must report that pose, with
    // x as column and y as row, and it must not swap the two.
    const auto expected = pose(12.0, -7.0, 5.0);
    cv::Mat    moved;
    cv::Mat    transform(2, 3, CV_64F);
    for (int row = 0; row < 2; ++row)
        for (int column = 0; column < 3; ++column)
            transform.at<double>(row, column) = expected.m[row * 3 + column];
    cv::warpAffine(reference, moved, transform, reference.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar{0, 0, 0});

    const auto result = tool.execute(alignmentInputs(moved, "any"));
    REQUIRE(testview::status(result).code == 0, testview::status(result).message);
    const auto matrices = visionrt::nestedOf<NestedVector<double, 3>>(
        std::get<Array>(result.outputs.at(align::out::ALIGNMENTS)));
    REQUIRE(matrices.size() == 1, "one matrix for one image");
    std::array<double, 9> flat{};
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            flat[static_cast<std::size_t>(row * 3 + column)] = matrices.front()[row][column];
    const geometry::Alignment recovered = *geometry::Alignment::fromRows(flat);
    REQUIRE(recovered.isSimilarity(), "a straight-edge pose is a similarity");

    for (const cv::Point2d probe : {cv::Point2d{200, 300}, cv::Point2d{300, 200}, cv::Point2d{150, 250}})
    {
        double expectedX = 0, expectedY = 0, actualX = 0, actualY = 0;
        expected.apply(probe.x, probe.y, expectedX, expectedY);
        recovered.apply(probe.x, probe.y, actualX, actualY);
        REQUIRE(std::hypot(actualX - expectedX, actualY - expectedY) < 0.75,
                "recovered pose must match the applied pose");
    }

    // Both trained edges are the same polarity from the band's point of view only for the right
    // border; asking for that polarity everywhere must fail instead of inventing the bottom edge.
    const auto mismatch = tool.execute(alignmentInputs(moved, "light_to_dark"));
    REQUIRE(testview::status(mismatch).code == static_cast<std::int64_t>(recognition::RecognitionError::EdgeNotFound),
            "a polarity that fits one edge must not fabricate the other");
    REQUIRE(testview::businessCount(mismatch) == 3, "a failed alignment still publishes every port as Missing");
}

void straightEdgeProbe()
{
    // Point-and-click feeds the same training asset the user would otherwise draw by hand: the band
    // must follow the edge under the cursor, and empty space must answer "nothing here".
    const auto image = connectorImage();
    const auto gray  = edge::grayscale(image);

    recognition::EdgeProbeRequest request;
    request.cursor       = {200.0, 300.0};
    request.searchExtent = 96;
    const auto bottom    = recognition::probeStraightEdge(gray, request);
    REQUIRE(bottom.valid, bottom.quality.c_str());
    REQUIRE(bottom.proposal.size() == 1, "one band per click");
    const auto rectangle = visionrt::RegionView(bottom.proposal.front()).rectangle();
    REQUIRE(std::abs(rectangle.center().y - 300.0) < 2.0, "the band sits on the edge under the cursor");
    REQUIRE(std::abs(std::remainder(rectangle.angle(), 180.0)) < 2.0, "a horizontal edge gives a level band");
    REQUIRE(rectangle.width() >= 96.0, "the band spans the probe window along the edge");

    // The same click on the vertical edge must rotate the band, not copy the first answer.
    request.cursor = {300.0, 200.0};
    const auto right = recognition::probeStraightEdge(gray, request);
    REQUIRE(right.valid, right.quality.c_str());
    const auto vertical = visionrt::RegionView(right.proposal.front()).rectangle();
    REQUIRE(std::abs(std::abs(std::remainder(vertical.angle(), 180.0)) - 90.0) < 2.0,
            "a vertical edge gives a quarter-turn band");

    // Empty space has no structure tensor, so nothing is proposed.
    request.cursor = {40.0, 40.0};
    const auto empty = recognition::probeStraightEdge(gray, request);
    REQUIRE(!empty.valid && !empty.quality.empty(), "empty space must not invent an edge");
    REQUIRE(empty.proposal.empty(), "an invalid probe carries no proposal");
}

void colorTraining(const std::filesystem::path &directory, std::filesystem::path &asset,
                   std::filesystem::path &reference)
{
    reference = directory / "colors.png";
    cv::Mat image(400, 400, CV_8UC3, cv::Scalar{128, 128, 128});
    const std::vector<std::pair<std::string, cv::Scalar>> patches{
        {"purple", {128, 0, 128}},
        {"red",    {0, 0, 255}  },
        {"blue",   {255, 0, 0}  },
        {"yellow", {0, 255, 255}},
    };
    std::string records;
    for (std::size_t index = 0; index < patches.size(); ++index)
    {
        const int left = 20 + static_cast<int>(index) * 100;
        cv::rectangle(image, cv::Rect{left, 20, 60, 60}, patches[index].second, cv::FILLED);
        if (index != 0)
            records += ",\n";
        records += "    {\"id\": \"" + patches[index].first + "-1\", \"label\": \"" + patches[index].first
                   + "\", \"image\": \"" + generic(reference) + "\", \"regions\": [["
                   + std::to_string(left + 30) + ", 50, 50, 50, 0]]}";
    }
    REQUIRE(cv::imwrite(reference.string(), image), "training image must be writable");
    asset = directory / "colors.json";
    writeText(asset,
              "{\n  \"schema_version\": 1,\n  \"kind\": \"color_samples\",\n  \"records\": [\n" + records
                  + "\n  ]\n}\n");
}

Values colorInputs(const cv::Mat &image, const std::vector<cv::Point2d> &centers, double radius)
{
    ROI regions;
    RegionGroup group;
    for (const auto &center : centers)
        group.push_back(makeCircle(center, radius));
    regions.push_back(std::move(group));

    Values inputs;
    inputs.emplace(color::in::IMAGES, DataRef<Image>(Image{image}));
    inputs.emplace(color::in::ALIGNMENT, geometry::alignmentArray(geometry::Alignment::identity()));
    inputs.emplace(color::in::REGIONS, DataRef<ROI>(std::move(regions)));
    inputs.emplace(color::in::MIN_SCORE, 0.05);
    inputs.emplace(color::in::MIN_RATIO, 0.5);
    return inputs;
}

void colorRecognition(const std::filesystem::path &directory)
{
    std::filesystem::path asset;
    std::filesystem::path reference;
    colorTraining(directory, asset, reference);

    recognition::RecognizeColorTool tool;
    Values                           init;
    init.emplace(color::init::TRAINING, generic(asset));
    const auto trained = tool.initialize(init);
    REQUIRE(trained.code == 0, trained.message);

    const std::vector<cv::Point2d> centers{{50, 200}, {120, 200}, {190, 200}, {260, 200}, {330, 200}};
    const std::vector<cv::Scalar>  order{{128, 0, 128}, {0, 0, 255}, {255, 0, 0}, {0, 255, 255}, {128, 0, 128}};
    cv::Mat                        image(400, 400, CV_8UC3, cv::Scalar{128, 128, 128});
    for (std::size_t index = 0; index < centers.size(); ++index)
        cv::circle(image, centers[index], 15, order[index], cv::FILLED);

    const auto result = tool.execute(colorInputs(image, centers, 15.0));
    REQUIRE(testview::status(result).code == 0, testview::status(result).message);
    const auto colors = visionrt::nestedOf<NestedVector<std::string, 2>>(
        std::get<Array>(result.outputs.at(color::out::COLORS)));
    REQUIRE(colors.size() == 1 && colors.front().size() == 5, "one colour per circle");
    const std::vector<std::string> expected{"purple", "red", "blue", "yellow", "purple"};
    for (std::size_t index = 0; index < expected.size(); ++index)
        REQUIRE(colors.front()[index] == expected[index], "position " + std::to_string(index) + " colour");

    const auto ratios = visionrt::nestedOf<NestedVector<float, 2>>(
        std::get<Array>(result.outputs.at(color::out::RATIOS)));
    REQUIRE(ratios.front().size() == 5 && ratios.front().front() > 0.9, "a pure patch is classified");

    const auto centersX = visionrt::nestedOf<NestedVector<float, 2>>(
        std::get<Array>(result.outputs.at(color::out::CENTER_X)));
    REQUIRE(std::abs(centersX.front()[0] - 50.0f) < 1e-3, "centre X follows the alignment input");

    // An untrained colour must be reported as unknown rather than as its nearest class.
    cv::Mat unknownImage(400, 400, CV_8UC3, cv::Scalar{128, 128, 128});
    for (std::size_t index = 0; index < centers.size(); ++index)
        cv::circle(unknownImage, centers[index], 15, order[index], cv::FILLED);
    cv::circle(unknownImage, centers.back(), 15, cv::Scalar{0, 255, 0}, cv::FILLED);
    const auto unknown = tool.execute(colorInputs(unknownImage, centers, 15.0));
    REQUIRE(testview::status(unknown).code == 0, testview::status(unknown).message);
    const auto unknownColors = visionrt::nestedOf<NestedVector<std::string, 2>>(
        std::get<Array>(unknown.outputs.at(color::out::COLORS)));
    REQUIRE(unknownColors.front().back() == "unknown", "an untrained colour is not forced into a class");

    // A batch keeps the per-image hierarchy: the second image is shifted by one circle and its
    // labels must follow that image, not the first one.
    cv::Mat shifted(400, 400, CV_8UC3, cv::Scalar{128, 128, 128});
    for (std::size_t index = 0; index < centers.size(); ++index)
        cv::circle(shifted, centers[index], 15, order[(index + 1) % order.size()], cv::FILLED);
    Values batch;
    batch.emplace(color::in::IMAGES, DataRef<Image>(Image{image, shifted}));
    batch.emplace(color::in::ALIGNMENT,
                  geometry::alignmentArray(geometry::Alignment::identity()));
    batch.emplace(color::in::REGIONS, colorInputs(image, centers, 15.0).at(color::in::REGIONS));
    batch.emplace(color::in::MIN_SCORE, 0.05);
    batch.emplace(color::in::MIN_RATIO, 0.5);
    const auto batchResult = tool.execute(batch);
    REQUIRE(testview::status(batchResult).code == 0, testview::status(batchResult).message);
    const auto batchColors = visionrt::nestedOf<NestedVector<std::string, 2>>(
        std::get<Array>(batchResult.outputs.at(color::out::COLORS)));
    REQUIRE(batchColors.size() == 2, "one colour list per image");
    for (std::size_t index = 0; index < expected.size(); ++index)
        REQUIRE(batchColors[1][index] == expected[(index + 1) % expected.size()],
                "second image keeps its own index order");
}

void assetValidation(const std::filesystem::path &directory)
{
    const auto brokenAlignment = directory / "broken_alignment.json";
    writeText(brokenAlignment,
              "{\n  \"schema_version\": 1,\n  \"kind\": \"straight_edges\",\n"
              "  \"reference_image\": \"none.png\",\n"
              "  \"segments\": [{\"name\": \"one\", \"from\": [0, 0], \"to\": [10, 0], \"search\": [5, 0, 10, 10, 0]}]\n}\n");
    recognition::AlignStraightBordersTool tool;
    Values                                init;
    init.emplace(align::init::TRAINING, generic(brokenAlignment));
    const auto status = tool.initialize(init);
    REQUIRE(status.code == static_cast<std::int64_t>(recognition::RecognitionError::ReferenceSegmentMissing),
            "one reference edge is not enough");

    const auto brokenColor = directory / "broken_color.json";
    writeText(brokenColor,
              "{\n  \"schema_version\": 1,\n  \"kind\": \"color_samples\",\n  \"records\": []\n}\n");
    recognition::RecognizeColorTool colorTool;
    Values                           colorInit;
    colorInit.emplace(color::init::TRAINING, generic(brokenColor));
    const auto colorStatus = colorTool.initialize(colorInit);
    REQUIRE(colorStatus.code == static_cast<std::int64_t>(recognition::RecognitionError::ColorSamplesMissing),
            "an empty training set is rejected");
}
} // namespace

int main()
{
    const auto directory = fixtureDirectory();
    straightEdgeAlignment(directory);
    colorRecognition(directory);
    straightEdgeProbe();
    assetValidation(directory);
    std::cout << "[Recognition Contract] Straight-edge pose and U/V colour classification verified.\n";
    return 0;
}
