// Recognition contract: the straight-edge alignment must recover a known reference-to-current pose,
// and the colour tool must return the trained class per detection circle while rejecting colours it
// never saw. Fixtures are synthetic so every expectation has an analytic answer.
#include "ToolResultAssertions.h"
#include "common/EdgeProfile.h"
#include "common/GeometryTransform.h"
#include "common/RoiSupport.h"
#include "recognition/AlignStraightBordersTool.h"
#include "recognition/ColorModel.h"
#include "recognition/LocateStraightEdgeTool.h"
#include "recognition/RecognizeColorTool.h"
#include "recognition/StraightEdgeProbe.h"
#include "recognition/errors/RecognitionErrors.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define REQUIRE(cond, message)                                                                                      \
    do                                                                                                              \
    {                                                                                                               \
        if (!(cond))                                                                                                \
        {                                                                                                           \
            std::cerr << "Assertion failed: " << #cond << " (" << message << ") at line " << __LINE__ << std::endl; \
            std::exit(1);                                                                                           \
        }                                                                                                           \
    }                                                                                                               \
    while (0)

using namespace visionrt;
using namespace visiontools;

namespace {
namespace align = recognition::ports::align_straight_borders;
namespace train = recognition::ports::align_straight_borders;
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

// The two bands a user draws on the reference picture: the long side follows the edge, the short
// side is how far across it the locator may look. The trainer turns them into the published model
// the alignment node loads, so this is the whole training half of the contract.
ToolResult trainEdges(const cv::Mat &reference, std::array<cv::Vec4d, 2> bands)
{
    TrainingSet set;
    set.images.emplace(1, trainingFrame(1, reference, ColorOrder::BGR));
    ROI         regions;
    RegionGroup group;
    for (const auto &band : bands) group.push_back(makeRectangle(band[0], band[1], band[2], 40, band[3]));
    regions.push_back(std::move(group));
    TrainingRecord record;
    record.id      = 1;
    record.imageId = 1;
    record.fields.emplace(2, Rectangle::fromEncoded(std::move(regions)));
    set.records.emplace(record.id, std::move(record));

    recognition::AlignStraightBordersTool trainer;

    Values inputs;
    inputs.emplace(train::train::MODEL_PATH, std::string{});
    inputs.emplace(train::train::CONTRAST, 30.0);
    inputs.emplace(train::train::EDGE_WIDTH, std::int64_t{3});
    inputs.emplace(train::train::POLARITY, std::string("any"));
    inputs.emplace(train::train::MIN_COMPLETENESS, 0.4);
    inputs.emplace(train::train::DATASET, DataRef<TrainingSet>(set));
    return trainer.train(inputs);
}

std::filesystem::path straightEdgeModel(const std::filesystem::path &directory,
                                        std::array<cv::Vec4d, 2>     bands = {
                                            cv::Vec4d{120, 280, 160,  0},
                                            cv::Vec4d{220, 180, 160, 90}
})
{
    const auto result = trainEdges(connectorImage(), bands);
    REQUIRE(testview::status(result).code == 0, testview::status(result).message);

    const auto &bytes = result.artifacts.at("edge-model").bytes;
    REQUIRE(bytes.size() > 16, "the published edge model cannot be empty");

    // The trainer also answers with what it measured, in reference coordinates: the training page
    // draws those lines and the host never derives an endpoint itself.
    const auto &found = std::get<Line>(result.outputs.at(align::out::FOUND_SEGMENTS)).get();
    REQUIRE(found.size() == 1 && found.front().size() == 2, "one measured line per drawn band");
    for (const auto &segment : found.front())
        REQUIRE(visionrt::RegionView(segment).kind() == visionrt::RoiKind::Segment,
                "a measured reference edge is a segment");

    std::error_code ignored;
    std::filesystem::create_directories(directory, ignored);
    const auto    path = directory / "edge.model";
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    REQUIRE(file.good(), "edge model must be writable");
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
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
    inputs.emplace(6, Rectangle::fromEncoded(ROI{
                          {makeRectangle(120, 280, 160, 40), makeRectangle(220, 180, 160, 40, 90)}
    }));
    return inputs;
}

void straightEdgeAlignment(const std::filesystem::path &directory)
{
    const auto modelPath = straightEdgeModel(directory);
    const auto reference = connectorImage();

    recognition::AlignStraightBordersTool tool;
    ToolConfiguration                                init;
    init.models.emplace("edge-model", generic(modelPath));
    REQUIRE(tool.initialize(init).code == 0, "the published edge model must load");

    const auto identity = tool.execute(alignmentInputs(reference, "any"));
    REQUIRE(testview::status(identity).code == 0, testview::status(identity).message);
    const auto identityMatrices
        = visionrt::nestedOf<NestedVector<double, 2>>(std::get<Array>(identity.outputs.at(align::out::ALIGNMENTS)));
    REQUIRE(identityMatrices.size() == 1 && identityMatrices.front().size() == 6, "one six-coefficient matrix per image");
    const auto &matrix = identityMatrices.front();
    // A filled rectangle covers whole pixels, so its intensity step sits half a pixel inside the
    // geometric border; the pose must reproduce the image up to that quantisation.
    const auto  identityError = [&](int row, int column)
    {
        return std::abs(matrix[row * 3 + column] - (row == column ? 1.0 : 0.0));
    };
    REQUIRE(identityError(0, 0) < 1e-3 && identityError(1, 1) < 1e-3, "reference image keeps its axes");
    REQUIRE(identityError(0, 1) < 1e-3 && identityError(1, 0) < 1e-3, "reference image keeps its rotation");
    REQUIRE(std::hypot(matrix[2], matrix[5]) < 0.75, "reference image keeps its position");

    auto offTarget = alignmentInputs(reference, "any");
    offTarget[6]   = Rectangle::fromEncoded(ROI{
          {makeRectangle(5, 5, 40, 10), makeRectangle(5, 5, 40, 10, 90)}
    });
    REQUIRE(testview::status(tool.execute(offTarget)).code != 0,
            "runtime search regions must take effect without retraining");
    REQUIRE(testview::status(tool.execute(alignmentInputs(reference, "any"))).code == 0,
            "editing search regions must leave the trained reference intact");

    // A5: moving a search band along its own normal changes where the locator looks, not what it
    // measures. Both directions must still recover the reference pose instead of following the band.
    for (const double shift : {-10.0, 10.0})
    {
        const auto shifted
            = straightEdgeModel(directory / (shift < 0 ? "shift-left" : "shift-right"),
                                {
                                    cv::Vec4d{        120, 280 + shift, 160,  0},
                                    cv::Vec4d{220 + shift,         180, 160, 90}
        });
        recognition::AlignStraightBordersTool shiftedTool;
        ToolConfiguration                                shiftedInit;
        shiftedInit.models.emplace("edge-model", generic(shifted));
        REQUIRE(shiftedTool.initialize(shiftedInit).code == 0, "the shifted-band model must load");
        const auto posed = shiftedTool.execute(alignmentInputs(reference, "any"));
        REQUIRE(testview::status(posed).code == 0, testview::status(posed).message);
        const auto shiftedMatrices
            = visionrt::nestedOf<NestedVector<double, 2>>(std::get<Array>(posed.outputs.at(align::out::ALIGNMENTS)));
        const auto &shiftedPose = shiftedMatrices.front();
        REQUIRE(std::abs(shiftedPose[1]) < 1e-3 && std::abs(shiftedPose[3]) < 1e-3,
                "a band offset must not tilt the reference pose");
        REQUIRE(std::hypot(shiftedPose[2] - matrix[2], shiftedPose[5] - matrix[5]) < 0.75,
                "a band offset must not move the reference pose");
    }

    // The part moves: rotation about the centre plus a shift. The tool must report that pose, with
    // x as column and y as row, and it must not swap the two.
    const auto expected = pose(12.0, -7.0, 5.0);
    cv::Mat    moved;
    cv::Mat    transform(2, 3, CV_64F);
    for (int row = 0; row < 2; ++row)
        for (int column = 0; column < 3; ++column) transform.at<double>(row, column) = expected.m[row * 3 + column];
    cv::warpAffine(reference, moved, transform, reference.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar{0, 0, 0});

    const auto result = tool.execute(alignmentInputs(moved, "any"));
    REQUIRE(testview::status(result).code == 0, testview::status(result).message);
    const auto matrices
        = visionrt::nestedOf<NestedVector<double, 2>>(std::get<Array>(result.outputs.at(align::out::ALIGNMENTS)));
    REQUIRE(matrices.size() == 1, "one matrix for one image");
    const geometry::Alignment recovered = *geometry::Alignment::fromRows(matrices.front());
    REQUIRE(recovered.isSimilarity(), "a straight-edge pose is a similarity");

    for (const cv::Point2d probe : {
             cv::Point2d{200, 300},
             cv::Point2d{300, 200},
             cv::Point2d{150, 250}
    })
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
    REQUIRE(testview::businessCount(mismatch) == 4, "a failed alignment still publishes every port as Missing");
    const recognition::ReferenceSegment firstReference{
        "first",
        {   0, 0},
        {1000, 0}
    };
    const recognition::ReferenceSegment secondReference{
        "second",
        {0,    0},
        {0, 1000}
    };
    REQUIRE(!recognition::rigidTransform(firstReference,
                                         {
                                             {   0, 0},
                                             {1000, 0},
                                             1
    },
                                         secondReference, {{0, 5}, {1000, 5.1}, 1}),
            "nearly parallel measured edges cannot determine a rigid pose");
}

void straightEdgeProbe()
{
    // Point-and-click feeds the same training asset the user would otherwise draw by hand: the band
    // must follow the edge under the cursor, and empty space must answer "nothing here".
    const auto image = connectorImage();
    const auto gray  = edge::grayscale(image);

    recognition::EdgeProbeRequest request;
    request.searchCenter = {200.0, 300.0};
    request.searchRadius = 96;
    const auto bottom    = recognition::probeStraightEdge(gray, request);
    REQUIRE(bottom.valid, bottom.quality.c_str());
    REQUIRE(bottom.proposal.size() == 1, "one band per click");
    const auto rectangle = visionrt::RegionView(bottom.proposal.front()).rectangle();
    REQUIRE(std::abs(rectangle.center().y - 300.0) < 2.0, "the band sits on the edge under the cursor");
    REQUIRE(std::abs(std::remainder(rectangle.angle(), 180.0)) < 2.0, "a horizontal edge gives a level band");
    REQUIRE(rectangle.width() >= 96.0, "the band spans the probe window along the edge");

    // The same click on the vertical edge must rotate the band, not copy the first answer.
    request.searchCenter = {300.0, 200.0};
    const auto right     = recognition::probeStraightEdge(gray, request);
    REQUIRE(right.valid, right.quality.c_str());
    const auto vertical = visionrt::RegionView(right.proposal.front()).rectangle();
    REQUIRE(std::abs(std::abs(std::remainder(vertical.angle(), 180.0)) - 90.0) < 2.0,
            "a vertical edge gives a quarter-turn band");

    // Empty space has no structure tensor, so nothing is proposed.
    request.searchCenter = {40.0, 40.0};
    const auto empty     = recognition::probeStraightEdge(gray, request);
    REQUIRE(!empty.valid && !empty.quality.empty(), "empty space must not invent an edge");
    REQUIRE(empty.proposal.empty(), "an invalid probe carries no proposal");
}

void edgeParameterCalibration()
{
    recognition::EdgeProbeRequest request;
    request.searchCenter   = {200, 300};
    request.searchRadius   = 96;
    const auto reference   = connectorImage();
    const auto sharp       = edge::grayscale(reference);
    const auto sharpResult = recognition::probeStraightEdge(sharp, request);
    REQUIRE(sharpResult.valid, sharpResult.quality.c_str());
    cv::Mat blurred;
    cv::GaussianBlur(sharp, blurred, {0, 0}, 3.0);
    const auto blurredResult = recognition::probeStraightEdge(blurred, request);
    REQUIRE(blurredResult.valid, blurredResult.quality.c_str());
    REQUIRE(blurredResult.width > sharpResult.width, "broader edge responses require a larger averaging width");
    REQUIRE(blurredResult.contrast < sharpResult.contrast,
            "a blurred step has less contrast at its selected smoothing scale");
    cv::Mat inverted;
    cv::bitwise_not(sharp, inverted);
    const auto reversed = recognition::probeStraightEdge(inverted, request);
    REQUIRE(reversed.valid && reversed.polarity != sharpResult.polarity,
            "normal polarity follows the actual intensity transition");
    request.parameterRegions = sharpResult.proposal;
    request.parameterRegions.front().back() += 180.0;
    request.searchCenter = {200, 300};
    const auto combined  = recognition::probeStraightEdge(sharp, request);
    REQUIRE(combined.valid && combined.polarity == "any",
            "opposite normals in the full region set require unrestricted polarity");
}

std::filesystem::path colorTraining(const std::filesystem::path &directory)
{
    cv::Mat                                               reference(400, 400, CV_8UC3, cv::Scalar{128, 128, 128});
    const std::vector<std::pair<std::string, cv::Scalar>> patches{
        {"purple", {128, 0, 128}},
        {   "red",   {0, 0, 255}},
        {  "blue",   {255, 0, 0}},
        {"yellow", {0, 255, 255}},
    };
    TrainingSet      set;
    TrainingRecordId nextRecord = 1;
    for (std::size_t index = 0; index < patches.size(); ++index)
    {
        const int left = 20 + static_cast<int>(index) * 100;
        cv::rectangle(reference, cv::Rect{left, 20, 60, 60}, patches[index].second, cv::FILLED);
    }
    const std::vector<cv::Point2d> triangle{
        {220, 300},
        {280, 300},
        {250, 350}
    };
    cv::fillConvexPoly(reference,
                       std::vector<cv::Point>{
                           cv::Point{220, 300},
                           cv::Point{280, 300},
                           cv::Point{250, 350}
    },
                       cv::Scalar{128, 0, 128}, cv::LINE_8);
    cv::circle(reference, cv::Point{330, 300}, 25, cv::Scalar{0, 255, 255}, cv::FILLED);
    cv::Mat second = reference.clone();
    cv::rectangle(second, cv::Rect{20, 300, 60, 60}, cv::Scalar{128, 0, 128}, cv::FILLED);
    set.images.emplace(1, trainingFrame(1, reference, ColorOrder::BGR));
    set.images.emplace(2, trainingFrame(2, second, ColorOrder::BGR));
    for (std::size_t index = 0; index < patches.size(); ++index)
    {
        const int   left = 20 + static_cast<int>(index) * 100;
        ROI         regions;
        RegionGroup group;
        group.push_back(makeRectangle(left, 20, 60, 60, 0));
        regions.push_back(std::move(group));
        TrainingRecord record;
        record.id      = nextRecord++;
        record.imageId = 1;
        record.fields.emplace(1, patches[index].first);
        record.fields.emplace(2, Rectangle::fromEncoded(std::move(regions)));
        set.records.emplace(record.id, std::move(record));
    }
    // Repeated rectangle samples on one frame contribute to the same class.
    ROI         polygonRegions;
    RegionGroup polygonGroup;
    polygonGroup.push_back(makeRectangle(246, 310, 8, 8));
    polygonRegions.push_back(std::move(polygonGroup));
    TrainingRecord polygonRecord;
    polygonRecord.id      = nextRecord++;
    polygonRecord.imageId = 1;
    polygonRecord.fields.emplace(1, std::string("purple"));
    polygonRecord.fields.emplace(2, Rectangle::fromEncoded(std::move(polygonRegions)));
    set.records.emplace(polygonRecord.id, std::move(polygonRecord));

    ROI         circleRegions;
    RegionGroup circleGroup;
    circleGroup.push_back(makeRectangle(320, 290, 20, 20));
    circleRegions.push_back(std::move(circleGroup));
    TrainingRecord circleRecord;
    circleRecord.id      = nextRecord++;
    circleRecord.imageId = 1;
    circleRecord.fields.emplace(1, std::string("yellow"));
    circleRecord.fields.emplace(2, Rectangle::fromEncoded(std::move(circleRegions)));
    set.records.emplace(circleRecord.id, std::move(circleRecord));

    // The same label on a second frame contributes to the one class rather than a second model.
    ROI         secondRegions;
    RegionGroup secondGroup;
    secondGroup.push_back(makeRectangle(20, 300, 60, 60, 0));
    secondRegions.push_back(std::move(secondGroup));
    TrainingRecord secondRecord;
    secondRecord.id      = nextRecord++;
    secondRecord.imageId = 2;
    secondRecord.fields.emplace(1, std::string("purple"));
    secondRecord.fields.emplace(2, Rectangle::fromEncoded(std::move(secondRegions)));
    set.records.emplace(secondRecord.id, std::move(secondRecord));

    recognition::RecognizeColorTool trainer;
    Values                           inputs;
    inputs.emplace(color::train::MODEL_PATH, std::string{});
    inputs.emplace(color::train::DATASET, DataRef<TrainingSet>(set));
    const auto result = trainer.train(inputs);
    REQUIRE(testview::status(result).code == 0, testview::status(result).message);
    const auto &bytes = result.artifacts.at("color-model").bytes;
    REQUIRE(!bytes.empty(), "the trainer returned no model bytes");
    const auto decoded = recognition::ColorModel::decode(bytes);
    REQUIRE(decoded.classCount() == 4, "rectangle samples with the same label aggregate into one class");
    const auto again = decoded.encode();
    REQUIRE(again.size() == bytes.size() && std::equal(again.begin(), again.end(), bytes.begin()),
            "the model byte format round-trips without changing a class");

    const auto    path = directory / "color.model";
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    REQUIRE(file.good(), "colour model must be writable");
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    file.close();

    const auto    truncated = directory / "color_truncated.model";
    std::ofstream cut(truncated, std::ios::binary | std::ios::trunc);
    REQUIRE(cut.good(), "truncated colour model must be writable");
    cut.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size() / 2));
    cut.close();
    return path;
}

Values colorInputs(const cv::Mat &image, const std::vector<cv::Point2d> &centers, double radius)
{
    ROI         regions;
    RegionGroup group;
    for (const auto &center : centers) group.push_back(makeCircle(center, radius));
    regions.push_back(std::move(group));

    Values inputs;
    inputs.emplace(color::in::IMAGES, DataRef<Image>(Image{image}));
    inputs.emplace(color::in::ALIGNMENT, geometry::alignmentArray(geometry::Alignment::identity()));
    inputs.emplace(color::in::REGIONS, Circle::fromEncoded(std::move(regions)));
    inputs.emplace(color::in::MIN_SCORE, 0.05);
    inputs.emplace(color::in::MIN_RATIO, 0.5);
    return inputs;
}

void colorRecognition(const std::filesystem::path &directory)
{
    const auto modelPath = colorTraining(directory);

    recognition::RecognizeColorTool tool;
    ToolConfiguration                          init;
    init.models.emplace("color-model", generic(modelPath));
    const auto trained = tool.initialize(init);
    REQUIRE(trained.code == 0, trained.message);

    const std::vector<cv::Point2d> centers{
        { 50, 200},
        {120, 200},
        {190, 200},
        {260, 200},
        {330, 200}
    };
    const std::vector<cv::Scalar> order{
        {128,   0, 128},
        {  0,   0, 255},
        {255,   0,   0},
        {  0, 255, 255},
        {128,   0, 128}
    };
    cv::Mat image(400, 400, CV_8UC3, cv::Scalar{128, 128, 128});
    for (std::size_t index = 0; index < centers.size(); ++index)
        cv::circle(image, centers[index], 15, order[index], cv::FILLED);

    const auto result = tool.execute(colorInputs(image, centers, 15.0));
    REQUIRE(testview::status(result).code == 0, testview::status(result).message);
    const auto colors
        = visionrt::nestedOf<NestedVector<std::string, 2>>(std::get<Array>(result.outputs.at(color::out::COLORS)));
    REQUIRE(colors.size() == 1 && colors.front().size() == 5, "one colour per circle");
    const std::vector<std::string> expected{"purple", "red", "blue", "yellow", "purple"};
    for (std::size_t index = 0; index < expected.size(); ++index)
        REQUIRE(colors.front()[index] == expected[index], "position " + std::to_string(index) + " colour");

    const auto ratios
        = visionrt::nestedOf<NestedVector<float, 2>>(std::get<Array>(result.outputs.at(color::out::RATIOS)));
    REQUIRE(ratios.front().size() == 5 && ratios.front().front() > 0.9, "a pure patch is classified");
    const auto scores
        = visionrt::nestedOf<NestedVector<float, 2>>(std::get<Array>(result.outputs.at(color::out::SCORES)));

    // The bytes written by the trainer and the model the tool loaded must answer identically to
    // the model decoded in memory, including the float scores and ratios the line exposes.
    {
        std::ifstream file(modelPath, std::ios::binary);
        REQUIRE(file.good(), "the published colour model must be readable");
        const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        const auto                      inMemory = recognition::ColorModel::decode(bytes);
        const recognition::ChromaImage  chroma(image);
        for (std::size_t index = 0; index < centers.size(); ++index)
        {
            const cv::Mat mask
                = roi::regionMask({makeCircle(centers[index], 15.0)}, image.size(), geometry::Alignment::identity());
            const auto answer = inMemory.classify(chroma, mask, 0.05, 0.5);
            REQUIRE(answer.label == colors.front()[index], "the reloaded model changes the label");
            REQUIRE(std::abs(answer.score - scores.front()[index]) < 1e-6, "the reloaded model changes the score");
            REQUIRE(std::abs(answer.ratio - ratios.front()[index]) < 1e-6, "the reloaded model changes the ratio");
        }
    }

    const auto centersX
        = visionrt::nestedOf<NestedVector<float, 2>>(std::get<Array>(result.outputs.at(color::out::CENTER_X)));
    REQUIRE(std::abs(centersX.front()[0] - 50.0f) < 1e-3, "centre X follows the alignment input");

    // An untrained colour must be reported as unknown rather than as its nearest class.
    cv::Mat unknownImage(400, 400, CV_8UC3, cv::Scalar{128, 128, 128});
    for (std::size_t index = 0; index < centers.size(); ++index)
        cv::circle(unknownImage, centers[index], 15, order[index], cv::FILLED);
    cv::circle(unknownImage, centers.back(), 15, cv::Scalar{0, 255, 0}, cv::FILLED);
    const auto unknown = tool.execute(colorInputs(unknownImage, centers, 15.0));
    REQUIRE(testview::status(unknown).code == 0, testview::status(unknown).message);
    const auto unknownColors
        = visionrt::nestedOf<NestedVector<std::string, 2>>(std::get<Array>(unknown.outputs.at(color::out::COLORS)));
    REQUIRE(unknownColors.front().back() == "unknown", "an untrained colour is not forced into a class");

    // A batch keeps the per-image hierarchy: the second image is shifted by one circle and its
    // labels must follow that image, not the first one.
    cv::Mat shifted(400, 400, CV_8UC3, cv::Scalar{128, 128, 128});
    for (std::size_t index = 0; index < centers.size(); ++index)
        cv::circle(shifted, centers[index], 15, order[(index + 1) % order.size()], cv::FILLED);
    Values batch;
    batch.emplace(color::in::IMAGES, DataRef<Image>(Image{image, shifted}));
    batch.emplace(color::in::ALIGNMENT, geometry::alignmentArray(geometry::Alignment::identity()));
    batch.emplace(color::in::REGIONS, colorInputs(image, centers, 15.0).at(color::in::REGIONS));
    batch.emplace(color::in::MIN_SCORE, 0.05);
    batch.emplace(color::in::MIN_RATIO, 0.5);
    const auto batchResult = tool.execute(batch);
    REQUIRE(testview::status(batchResult).code == 0, testview::status(batchResult).message);
    const auto batchColors
        = visionrt::nestedOf<NestedVector<std::string, 2>>(std::get<Array>(batchResult.outputs.at(color::out::COLORS)));
    REQUIRE(batchColors.size() == 2, "one colour list per image");
    for (std::size_t index = 0; index < expected.size(); ++index)
        REQUIRE(batchColors[1][index] == expected[(index + 1) % expected.size()],
                "second image keeps its own index order");
}

void assetValidation(const std::filesystem::path &directory)
{
    // A model that cannot be read back is rejected by its own code, never retrained from a band.
    const auto corruptModel = directory / "corrupt_edge.model";
    writeText(corruptModel, "not an edge model");
    recognition::AlignStraightBordersTool tool;
    ToolConfiguration                                init;
    init.models.emplace("edge-model", generic(corruptModel));
    const auto status = tool.initialize(init);
    REQUIRE(status.code == static_cast<std::int64_t>(recognition::RecognitionError::StraightEdgeModelIncompatible),
            "a damaged edge model is rejected");

    ToolConfiguration missing;
    missing.models.emplace("edge-model", std::string{});
    REQUIRE(tool.initialize(missing).code
                == static_cast<std::int64_t>(recognition::RecognitionError::StraightEdgeModelPathMissing),
            "an unconfigured model path has its own diagnosis");

    // A model of one reference must not claim a match on a picture that has no such edge.
    const auto trained = straightEdgeModel(directory);
    ToolConfiguration     good;
    good.models.emplace("edge-model", generic(trained));
    REQUIRE(tool.initialize(good).code == 0, "the trained model must load");
    const auto blank = tool.execute(alignmentInputs(cv::Mat(400, 400, CV_8UC3, cv::Scalar{0, 0, 0}), "any"));
    REQUIRE(testview::status(blank).code == static_cast<std::int64_t>(recognition::RecognitionError::EdgeNotFound),
            "a missing edge is reported instead of an invented pose");
    REQUIRE(testview::status(blank).message.find("第 1 条参考边") != std::string::npos,
            "the failure names the region that found no edge");

    // Two parallel bands can be found but never fix a pose; training refuses them instead of
    {
        cv::Mat                      oblique(400, 400, CV_8UC3, cv::Scalar{0, 0, 0});
        const std::vector<cv::Point> triangle{
            {100,  60},
            {300, 300},
            { 40, 300}
        };
        cv::fillConvexPoly(oblique, triangle, cv::Scalar{255, 255, 255});
        const double angle  = std::atan2(240.0, 200.0) * 180.0 / 3.14159265358979323846;
        const auto   fitted = trainEdges(oblique, {
                                                    cv::Vec4d{ 80, 280, 160,     0},
                                                    cv::Vec4d{130, 160, 140, angle}
        });
        REQUIRE(testview::status(fitted).code == 0, "two non-right-angle edges must train a valid pose model");
        const auto model = fitted.artifacts.at("edge-model").bytes;
        const auto path  = directory / "oblique.model";
        {
            std::ofstream file(path, std::ios::binary);
            file.write(reinterpret_cast<const char *>(model.data()), model.size());
        }
        recognition::AlignStraightBordersTool obliqueTool;
        REQUIRE(obliqueTool
                        .initialize(ToolConfiguration{{}, {{"edge-model", path}}})
                        .code
                    == 0,
                "the non-right-angle model must load");
        auto search         = alignmentInputs(oblique, "any");
        search[6]           = Rectangle::fromEncoded(ROI{
                      {makeRectangle(80, 280, 160, 40), makeRectangle(130, 160, 140, 40, angle)}
        });
        const auto identity = obliqueTool.execute(search);
        REQUIRE(testview::status(identity).code == 0,
                "non-right-angle reference edges must locate their reference image");
        const auto matrix = std::get<Array>(identity.outputs.at(align::out::ALIGNMENTS)).elements<double>();
        REQUIRE(std::hypot(matrix[2], matrix[5]) <= 0.75
                    && std::abs(std::atan2(matrix[3], matrix[0])) <= 0.05 * 3.14159265358979323846 / 180.0,
                "the non-right-angle reference pose must remain identity");
    }

    // Two parallel bands can be found but never fix a pose; training refuses them instead of
    // publishing a model that would silently rotate by an arbitrary angle.
    {
        const auto parallel
            = trainEdges(connectorImage(), {
                                               cv::Vec4d{120, 280, 160, 0},
                                               cv::Vec4d{120,  80, 160, 0}
        });
        REQUIRE(testview::status(parallel).code
                    == static_cast<std::int64_t>(recognition::RecognitionError::ReferenceSegmentsParallel),
                "two parallel edges are refused at training time");
    }
    // A band over background finds no edge at all, and the message names which one.
    {
        const auto missingEdge
            = trainEdges(connectorImage(), {
                                               cv::Vec4d{120, 280, 160,  0},
                                               cv::Vec4d{220, 380, 160, 90}
        });
        REQUIRE(testview::status(missingEdge).code
                    == static_cast<std::int64_t>(recognition::RecognitionError::EdgeNotFound),
                "a band without an edge is refused");
        REQUIRE(testview::status(missingEdge).message.find("第 2 条参考边") != std::string::npos,
                "the refusal names the band that found nothing");
    }
    // One band is not a pair: the declaration already caps the field, and the trainer repeats the
    // rule so a hand-written asset cannot slip through.
    {
        TrainingSet set;
        set.images.emplace(1, trainingFrame(1, connectorImage(), ColorOrder::BGR));
        ROI         regions;
        RegionGroup group;
        group.push_back(makeRectangle(120, 280, 160, 40, 0));
        regions.push_back(std::move(group));
        TrainingRecord record;
        record.id      = 1;
        record.imageId = 1;
        record.fields.emplace(2, Rectangle::fromEncoded(std::move(regions)));
        set.records.emplace(record.id, std::move(record));

        recognition::AlignStraightBordersTool trainer;
        trainer.initialize({});
        Values inputs;
        inputs.emplace(train::train::MODEL_PATH, std::string{});
        inputs.emplace(train::train::CONTRAST, 30.0);
        inputs.emplace(train::train::EDGE_WIDTH, std::int64_t{3});
        inputs.emplace(train::train::POLARITY, std::string("any"));
        inputs.emplace(train::train::MIN_COMPLETENESS, 0.4);
        inputs.emplace(train::train::DATASET, DataRef<TrainingSet>(set));
        const auto single = trainer.train(inputs);
        REQUIRE(testview::status(single).code
                    == static_cast<std::int64_t>(recognition::RecognitionError::StraightEdgeRegionsInvalid),
                "one band cannot produce a pose");
    }

    recognition::RecognizeColorTool trainer;
    Values                           empty;
    TrainingSet                      noRecords;
    empty.emplace(color::train::MODEL_PATH, std::string{});
            empty.emplace(color::train::DATASET, DataRef<TrainingSet>(noRecords));
    const auto emptyResult = trainer.train(empty);
    REQUIRE(testview::status(emptyResult).code
                == static_cast<std::int64_t>(recognition::RecognitionError::ColorSamplesMissing),
            "an empty training set is rejected");

    TrainingSet unsupported;
    unsupported.images.emplace(1, trainingFrame(1, cv::Mat(80, 80, CV_16UC3, cv::Scalar{1, 2, 3}), ColorOrder::BGR));
    ROI         unsupportedRegions;
    RegionGroup unsupportedGroup;
    unsupportedGroup.push_back(makeRectangle(10, 10, 40, 40, 0));
    unsupportedRegions.push_back(std::move(unsupportedGroup));
    TrainingRecord unsupportedRecord;
    unsupportedRecord.id      = 1;
    unsupportedRecord.imageId = 1;
    unsupportedRecord.fields.emplace(1, std::string("wide"));
    unsupportedRecord.fields.emplace(2, Rectangle::fromEncoded(std::move(unsupportedRegions)));
    unsupported.records.emplace(1, std::move(unsupportedRecord));
    Values wide;
    wide.emplace(color::train::MODEL_PATH, std::string{});
            wide.emplace(color::train::DATASET, DataRef<TrainingSet>(unsupported));
    const auto wideResult = trainer.train(wide);
    REQUIRE(testview::status(wideResult).code
                == static_cast<std::int64_t>(recognition::RecognitionError::UnsupportedColorImage),
            "training accepts only 8-bit three-channel frames");

    const auto brokenColor = directory / "broken_color.model";
    writeText(brokenColor, "not a colour model");
    recognition::RecognizeColorTool colorTool;
    ToolConfiguration                          colorInit;
    colorInit.models.emplace("color-model", generic(brokenColor));
    const auto colorStatus = colorTool.initialize(colorInit);
    REQUIRE(colorStatus.code == static_cast<std::int64_t>(recognition::RecognitionError::ColorModelIncompatible),
            "a corrupt model is rejected instead of retrained");

    ToolConfiguration truncatedInit;
    truncatedInit.models.emplace("color-model", generic(directory / "color_truncated.model"));
    const auto truncatedStatus = colorTool.initialize(truncatedInit);
    REQUIRE(truncatedStatus.code == static_cast<std::int64_t>(recognition::RecognitionError::ColorModelIncompatible),
            "a truncated model is rejected by its length check");

    ToolConfiguration missingInit;
    missingInit.models.emplace("color-model", generic(directory / "absent.model"));
    const auto missingStatus = colorTool.initialize(missingInit);
    REQUIRE(missingStatus.code == static_cast<std::int64_t>(recognition::RecognitionError::ColorModelUnreadable),
            "a missing model reports its own path problem");

    const auto noPath = colorTool.initialize(Values{});
    REQUIRE(noPath.code == static_cast<std::int64_t>(recognition::RecognitionError::ColorModelPathMissing),
            "an unconfigured model port is named instead of silently training");
}
} // namespace

int main()
{
    const auto directory = fixtureDirectory();
    straightEdgeAlignment(directory);
    colorRecognition(directory);
    straightEdgeProbe();
    edgeParameterCalibration();
    assetValidation(directory);
    std::cout << "[Recognition Contract] Straight-edge pose and U/V colour classification verified.\n";
    return 0;
}
