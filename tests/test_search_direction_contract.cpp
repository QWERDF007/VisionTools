#include "recognition/AlignStraightBordersTool.h"
#include "ToolResultAssertions.h"

#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace visionrt;
using namespace visiontools;
namespace p = recognition::ports::align_straight_borders;

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

int main()
try
{
    cv::Mat image(400, 400, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::rectangle(image, cv::Rect(100, 100, 200, 200), cv::Scalar(255, 255, 255), cv::FILLED);
    const ROI regions{{makeRectangle(120, 280, 160, 40), makeRectangle(220, 180, 160, 40, 90)}};
    TrainingSet set;
    set.images.emplace(1, trainingFrame(1, image, ColorOrder::BGR));
    set.records.emplace(1, TrainingRecord{1, 1, {{p::field::SEARCH_REGIONS, Rectangle::fromEncoded(regions)}}});
    recognition::AlignStraightBordersTool tool;
    const auto trained = tool.train({{p::train::DATASET, DataRef<TrainingSet>(set)}, {p::train::MODEL_PATH, std::string{}},
        {p::train::CONTRAST, 30.0}, {p::train::EDGE_WIDTH, std::int64_t{3}},
        {p::train::POLARITY, std::string("any")}, {p::train::MIN_COMPLETENESS, 0.4}});
    require(static_cast<bool>(testview::status(trained)), "real edge training failed");
    const auto check = [](const ToolResult &result)
    {
        const auto &directions = std::get<Direction>(result.outputs.at(p::out::SEARCH_DIRECTIONS));
        require(directions.images() == 1 && directions.count(0) == 2, "two directions in search region order");
        require(cv::norm(directions.at(0, 0).start() - cv::Point2d(200, 280)) < 1e-9, "horizontal edge search starts at negative normal end");
        require(cv::norm(directions.at(0, 0).end() - cv::Point2d(200, 320)) < 1e-9, "horizontal search spans full sampling length");
        require(cv::norm(directions.at(0, 1).start() - cv::Point2d(320, 200)) < 1e-9, "vertical edge searches towards decreasing x");
        require(cv::norm(directions.at(0, 1).end() - cv::Point2d(280, 200)) < 1e-9, "vertical edge search end");
    };
    check(trained);
    const auto modelFile = std::filesystem::temp_directory_path()
        / ("visiontools-direction-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
    const auto &bytes = trained.artifacts.at(p::MODEL_ARTIFACT).bytes;
    { std::ofstream file(modelFile, std::ios::binary); file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size()); }
    ToolConfiguration configuration;
    configuration.models.emplace(p::MODEL_ARTIFACT, modelFile);
    require(static_cast<bool>(tool.initialize(configuration)), "trained model loads");
    const auto result = tool.execute({{p::in::IMAGES, DataRef<Image>(Image{image})}, {p::in::SEARCH_REGIONS, Rectangle::fromEncoded(regions)},
        {p::in::CONTRAST, 30.0}, {p::in::EDGE_WIDTH, std::int64_t{3}},
        {p::in::POLARITY, std::string("any")}, {p::in::MIN_COMPLETENESS, 0.4}});
    require(static_cast<bool>(testview::status(result)), "real alignment execute failed");
    check(result);
    const auto &alignment = std::get<Array>(result.outputs.at(p::out::ALIGNMENTS));
    require(alignment.rank() == 2 && alignment.shape()[1] == Affine2D::CoefficientCount, "one flat alignment per image");
    tool.release();
    std::filesystem::remove(modelFile);
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
