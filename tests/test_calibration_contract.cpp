#include "metrology/CalibrateCameraTool.h"
#include "metrology/RowSpacingTool.h"
#include "common/GeometryTransform.h"
#include "common/PlanarCalibration.h"
#include "ToolResultAssertions.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numbers>
#include <numeric>
#include <stdexcept>

using namespace visionrt;
using namespace visiontools;
namespace p = metrology::ports::calibrate_camera;
namespace row = metrology::ports::measure_row_spacing;

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Files
{
    std::filesystem::path descriptor, model;
    Files()
    {
        const auto stem = "visiontools-calibration-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        descriptor = std::filesystem::temp_directory_path() / (stem + ".cpd");
        model = std::filesystem::temp_directory_path() / (stem + ".bin");
    }
    ~Files()
    {
        std::error_code ignored;
        std::filesystem::remove(descriptor, ignored);
        std::filesystem::remove(model, ignored);
    }
};

cv::Mat syntheticPlate(const std::filesystem::path &descriptor)
{
    cv::Mat image(900, 1000, CV_8UC1, cv::Scalar(0));
    std::ofstream file(descriptor);
    file.precision(17);
    file << "# Plate Description Version 3\nr 9\nc 9\nz 0\np light\nf 1\n4 4\n";
    for (int y = 0; y < 9; ++y)
        for (int x = 0; x < 9; ++x)
        {
            const double physicalX = x - 4 + (y % 2) * 0.5;
            const double physicalY = (y - 4) * std::sqrt(3.0) / 2;
            file << physicalX / 1000 << ' ' << physicalY / 1000 << " 0.00024\n";
            const cv::Point center(cvRound((500 + 50 * physicalX) * 256), cvRound((450 + 50 * physicalY) * 256));
            cv::ellipse(image, center, {12 * 256, 12 * 256}, 0, 0, 360, cv::Scalar(255), cv::FILLED, cv::LINE_AA, 8);
            const int r = y - 4, q = x - 4 + (y % 2 - r) / 2;
            const bool hole = (q == 0 && r == 0) || (q == 0 && r == 1) || (q == -1 && r == 1)
                || (q == -1 && r == 0) || (q == 0 && r == -1) || (q == 1 && r == -1);
            if (hole) cv::ellipse(image, center, {4 * 256, 4 * 256}, 0, 0, 360, cv::Scalar(0), cv::FILLED, cv::LINE_AA, 8);
        }
    return image;
}

cv::Mat loadImage(const std::filesystem::path &path)
{
    std::ifstream stream(path, std::ios::binary);
    std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    auto image = cv::imdecode(bytes, cv::IMREAD_UNCHANGED);
    require(!image.empty(), "calibration test image unavailable");
    return image;
}

#ifdef _WIN32
int wmain(int argc, wchar_t **argv)
#else
int main(int argc, char **argv)
#endif
try
{
    require(argc == 1 || argc == 4 || argc == 5, "usage: test_calibration_contract [plate.png ruler.png plate.cpd [model-output]]");
    Files files;
    const bool real = argc >= 4;
    const auto image = real ? loadImage(std::filesystem::path(argv[1])) : syntheticPlate(files.descriptor);
    const auto descriptor = real ? std::filesystem::path(argv[3]) : files.descriptor;
    TrainingSet set;
    set.images.emplace(1, trainingFrame(1, image, ColorOrder::Automatic));
    set.records.emplace(1, TrainingRecord{1, 1, {}});
    metrology::CalibrateCameraTool tool;
    const auto text = descriptor.generic_u8string();
    const auto trained = tool.train({{p::train::DATASET, DataRef<TrainingSet>(set)},
        {p::train::PLATE_DESCRIPTION, std::string(reinterpret_cast<const char *>(text.data()), text.size())},
        {p::train::MODEL_PATH, std::string{}}, {p::train::MAX_REPROJECTION_ERROR, 0.5}});
    if (!testview::status(trained)) std::cerr << testview::status(trained).message << '\n' << testview::status(trained).detail << '\n';
    require(static_cast<bool>(testview::status(trained)), "plate training failed");
    const auto rms = std::get<double>(trained.outputs.at(p::out::REPROJECTION_ERROR));
    require(rms < 0.5, "reprojection quality threshold");
    const auto base = geometry::calibrationsFromArray(std::get<Array>(trained.outputs.at(p::out::CALIBRATION)), 1).front();
    require(base.imageSize == image.size(), "calibration preserves image geometry");
    require(std::get<CoordinateFrame2D>(trained.outputs.at(p::out::COORDINATE_FRAME)).count(0) == 1, "generic coordinate frame output");
    const auto frame = std::get<CoordinateFrame2D>(trained.outputs.at(p::out::COORDINATE_FRAME)).at(0, 0);
    require(frame.xAxis().x > 0 && frame.yAxis().y < 0, "world positive X points right and positive Y points up");
    require(base.map({500, 400}).y > base.map({500, 500}).y, "pixel positions above the origin have larger world Y");
    if (!real) require(std::abs(cv::norm(base.map({550, 450}) - base.map({500, 450})) - 1) < 0.01, "independent synthetic one millimeter reference");
    const auto &model = trained.artifacts.at(p::MODEL_ARTIFACT).bytes;
    { std::ofstream file(files.model, std::ios::binary); file.write(reinterpret_cast<const char *>(model.data()), model.size()); }
    ToolConfiguration configuration;
    configuration.models.emplace(p::MODEL_ARTIFACT, files.model);
    require(static_cast<bool>(tool.initialize(configuration)), "serialized calibration model loads");

    const auto current = real ? loadImage(std::filesystem::path(argv[2])) : image;
    const cv::Point2d first = real ? cv::Point2d(46.10336134453775, 131.2736660612605) : cv::Point2d(350, 450);
    const cv::Point2d second = real ? cv::Point2d(1201.316384735713, 131.4324900598861) : cv::Point2d(650, 450);
    const double known = real ? 55.0 : 12.0;
    const auto result = tool.execute({{p::in::IMAGES, DataRef<Image>(Image{current})},
        {p::in::REFERENCE_SEGMENTS, Line::fromEncoded(ROI{{makeSegment(first, second)}})},
        {p::in::REFERENCE_LENGTH, known}, {p::in::UNIT, std::string("mm")}});
    require(static_cast<bool>(testview::status(result)), "measurement plane scaling failed");
    const auto &array = std::get<Array>(result.outputs.at(p::out::CALIBRATION));
    const auto calibrated = geometry::calibrationsFromArray(array, 1).front();
    require(std::abs(cv::norm(calibrated.map(second) - calibrated.map(first)) - known) < 1e-8, "current plane uses the known segment length");
    auto baseInputs = Values{{p::in::IMAGES, DataRef<Image>(Image{current})}, {p::in::REFERENCE_SEGMENTS, Line{}},
        {p::in::REFERENCE_LENGTH, known}, {p::in::UNIT, std::string("mm")}};
    const auto unchanged = tool.execute(baseInputs);
    require(static_cast<bool>(testview::status(unchanged)), "empty reference keeps the trained plane");
    const auto original = geometry::calibrationsFromArray(std::get<Array>(unchanged.outputs.at(p::out::CALIBRATION)), 1).front();
    require(cv::norm(original.map(second) - base.map(second)) < 1e-10, "without a reference there is no plane scale adjustment");
    if (real)
    {
        metrology::RowSpacingTool ruler;
        const double width = 1255.622284175208, height = 49.924994627343;
        const auto measurement = ruler.execute({{row::in::IMAGES, DataRef<Image>(Image{current})},
            {row::in::ROIS, Rectangle::fromEncoded(ROI{{makeRectangle(643.7009629045889-width/2, 79.0053147532451-height/2,
                width, height, 0.000990063*180/std::numbers::pi)}})},
            {row::in::ALIGNMENT, geometry::alignmentArray(geometry::Alignment::identity())}, {row::in::CALIBRATION, array},
            {row::in::SIGMA, 1.0}, {row::in::MIN_GRADIENT, 8.0}, {row::in::OBJECT_POLARITY, std::string("dark")},
            {row::in::WIDTH_MIN, 1.0}, {row::in::WIDTH_MAX, 1000.0}, {row::in::UNITS_PER_PIXEL, 1.0}, {row::in::UNIT, std::string("mm")}});
        if (!testview::status(measurement)) std::cerr << testview::status(measurement).message << '\n';
        require(static_cast<bool>(testview::status(measurement)), "calibrated ruler measurement failed");
        const auto values = nestedOf<NestedVector<double, 3>>(std::get<Array>(measurement.outputs.at(row::out::PITCH))).front().front();
        require(values.size() > 1, "ruler provides multiple independent pitches");
        const auto mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double squared = 0;
        for (const auto value : values) squared += (value - mean) * (value - mean);
        const auto deviation = std::sqrt(squared / (values.size() - 1)) * 1000;
        std::cout << "RMS " << rms << " px; pitches " << values.size() << "; mean " << mean << " mm; sample deviation " << deviation << " um\n";
        require(mean >= 0.99 && mean <= 1.01, "ruler mean must stay within one percent of one millimeter");
        require(deviation < 20, "ruler sample deviation must stay below twenty micrometers");
    }
    if (argc == 5)
    {
        const std::filesystem::path destination(argv[4]);
        std::filesystem::create_directories(destination.parent_path());
        std::filesystem::copy_file(files.model, destination, std::filesystem::copy_options::overwrite_existing);
    }
    tool.release();
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
