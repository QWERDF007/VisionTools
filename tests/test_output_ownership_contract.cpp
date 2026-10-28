#include "src/preprocess/BrightnessTool.h"
#include "src/preprocess/RotateImageTool.h"

#include "preprocess/TestSupport.h"

#include <opencv2/imgproc.hpp>

#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace visionrt;
using namespace visiontools;
using namespace visiontools::preprocess;

namespace {

namespace rotate_ports = visiontools::preprocess::ports::image_rotate;
namespace bright_ports = visiontools::preprocess::ports::image_brightness;

Values withImages(const cv::Mat &image, std::vector<Owner> keepAlive)
{
    Values values;
    // The value owns a copy so the test can drop it without touching the caller's buffer.
    values.emplace(rotate_ports::in::IMAGES, DataRef<Image>(Image{image.clone()}, std::move(keepAlive)));
    return values;
}

void requireIndependent(const DataRef<Image> &input, const DataRef<Image> &output, const char *context)
{
    const auto &source = input.get();
    const auto &result = output.get();
    REQUIRE(source.size() == result.size(), context);
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        REQUIRE(!result[index].empty(), context);
        // Any shared row pointer would mean the "new" output is a view of the input.
        REQUIRE(result[index].data != source[index].data, context);
        REQUIRE(result[index].ptr(0) != source[index].ptr(0), context);
    }
}

// A tool that allocates new pixels has no reason to keep the upstream read-cache token alive: doing
// so pins decoded images for as long as any downstream result exists. The observable contract is
// "the output survives on its own", so the test drops the input and checks the token, the pointer
// identity and the frame size instead of inspecting an implementation detail.
void requireSelfContained(const char *context, const std::function<ToolResult(const Values &)> &run,
                          const cv::Mat &image)
{
    auto               token    = std::make_shared<int>(7);
    std::weak_ptr<int> observed = token;
    Values             inputs   = withImages(image, {token});
    token.reset();
    REQUIRE(!observed.expired(), "the input owner must stay alive while the input reference exists");

    std::optional<DataRef<Image>> output;
    {
        const auto result = run(inputs);
        REQUIRE(testview::status(result).code == 0, testview::status(result).message);
        const auto found = result.outputs.find(rotate_ports::out::IMAGES);
        REQUIRE(found != result.outputs.end(), context);
        const auto *images = std::get_if<DataRef<Image>>(&found->second);
        REQUIRE(images != nullptr, context);
        output = *images;
        requireIndependent(std::get<DataRef<Image>>(inputs.at(rotate_ports::in::IMAGES)), *images, context);
        REQUIRE(images->get().front().size() == image.size(), context);
    }
    // Drop the only remaining reference to the input, so what survives is the output alone.
    inputs.clear();
    REQUIRE(observed.expired(), std::string("the output retained the input owner although it allocated its own "
                                            "pixels: ")
                                    + context);
    REQUIRE(output.has_value() && !output->get().empty() && !output->get().front().empty(), context);
}

} // namespace

int main()
{
    try
    {
        cv::Mat image(32, 40, CV_8UC1);
        for (int y = 0; y < image.rows; ++y)
            for (int x = 0; x < image.cols; ++x)
                image.at<std::uint8_t>(y, x) = static_cast<std::uint8_t>((x * 7 + y * 3) % 251);

        {
            RotateImageTool tool(ToolServices{});
            const auto      run = [&tool](const Values &inputs)
            {
                Values values = inputs;
                values.emplace(rotate_ports::in::ANGLE_DEG, 90.0);
                values.emplace(rotate_ports::in::CENTER,
                               Array(DataRef<NestedVector<double, 1>>(NestedVector<double, 1>{})));
                values.emplace(rotate_ports::in::CANVAS, std::string("keep"));
                values.emplace(rotate_ports::in::INTERPOLATION, std::string("nearest"));
                values.emplace(rotate_ports::in::BORDER_VALUE,
                               Array(DataRef<NestedVector<double, 1>>(NestedVector<double, 1>{0.0})));
                values.emplace(rotate_ports::in::ALIGNMENT,
                               Array(DataRef<NestedVector<double, 3>>(NestedVector<double, 3>{})));
                return tool.execute(values);
            };
            requireSelfContained("rotate must return pixels it owns", run, image);
        }

        {
            // gain 1 / offset 0: the constant path still clones, so pixel equality is exact and proves
            // the dropped input buffer was not the one being read afterwards.
            const auto run = [](const Values &inputs)
            {
                Values values = inputs;
                values.emplace(bright_ports::in::GAIN, 1.0);
                values.emplace(bright_ports::in::OFFSET, 0.0);
                values.emplace(bright_ports::in::ROIS, DataRef<ROI>(ROI{}));
                values.emplace(bright_ports::in::ALIGNMENT,
                               Array(DataRef<NestedVector<double, 3>>(NestedVector<double, 3>{})));
                values.emplace(bright_ports::in::EMIT_MASK, false);
                return brightnessTool(values, ToolServices{});
            };
            requireSelfContained("brightness must return pixels it owns", run, image);
        }

        std::cout << "[OK] rotate and brightness own their output pixels and release the input owner\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 1;
    }
}
