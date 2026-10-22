#include "src/preprocess/RoiProjectionTool.h"
#include "TestSupport.h"

using namespace visionrt;
using namespace visiontools::preprocess;
using namespace visiontools::tests;

namespace projection_ports = visiontools::preprocess::ports::roi_projection;

static Values projectionInputs()
{
    Values inputs;
    inputs.emplace(projection_ports::in::AXIS, std::string("x"));
    inputs.emplace(projection_ports::in::POLARITY, std::string("bright"));
    inputs.emplace(projection_ports::in::THRESHOLD, std::int64_t(128));
    inputs.emplace(projection_ports::in::MIN_PIXELS, std::int64_t(1));
    inputs.emplace(projection_ports::in::MIN_EXTENT, std::int64_t(1));
    inputs.emplace(projection_ports::in::MERGE_GAP, std::int64_t(0));
    inputs.emplace(projection_ports::in::PADDING, std::int64_t(0));
    return inputs;
}

void testRoiProjectionTool()
{
    std::cout << "[TEST] Running testRoiProjectionTool..." << std::endl;
    ToolServices      services;
    RoiProjectionTool tool(services);

    static_assert(std::string_view(projection_ports::in::AXIS) == "axis");
    static_assert(std::string_view(projection_ports::in::MIN_PIXELS) == "min_pixels");
    static_assert(std::string_view(projection_ports::in::MERGE_GAP) == "merge_gap");
    static_assert(std::string_view(projection_ports::out::ROIS) == "rois");

    cv::Mat img(100, 100, CV_8UC1, cv::Scalar(0));
    img(cv::Rect(30, 30, 40, 40)).setTo(cv::Scalar(255));

    // Missing / invalid values
    {
        Values inputs = projectionInputs();
        inputs.emplace(projection_ports::in::IMAGES, DataRef<Image>(Image{(img).clone()}));
        inputs.erase(projection_ports::in::AXIS);
        auto res = tool.execute(inputs);
        requireFailure(res, "axis", "Missing axis must fail");
        REQUIRE(res.status.message.find("缺失") != std::string::npos, "Missing value must be Chinese");
    }
    {
        Values inputs = projectionInputs();
        inputs.emplace(projection_ports::in::IMAGES, DataRef<Image>(Image{(img).clone()}));
        inputs.insert_or_assign(projection_ports::in::AXIS, std::string("z"));
        auto res = tool.execute(inputs);
        requireFailure(res, "axis", "Invalid axis must fail");
        REQUIRE(res.status.message.find("无效") != std::string::npos, "Invalid option must be Chinese");
    }
    {
        Values inputs = projectionInputs();
        inputs.emplace(projection_ports::in::IMAGES, DataRef<Image>(Image{(img).clone()}));
        inputs.insert_or_assign(projection_ports::in::POLARITY, std::string("mixed"));
        auto res = tool.execute(inputs);
        requireFailure(res, "polarity", "Invalid polarity must fail");
    }
    {
        Values inputs = projectionInputs();
        inputs.emplace(projection_ports::in::IMAGES, DataRef<Image>(Image{(img).clone()}));
        inputs.insert_or_assign(projection_ports::in::THRESHOLD, std::int64_t(300));
        auto res = tool.execute(inputs);
        requireFailure(res, "threshold", "Out-of-range threshold must fail");
        REQUIRE(res.status.message.find("[0, 255]") != std::string::npos, "Range must be stated");
    }
    for (const auto &[key, value] : std::vector<std::pair<const char *, std::int64_t>>{
             {projection_ports::in::MIN_PIXELS,  0},
             {projection_ports::in::MIN_EXTENT,  0},
             { projection_ports::in::MERGE_GAP, -1},
             {   projection_ports::in::PADDING, -1}
    })
    {
        Values inputs = projectionInputs();
        inputs.emplace(projection_ports::in::IMAGES, DataRef<Image>(Image{(img).clone()}));
        inputs.insert_or_assign(key, value);
        auto res = tool.execute(inputs);
        requireFailure(res, key, "Out-of-range projection parameter must fail");
    }

    // Valid projection
    {
        Values inputs = projectionInputs();
        inputs.emplace(projection_ports::in::IMAGES, DataRef<Image>(Image{(img).clone()}));
        inputs.insert_or_assign(projection_ports::in::MIN_PIXELS, std::int64_t(10));
        inputs.insert_or_assign(projection_ports::in::MIN_EXTENT, std::int64_t(10));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Projection must succeed");
        const auto *rois = std::get_if<DataRef<ROI>>(&res.outputs["rois"]);
        REQUIRE(rois != nullptr && rois->get().size() == 1 && rois->get()[0].size() == 1, "Expected 1 projected ROI");
        const auto encoded = rois->get()[0][0];
        REQUIRE(encoded[0] == 2.0F, "ROI must be rectangle");
        REQUIRE(std::abs(encoded[1] - 30.0F) < 1e-3, "ROI left must be 30");
        REQUIRE(std::abs(encoded[3] - 40.0F) < 1e-3, "ROI width must be 40");
    }

    // Empty batch
    {
        Values inputs = projectionInputs();
        inputs.emplace(projection_ports::in::IMAGES, DataRef<Image>(Image{}));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Empty batch must succeed");
        REQUIRE(std::get<DataRef<Image>>(res.outputs["images"]).get().empty(), "Empty images must stay empty");
        REQUIRE(std::get<DataRef<ROI>>(res.outputs["rois"]).get().empty(), "Empty rois must stay empty");
    }

    std::cout << "[PASS] testRoiProjectionTool passed." << std::endl;
}
