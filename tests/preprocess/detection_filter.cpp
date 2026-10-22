#include "src/preprocess/DetectionFilterTool.h"
#include "TestSupport.h"

using namespace visionrt;
using namespace visiontools::preprocess;
using namespace visiontools::tests;

namespace filter_ports = visiontools::preprocess::ports::detection_filter;

namespace {

Values detectionInputs()
{
    const auto                roi     = std::vector<double>{2, 10.0, 10.0, 20.0, 20.0, 0.0};
    std::vector<float>        scores  = {0.85F, 0.35F};
    std::vector<std::int64_t> classes = {1, 2};
    std::vector<std::string>  names   = {"cat", "dog"};
    Values                    inputs;
    inputs.emplace(filter_ports::in::IMAGES, DataRef<Image>(Image{(gradient(100, 100)).clone()}));
    inputs.emplace(filter_ports::in::ROIS, DataRef<ROI>(ROI{
                                               {roi, roi}
    }));
    inputs.emplace(filter_ports::in::SCORES, Array(std::vector<std::vector<float>>{scores}));
    inputs.emplace(filter_ports::in::CLASSES, Array(std::vector<std::vector<std::int64_t>>{classes}));
    inputs.emplace(filter_ports::in::CLASS_NAMES, Array(std::vector<std::vector<std::string>>{names}));
    inputs.emplace(filter_ports::in::ALLOWED_CLASSES, Array(std::vector<std::int64_t>{1, 2}));
    inputs.emplace(filter_ports::in::MIN_SCORE, 0.5);
    inputs.emplace(filter_ports::in::MIN_AREA, 0.0);
    return inputs;
}

} // namespace

void testDetectionFilterTool()
{
    std::cout << "[TEST] Running testDetectionFilterTool..." << std::endl;
    ToolServices        services;
    DetectionFilterTool tool(services);

    static_assert(std::string_view(filter_ports::in::MIN_SCORE) == "min_score");
    static_assert(std::string_view(filter_ports::in::ALLOWED_CLASSES) == "allowed_classes");
    static_assert(std::string_view(filter_ports::out::COUNTS) == "counts");

    {
        auto inputs = detectionInputs();
        inputs.insert_or_assign(filter_ports::in::MIN_SCORE, std::string("not_a_number"));
        auto res = tool.execute(inputs);
        requireFailure(res, "浮点", "String min_score must be rejected");
        REQUIRE(res.status.message.find("min_score") != std::string::npos, "Error must mention min_score");
    }
    {
        auto inputs = detectionInputs();
        inputs.insert_or_assign(filter_ports::in::MIN_SCORE, std::int64_t(0));
        auto res = tool.execute(inputs);
        requireFailure(res, "浮点", "Int min_score must be rejected");
    }
    {
        auto inputs = detectionInputs();
        inputs.erase(filter_ports::in::MIN_SCORE);
        auto res = tool.execute(inputs);
        requireFailure(res, "min_score", "Missing min_score must be rejected");
        REQUIRE(res.status.message.find("缺失") != std::string::npos, "Missing value must be Chinese");
    }
    {
        auto inputs = detectionInputs();
        inputs.insert_or_assign(filter_ports::in::MIN_SCORE, 1.5);
        auto res = tool.execute(inputs);
        requireFailure(res, "min_score", "Out-of-range min_score must be rejected");
        REQUIRE(res.status.message.find("[0") != std::string::npos, "Range must be stated");
    }
    {
        auto inputs = detectionInputs();
        inputs.insert_or_assign(filter_ports::in::MIN_AREA, -1.0);
        auto res = tool.execute(inputs);
        requireFailure(res, "min_area", "Negative min_area must be rejected");
        REQUIRE(res.status.message.find("非负") != std::string::npos, "Error must state non-negative");
    }
    {
        auto inputs = detectionInputs();
        auto res    = tool.execute(inputs);
        requireSuccess(res, "Filter must succeed");
        const auto *rois = std::get_if<DataRef<ROI>>(&res.outputs[filter_ports::out::ROIS]);
        REQUIRE(rois != nullptr && rois->get()[0].size() == 1, "Only score >= 0.5 passes");
        const auto counts = std::get<Array>(res.outputs[filter_ports::out::COUNTS]).get<std::vector<std::int64_t>>();
        REQUIRE(counts[0] == 1, "Counts must report the filtered count");
    }
    {
        auto inputs = detectionInputs();
        inputs.insert_or_assign(filter_ports::in::ALLOWED_CLASSES, Array(std::vector<std::int64_t>{1}));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Allowed-class filter must succeed");
        const auto *rois = std::get_if<DataRef<ROI>>(&res.outputs[filter_ports::out::ROIS]);
        REQUIRE(rois->get()[0].size() == 1, "Only the allowed class passes");
        const auto names = std::get<Array>(res.outputs[filter_ports::out::CLASS_NAMES])
                               .get<std::vector<std::vector<std::string>>>()[0];
        REQUIRE(names[0] == "cat", "Filtered class name must be retained");
    }
    {
        auto                      inputs   = detectionInputs();
        std::vector<float>        one      = {0.5F};
        std::vector<std::int64_t> oneClass = {1};
        std::vector<std::string>  oneName  = {"cat"};
        inputs.insert_or_assign(filter_ports::in::SCORES, Array(std::vector<std::vector<float>>{one}));
        inputs.insert_or_assign(filter_ports::in::CLASSES, Array(std::vector<std::vector<std::int64_t>>{oneClass}));
        inputs.insert_or_assign(filter_ports::in::CLASS_NAMES, Array(std::vector<std::vector<std::string>>{oneName}));
        auto res = tool.execute(inputs);
        requireFailure(res, "长度不一致", "Field length mismatch must fail");
        REQUIRE(res.status.message.find("批次索引 0") != std::string::npos, "Error must name the batch index");
    }
    {
        auto       inputs = detectionInputs();
        const auto circle = std::vector<double>{3, 5, 5, 2};
        inputs.insert_or_assign(filter_ports::in::ROIS, DataRef<ROI>(ROI{
                                                            {circle, circle}
        }));
        auto res = tool.execute(inputs);
        requireFailure(res, "矩形", "Non-rectangle ROI must be rejected");
    }
    {
        auto                      inputs   = detectionInputs();
        std::vector<std::int64_t> negative = {-1, 2};
        inputs.insert_or_assign(filter_ports::in::CLASSES, Array(std::vector<std::vector<std::int64_t>>{negative}));
        auto res = tool.execute(inputs);
        requireFailure(res, "得分或类别", "Negative class must be rejected");
    }
    {
        Values inputs;
        inputs.emplace(filter_ports::in::IMAGES, DataRef<Image>(Image{}));
        inputs.emplace(filter_ports::in::ROIS, DataRef<ROI>(ROI{}));
        inputs.emplace(filter_ports::in::SCORES, Array(std::vector<std::vector<float>>{}));
        inputs.emplace(filter_ports::in::CLASSES, Array(std::vector<std::vector<std::int64_t>>{}));
        inputs.emplace(filter_ports::in::CLASS_NAMES, Array(std::vector<std::vector<std::string>>{}));
        inputs.emplace(filter_ports::in::ALLOWED_CLASSES, Array(std::vector<std::int64_t>{}));
        inputs.emplace(filter_ports::in::MIN_SCORE, 0.5);
        inputs.emplace(filter_ports::in::MIN_AREA, 0.0);
        auto res = tool.execute(inputs);
        requireSuccess(res, "Empty detection batch must succeed");
        REQUIRE(std::get<DataRef<ROI>>(res.outputs[filter_ports::out::ROIS]).get().empty(),
                "Empty batch must produce empty rois");
    }

    std::cout << "[PASS] testDetectionFilterTool passed." << std::endl;
}
