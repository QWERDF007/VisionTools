// A condition branch hands a non-zero status to the tools it did not select. The framework never
// blocks a node on a business status, so every formal tool decides for itself: it must not run its
// algorithm, must not touch the file system, and must answer with the same status plus Missing
// values for its own outputs.

#include "common/RoiSupport.h"
#include "preprocess/AdaptiveBinaryTool.h"
#include "preprocess/BrightnessTool.h"
#include "preprocess/ReadImageTool.h"
#include "result_view.h"
#include "visionrt/contracts/Tool.h"

#include <opencv2/core.hpp>

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace visionrt;
using namespace visiontools;

template<class Tool>
ToolResult run(const Values &inputs, const std::vector<std::string> &businessOutputs)
{
    Tool                    tool{ToolServices{}};
    const ToolStatus        initialized = tool.initialize({});
    assert(initialized.code == 0);
    auto result = tool.execute(inputs);
    assert(testview::businessCount(result) == businessOutputs.size());
    for (const auto &id : businessOutputs)
        assert(result.outputs.contains(id));
    assert(tool.release().code == 0);
    return result;
}

void unselectedBranchDoesNothing()
{
    // A path that cannot be read: if the tool touched the disk it would report a decode failure
    // instead of the upstream status.
    Values readInputs{{"path", std::string("does-not-exist.png")},
                      {"recursive", false},
                      {"traversal", std::string("循环")},
                      {"order", std::string("顺序（下一张）")},
                      {"count", std::int64_t{1}},
                      {"reload", false},
                      {"upstream_status", std::int64_t{7}}};
    const auto read = run<preprocess::ReadImageTool>(readInputs,
                                        {"images", "paths", "current_index", "total_count", "end_reached"});
    assert(testview::status(read).code == 7);
    assert(testview::status(read).message.find("未选中") != std::string::npos);
    for (const auto &[id, value] : read.outputs)
        if (id != "__status_code" && id != "__status_message")
            assert(isMissing(value));

    cv::Mat image(4, 4, CV_8UC1, cv::Scalar(3));
    Values  brightnessInputs{
         {"images", DataRef<Image>(Image{image})},
         {"gain", 2.0},
         {"offset", 0.0},
         {"rois", DataRef<ROI>(ROI{})},
         {"alignment", Array(DataRef<NestedVector<double, 3>>(NestedVector<double, 3>{}))},
         {"emit_mask", false},
         {"upstream_status", std::int64_t{-3}}};
    const auto brightness
        = run<preprocess::BrightnessTool>(brightnessInputs, {"images", "applied_mask"});
    assert(testview::status(brightness).code == -3);
    assert(isMissing(brightness.outputs.at("images")));

    // The same tool with a zero status runs normally, so the gate is the status alone.
    brightnessInputs["upstream_status"] = std::int64_t{0};
    const auto processed = run<preprocess::BrightnessTool>(brightnessInputs, {"images", "applied_mask"});
    assert(testview::status(processed).code == 0);
    const auto *images = std::get_if<DataRef<Image>>(&processed.outputs.at("images"));
    assert(images && images->get().size() == 1 && images->get().front().at<unsigned char>(0, 0) == 6);
}

void unselectedBranchStillValidatesItsContract()
{
    // A missing status means "no upstream condition": the tool runs and reports the real problem.
    Values inputs{{"images", Missing{}},
                  {"gain", 1.0},
                  {"offset", 0.0},
                  {"rois", DataRef<ROI>(ROI{})},
                  {"alignment", Array(DataRef<NestedVector<double, 3>>(NestedVector<double, 3>{}))},
                  {"emit_mask", false},
                  {"upstream_status", std::int64_t{0}}};
    const auto result = run<preprocess::BrightnessTool>(inputs, {"images", "applied_mask"});
    assert(testview::status(result).code != 0);
    assert(testview::status(result).message.find("images") != std::string::npos);
}

} // namespace

int main()
try
{
    unselectedBranchDoesNothing();
    unselectedBranchStillValidatesItsContract();
    std::cout << "Unselected upstream branches are skipped by the tool itself, without side effects\n";
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << "upstream status contract failed: " << error.what() << '\n';
    return 1;
}
