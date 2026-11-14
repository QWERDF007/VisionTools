// A condition branch hands a non-zero status to the tools it did not select. The framework never
// blocks a node on a business status, so every formal tool decides for itself: it must not run its
// algorithm, must not touch the file system, and must answer with the same status plus Missing
// values for its own outputs.

#include "common/RoiSupport.h"
#include "preprocess/AdaptiveBinaryTool.h"
#include "preprocess/BrightnessTool.h"
#include "preprocess/ReadImageTool.h"
#include "ToolResultAssertions.h"
#include "visionrt/contracts/Tool.h"

#include <opencv2/core.hpp>

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace visionrt;
using namespace visiontools;

namespace read_ports = visiontools::preprocess::ports::image_read;
namespace bright     = visiontools::preprocess::ports::image_brightness;

ToolResult runBrightness(const Values &inputs, const std::vector<PortId> &businessOutputs)
{
    auto result = preprocess::brightnessTool(inputs, ToolServices{});
    assert(testview::businessCount(result) == businessOutputs.size());
    for (const auto id : businessOutputs)
        assert(result.outputs.contains(id));
    return result;
}

template<class Tool>
ToolResult run(const Values &inputs, const std::vector<PortId> &businessOutputs)
{
    Tool                    tool{ToolServices{}};
    const ToolStatus        initialized = tool.initialize({});
    assert(initialized.code == 0);
    auto result = tool.execute(inputs);
    assert(testview::businessCount(result) == businessOutputs.size());
    for (const auto id : businessOutputs)
        assert(result.outputs.contains(id));
    assert(tool.release().code == 0);
    return result;
}

void unselectedBranchDoesNothing()
{
    // A path that cannot be read: if the tool touched the disk it would report a decode failure
    // instead of the upstream status.
    Values readInputs{{read_ports::in::PATH, std::string("does-not-exist.png")},
                      {read_ports::in::RECURSIVE, false},
                      {read_ports::in::TRAVERSAL, std::string("循环")},
                      {read_ports::in::ORDER, std::string("顺序（下一张）")},
                      {read_ports::in::COUNT, std::int64_t{1}},
                      {read_ports::in::RELOAD, false},
                      {kUpstreamStatusPort, std::int64_t{7}}};
    const auto read = run<preprocess::ReadImageTool>(
        readInputs, {read_ports::out::IMAGES, read_ports::out::PATHS, read_ports::out::CURRENT_INDEX,
                     read_ports::out::TOTAL_COUNT, read_ports::out::END_REACHED});
    assert(testview::status(read).code == 7);
    assert(testview::status(read).message.find("未选中") != std::string::npos);
    for (const auto &[id, value] : read.outputs)
        if (!isStatusPort(id))
            assert(isMissing(value));

    cv::Mat image(4, 4, CV_8UC1, cv::Scalar(3));
    Values  brightnessInputs{
         {bright::in::IMAGES, DataRef<Image>(Image{image})},
         {bright::in::GAIN, 2.0},
         {bright::in::OFFSET, 0.0},
         {bright::in::ROIS, DataRef<ROI>(ROI{})},
         {bright::in::ALIGNMENT, Array::fromNested(NestedVector<double, 3>{})},
         {kUpstreamStatusPort, std::int64_t{-3}}};
    const auto brightness = runBrightness(brightnessInputs, {bright::out::IMAGES, bright::out::APPLIED_MASK});
    assert(testview::status(brightness).code == -3);
    assert(isMissing(brightness.outputs.at(bright::out::IMAGES)));

    // `images` is declared optional, so an unselected branch without an image must still be skipped
    // instead of failing on the missing input before the tool's own policy runs.
    auto skipped               = brightnessInputs;
    skipped[kUpstreamStatusPort] = std::int64_t{9};
    skipped[bright::in::IMAGES]  = Missing{};
    const auto branch = runBrightness(skipped, {bright::out::IMAGES, bright::out::APPLIED_MASK});
    assert(testview::status(branch).code == 9);
    assert(testview::status(branch).message.find("未选中") != std::string::npos);
    assert(isMissing(branch.outputs.at(bright::out::IMAGES)));

    // The same tool with a zero status runs normally, so the gate is the status alone.
    brightnessInputs[kUpstreamStatusPort] = std::int64_t{0};
    const auto processed = runBrightness(brightnessInputs, {bright::out::IMAGES, bright::out::APPLIED_MASK});
    assert(testview::status(processed).code == 0);
    const auto *images = std::get_if<DataRef<Image>>(&processed.outputs.at(bright::out::IMAGES));
    assert(images && images->get().size() == 1 && images->get().front().at<unsigned char>(0, 0) == 6);
}

void unselectedBranchStillValidatesItsContract()
{
    // A missing status means "no upstream condition": the tool runs and reports the real problem.
    Values inputs{{bright::in::IMAGES, Missing{}},
                  {bright::in::GAIN, 1.0},
                  {bright::in::OFFSET, 0.0},
                  {bright::in::ROIS, DataRef<ROI>(ROI{})},
                  {bright::in::ALIGNMENT, Array::fromNested(NestedVector<double, 3>{})},
                           {kUpstreamStatusPort, std::int64_t{0}}};
    const auto result = runBrightness(inputs, {bright::out::IMAGES, bright::out::APPLIED_MASK});
    assert(testview::status(result).code != 0);
    assert(testview::status(result).message.find(portLabel(bright::in::IMAGES)) != std::string::npos);
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
