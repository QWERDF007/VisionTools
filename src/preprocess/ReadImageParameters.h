#pragma once

#include "ReadImageTool.h"

#include <visionrt/contracts/Parameters.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace visiontools::preprocess {

// Typed view of the read tool's inputs. The tool keeps every business decision (sequence policy,
// cache behaviour, failure handling); this struct only centralises the port reads and the injected
// upstream status.
struct ReadImageInputs final : visionrt::InputParametersBase
{
    std::string  path;
    bool         recursive = false;
    std::string  traversal;
    std::string  order;
    std::int64_t count = 1;
    bool         reload = false;

protected:
    visionrt::ToolStatus readFields(visionrt::InputReader &reader) override;
};

// Typed view of the read tool's outputs. All five declared ports are written explicitly: a failed
// read publishes Missing for every port, a successful one publishes the batch and its sequence
// position. The tool assembles the fields, the base class owns the status keys.
struct ReadImageOutputs final : visionrt::OutputParametersBase
{
    ReadImageOutputs()
    {
        status = visionrt::executionSuccess();
    }

    // One spelling of "this read failed": every declared port carries Missing.
    static ReadImageOutputs missing()
    {
        ReadImageOutputs outputs;
        outputs.present = false;
        outputs.status  = visionrt::executionSuccess();
        return outputs;
    }

    std::optional<visionrt::DataRef<visionrt::Image>>             images;
    std::optional<visionrt::DataRef<std::vector<std::string>>>    paths;
    std::int64_t currentIndex = 0;
    std::int64_t totalCount   = 0;
    bool         endReached   = false;
    bool         present      = false;

    void markFailed(std::int64_t code, std::string message, std::string detail = {})
    {
        present            = false;
        status.code        = code;
        status.message     = std::move(message);
        status.detail      = std::move(detail);
    }

protected:
    void writeFields(visionrt::OutputWriter &writer) override;
};

} // namespace visiontools::preprocess
