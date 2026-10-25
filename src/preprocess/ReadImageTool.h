#pragma once

#include <visionrt/contracts/Tool.h>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace visiontools::preprocess {

namespace ports::image_read {
inline constexpr char TOOL_ID[] = "visionrt.image.read";

namespace in {
inline constexpr char COLOR[]           = "color";
inline constexpr char LIMIT[]           = "limit";
inline constexpr char MAX_BATCH_BYTES[] = "max_batch_bytes";
inline constexpr char MODE[]            = "mode";
inline constexpr char PATH[]            = "path";
inline constexpr char PATHS_IN[]        = "paths_in";
inline constexpr char READ_MODE[]       = "read_mode";
inline constexpr char RECURSIVE[]       = "recursive";
inline constexpr char EXTENSIONS[]      = "extensions";
inline constexpr char END_POLICY[]      = "end_policy";
}

namespace out {
inline constexpr char IMAGES[]        = "images";
inline constexpr char PATHS[]         = "paths";
inline constexpr char CURRENT_INDEX[] = "current_index";
inline constexpr char TOTAL_COUNT[]   = "total_count";
inline constexpr char END_REACHED[]   = "end_reached";
}
}

class ReadImageTool final : public visionrt::IToolInstance
{
public:
    explicit ReadImageTool(const visionrt::ToolServices &services);
    ~ReadImageTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;

private:
    struct Catalog
    {
        std::string                        key;
        std::vector<std::filesystem::path> files;
    };

    visionrt::ToolServices services_;
    std::optional<Catalog> catalog_;
    std::size_t            nextIndex_ = 0;
    std::string            readMode_;
};

#ifdef VISIONTOOLS_TESTING
namespace testing {
// Test-only read boundary: runs after the size checks and before stream.read.
extern std::function<void(const std::filesystem::path &)> beforeRead;
} // namespace testing
#endif

} // namespace visiontools::preprocess
