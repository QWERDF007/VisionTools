#pragma once

#include <visionrt/contracts/Tool.h>

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace visiontools::preprocess {

namespace ports::image_read {
inline constexpr char TOOL_ID[] = "visionrt.image.read";

namespace in {
inline constexpr char PATH[]      = "path";
inline constexpr char RECURSIVE[] = "recursive";
inline constexpr char TRAVERSAL[] = "traversal";
inline constexpr char ORDER[]     = "order";
inline constexpr char COUNT[]     = "count";
inline constexpr char RELOAD[]    = "reload";
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
        std::string                        rawPath;
        bool                               isDirectory = false;
        bool                               recursive   = false;
        std::vector<std::filesystem::path> files;
    };

    visionrt::ToolServices services_;
    std::optional<Catalog> catalog_;
    std::size_t            currentIndex_ = 0;
    bool                   lastReverse_  = false;

    // Two-tier cache:
    // 1) Decoded cv::Mat per normalized utf-8 file path
    std::unordered_map<std::string, cv::Mat>           fileCache_;
    // 2) Identical consecutive batch reuse
    std::vector<std::string>                           lastPlannedPaths_;
    std::optional<visionrt::DataRef<visionrt::Image>> lastBatchRef_;
};

#ifdef VISIONTOOLS_TESTING
namespace testing {
extern std::function<void(const std::filesystem::path &)> beforeRead;
extern std::function<void(const std::filesystem::path &)> afterDecode;
extern std::atomic<std::size_t>                           fileReadCount;
extern std::atomic<std::size_t>                           decodeCount;
} // namespace testing
#endif

} // namespace visiontools::preprocess
