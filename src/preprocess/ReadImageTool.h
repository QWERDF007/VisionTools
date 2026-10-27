#pragma once

#include "ImageSequence.h"

#include <visionrt/contracts/Tool.h>

#include <atomic>
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

    // Cache writes planned by a decode. They are applied only when the whole batch succeeded, so a
    // failed attempt can never mix its pixels into the retained history.
    using PendingInserts = std::vector<std::pair<std::string, cv::Mat>>;

    // Local decode result of one plan; committed only after the whole batch decoded.
    struct DecodedBatch
    {
        visionrt::Image          images;
        std::vector<std::string> keys;
        std::vector<std::string> paths;
        PendingInserts           pending;
        std::size_t              decoded = 0;
    };

    // Reads and decodes the plan. Throws without touching any committed state.
    DecodedBatch decode(const Catalog &catalog, const SequencePlan &plan, bool reload, PendingInserts &pending);
    // Swaps a fully successful batch into the tool. Nothing here can fail, which is what makes a
    // failed read leave the previous catalog, cursor and last batch untouched.
    visionrt::ToolResult commit(DecodedBatch batch, const SequencePlan &plan, SequenceCursor cursor,
                                const Catalog *scannedCatalog, bool sourceChanged, bool reverse,
                                std::size_t totalFiles);
    void                 resetSequence();

    visionrt::ToolServices services_;
    std::optional<Catalog>  catalog_;
    SequenceCursor          cursor_;
    bool                    reverse_ = false;
    DecodedImageCache       cache_{std::size_t{512} * 1024 * 1024};
    std::vector<std::string>                          lastPlannedPaths_;
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
