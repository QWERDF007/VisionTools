// VT-02: what the decoded-image cache actually costs. The ticket asks for measurements instead of a
// guess about "512 MiB", so this test reports the budget, the retained bytes, the hit rate, the
// latency of a cold and a warm pass, the process working set and what a second reader instance adds.

#include "src/preprocess/ReadImageTool.h"

#include "preprocess/TestSupport.h"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#endif

using namespace visionrt;
using namespace visiontools::preprocess;

namespace read_ports = visiontools::preprocess::ports::image_read;

namespace {

std::string utf8(const std::filesystem::path &path)
{
    const auto value = path.generic_u8string();
    return {reinterpret_cast<const char *>(value.data()), value.size()};
}

Values readInputs(const std::filesystem::path &path)
{
    Values inputs;
    inputs.emplace(read_ports::in::PATH, utf8(path));
    inputs.emplace(read_ports::in::RECURSIVE, false);
    inputs.emplace(read_ports::in::TRAVERSAL, std::string("循环"));
    inputs.emplace(read_ports::in::ORDER, std::string("顺序（下一张）"));
    inputs.emplace(read_ports::in::COUNT, std::int64_t(1));
    inputs.emplace(read_ports::in::RELOAD, false);
    return inputs;
}

void writePng(const std::filesystem::path &path, const cv::Mat &image)
{
    std::vector<unsigned char> encoded;
    REQUIRE(cv::imencode(".png", image, encoded), "image fixture must encode");
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
    REQUIRE(stream.good(), "image fixture must be written");
}

// Working set in MiB; 0 when the platform cannot report it, in which case the caller only skips the
// memory assertions instead of silently passing them.
double workingSetMiB()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
        return 0.0;
    return static_cast<double>(counters.WorkingSetSize) / (1024.0 * 1024.0);
#else
    return 0.0;
#endif
}

double runPass(ReadImageTool &tool, const std::filesystem::path &path, int count)
{
    const auto started = std::chrono::steady_clock::now();
    for (int index = 0; index < count; ++index)
    {
        const auto result = tool.execute(readInputs(path));
        REQUIRE(testview::status(result).code == 0, testview::status(result).message);
        const auto found = result.outputs.find(read_ports::out::IMAGES);
        REQUIRE(found != result.outputs.end(), "read must publish images");
        REQUIRE(std::get<DataRef<Image>>(found->second).get().size() == 1, "read must publish one image per step");
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const std::filesystem::path evidenceDir = argc > 1 ? argv[1] : ".";
        std::filesystem::create_directories(evidenceDir);

        // 1. The cache class enforces its own budget, independently of the tool.
        {
            DecodedImageCache cache(std::size_t{8} * 1024 * 1024);
            const cv::Mat     image(1024, 1024, CV_8UC3, cv::Scalar(7)); // 3 MiB decoded
            REQUIRE(cache.budget() == std::size_t{8} * 1024 * 1024, "budget must be reported verbatim");
            REQUIRE(cache.insert("a", image) && cache.insert("b", image), "two 3 MiB images must fit 8 MiB");
            REQUIRE(cache.bytes() == std::size_t{6} * 1024 * 1024, "retained bytes must count decoded pixels");
            REQUIRE(cache.insert("c", image), "the third image must evict instead of being refused");
            REQUIRE(cache.bytes() <= cache.budget(), "the cache must never exceed its budget");
            REQUIRE(cache.find("a") == nullptr, "the least recently used entry must be evicted first");
            REQUIRE(cache.find("b") != nullptr && cache.find("c") != nullptr, "recent entries must survive");

            cache.protect({"b"});
            REQUIRE(cache.insert("d", image), "insert must still succeed while one entry is protected");
            REQUIRE(cache.find("b") != nullptr, "a protected batch must survive eviction");
            REQUIRE(cache.find("c") == nullptr, "only unprotected entries may be evicted");

            const cv::Mat oversized(2048, 2048, CV_8UC3); // 12 MiB > 8 MiB budget
            const auto    before = cache.bytes();
            REQUIRE(!cache.insert("huge", oversized), "an image larger than the budget must not be retained");
            REQUIRE(cache.bytes() == before, "a refused image must not change the retained bytes");
        }

        // 2. Reader residency: a cold pass decodes, a warm pass hits.
        const auto directory = std::filesystem::current_path() / "test_read_cache_suite";
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        constexpr int     kImages      = 12;
        constexpr int     kWidth       = 1600;
        constexpr int     kHeight      = 900;
        const std::size_t bytesPerImage = static_cast<std::size_t>(kWidth) * kHeight * 3;
        for (int index = 0; index < kImages; ++index)
        {
            cv::Mat image(kHeight, kWidth, CV_8UC3);
            for (int y = 0; y < kHeight; ++y)
                for (int x = 0; x < kWidth; ++x)
                    image.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<unsigned char>((x + index) % 256),
                                                         static_cast<unsigned char>(y % 256),
                                                         static_cast<unsigned char>((index * 13) % 256));
            writePng(directory / ("frame_" + std::to_string(index) + ".png"), image);
        }

        ReadImageTool tool(ToolServices{PathResolver(std::filesystem::current_path())});
        const double  baselineMiB = workingSetMiB();
        const auto    decodesAtStart = testing::decodeCount.load();

        runPass(tool, directory, 2); // warm-up: catalog scan and the first decoded images
        const auto   coldDecodes = testing::decodeCount.load() - decodesAtStart;
        const double coldMs      = runPass(tool, directory, kImages - 2);
        const double afterColdMiB = workingSetMiB();
        const auto   decodesAfterCold = testing::decodeCount.load();

        const double warmMs = runPass(tool, directory, kImages);
        const auto   decodesAfterWarm = testing::decodeCount.load();
        const double afterWarmMiB = workingSetMiB();

        REQUIRE(coldDecodes == 2, "the warm-up must decode exactly the images it was asked for");
        REQUIRE(decodesAfterCold - decodesAtStart == static_cast<std::size_t>(kImages),
                "the cold pass must decode every image exactly once");
        REQUIRE(decodesAfterWarm == decodesAfterCold, "a repeated sequence must be served from the cache");

        // 3. A long loop over the same sequence must not keep growing the working set.
        const double loopMs     = runPass(tool, directory, 120);
        const auto   loopDecodes = testing::decodeCount.load();
        const double afterLoopMiB = workingSetMiB();
        REQUIRE(loopDecodes == decodesAfterWarm, "a long loop must stay on cached frames");

        const double retainedMiB = afterColdMiB - baselineMiB;
        const double loopGrowthMiB = afterLoopMiB - afterWarmMiB;
        if (afterColdMiB > 0.0)
        {
            REQUIRE(retainedMiB >= 20.0, "twelve decoded frames must show up in the working set");
            REQUIRE(retainedMiB <= 256.0, "retention must stay far below the cache budget for this batch");
            REQUIRE(loopGrowthMiB <= 16.0, "a long loop must not grow the working set");
        }

        // 4. Reader instances do not share a cache: the second instance re-decodes, then hits.
        const auto decodesBeforeSecond = testing::decodeCount.load();
        ReadImageTool second(ToolServices{PathResolver(std::filesystem::current_path())});
        runPass(second, directory, kImages);
        const auto secondCold = testing::decodeCount.load() - decodesBeforeSecond;
        runPass(second, directory, kImages);
        const auto secondWarm = testing::decodeCount.load() - decodesBeforeSecond;
        REQUIRE(secondCold == static_cast<std::size_t>(kImages),
                "a second instance must not reuse the first instance's cache");
        REQUIRE(secondWarm == secondCold, "the second instance must cache its own frames");

        // 5. Evidence. Numbers are per-run and machine-specific; the JSON keeps them next to the test.
        std::ofstream evidence(evidenceDir / "read_cache_evidence.json");
        REQUIRE(evidence.is_open(), "cannot open read_cache_evidence.json");
        evidence.precision(6);
        evidence << std::fixed;
        evidence << "{\n"
                 << "  \"cache_budget_bytes\": " << ReadImageTool::kCacheBudgetBytes << ",\n"
                 << "  \"images\": " << kImages << ",\n"
                 << "  \"decoded_bytes_per_image\": " << bytesPerImage << ",\n"
                 << "  \"cold_pass_ms\": " << coldMs << ",\n"
                 << "  \"warm_pass_ms\": " << warmMs << ",\n"
                 << "  \"loop_120_ms\": " << loopMs << ",\n"
                 << "  \"decodes_cold\": " << (decodesAfterCold - decodesAtStart) << ",\n"
                 << "  \"decodes_warm\": " << (decodesAfterWarm - decodesAfterCold) << ",\n"
                 << "  \"working_set_baseline_mib\": " << baselineMiB << ",\n"
                 << "  \"working_set_after_cold_mib\": " << afterColdMiB << ",\n"
                 << "  \"working_set_after_loop_mib\": " << afterLoopMiB << ",\n"
                 << "  \"second_instance_cold_decodes\": " << secondCold << ",\n"
                 << "  \"second_instance_warm_decodes\": " << (secondWarm - secondCold) << "\n"
                 << "}\n";
        REQUIRE(!evidence.bad(), "error writing read_cache_evidence.json");

        std::cout << "[OK] read cache: budget=" << ReadImageTool::kCacheBudgetBytes / (1024 * 1024) << " MiB"
                  << ", cold=" << coldMs << " ms for " << (kImages - 2) << " frames, warm=" << warmMs
                  << " ms for " << kImages << " frames, loop=" << loopMs << " ms for 120 frames\n";
        std::cout << "[OK] working set: baseline=" << baselineMiB << " MiB, after cold=" << afterColdMiB
                  << " MiB (+" << retainedMiB << "), after 120-frame loop=" << afterLoopMiB << " MiB\n";
        std::cout << "[OK] second reader instance decoded " << secondCold
                  << " frames on its own cold pass and " << (secondWarm - secondCold) << " on the warm one\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 1;
    }
}
