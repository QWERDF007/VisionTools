// The decoded-image cache has two different jobs in this suite: proving the cache is correct, and
// measuring what a large batch costs. Correctness runs on a few small frames; the residency numbers
// need 1600x900 frames and a long loop, so they live behind --perf and are registered separately.

#include "src/preprocess/ReadImageTool.h"

#include "preprocess/TestSupport.h"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <process.h>
#    include <windows.h>
#    include <psapi.h>
#else
#    include <unistd.h>
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

Values readInputs(const std::filesystem::path &path, bool reload = false)
{
    Values inputs;
    inputs.emplace(read_ports::in::PATH, utf8(path));
    inputs.emplace(read_ports::in::RECURSIVE, false);
    inputs.emplace(read_ports::in::TRAVERSAL, std::string("循环"));
    inputs.emplace(read_ports::in::ORDER, std::string("顺序（下一张）"));
    inputs.emplace(read_ports::in::COUNT, std::int64_t(1));
    inputs.emplace(read_ports::in::RELOAD, reload);
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

// One directory per run, next to the temporary directory the test environment already points at.
// Parallel ctest entries must not share it. Assertions end the process with std::exit, which skips
// stack unwinding, so cleanup is registered with atexit as well as the destructor.
std::filesystem::path &temporaryRoot()
{
    static std::filesystem::path value;
    return value;
}

void removeTemporaryRoot()
{
    if (temporaryRoot().empty())
        return;
    std::error_code ignored;
    std::filesystem::remove_all(temporaryRoot(), ignored);
    temporaryRoot().clear();
}

struct TemporaryDirectory
{
    std::filesystem::path path;

    TemporaryDirectory()
    {
        const auto base = std::filesystem::temp_directory_path();
        std::random_device device;
#ifdef _WIN32
        const auto pid = static_cast<unsigned long>(_getpid());
#else
        const auto pid = static_cast<unsigned long>(getpid());
#endif
        static const bool cleanupRegistered = []()
        {
            std::atexit(removeTemporaryRoot);
            return true;
        }();
        static_cast<void>(cleanupRegistered);
        for (int attempt = 0; attempt < 64; ++attempt)
        {
            const auto candidate = base / ("visionrt-image-cache-" + std::to_string(pid) + "-"
                                           + std::to_string(device()) + "-" + std::to_string(attempt));
            std::error_code error;
            if (std::filesystem::create_directories(candidate, error) && !error)
            {
                path            = candidate;
                temporaryRoot() = candidate;
                return;
            }
        }
        throw std::runtime_error("cannot create a unique temporary directory for the image cache test");
    }

    ~TemporaryDirectory() { removeTemporaryRoot(); }

    TemporaryDirectory(const TemporaryDirectory &)            = delete;
    TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;
};

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

void writeFrames(const std::filesystem::path &directory, int count, int width, int height)
{
    std::filesystem::create_directories(directory);
    for (int index = 0; index < count; ++index)
    {
        cv::Mat image(height, width, CV_8UC3);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                image.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<unsigned char>((x + index) % 256),
                                                     static_cast<unsigned char>(y % 256),
                                                     static_cast<unsigned char>((index * 13) % 256));
        writePng(directory / ("frame_" + std::to_string(index) + ".png"), image);
    }
}

// The cache class is independent of the tool: budget, LRU order, protected batches and an item that
// can never fit. All of it fits in a few MiB and belongs in the quick entry.
void cacheBudgetContract()
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

// Small frames, real decodes: warm pass hits, reload decodes once, a failed read leaves the frames
// resident, and a second reader keeps its own cache.
void readerCacheContract()
{
    TemporaryDirectory directory;
    const auto         frames = directory.path / "frames";
    constexpr int      kFrames = 3;
    writeFrames(frames, kFrames, 96, 64);

    ReadImageTool tool(ToolServices{PathResolver(std::filesystem::current_path())});
    const auto    start = testing::decodeCount.load();

    runPass(tool, frames, kFrames);
    const auto afterCold = testing::decodeCount.load();
    REQUIRE(afterCold - start == static_cast<std::size_t>(kFrames),
            "the first pass must decode every small frame exactly once");

    runPass(tool, frames, kFrames);
    REQUIRE(testing::decodeCount.load() == afterCold, "a repeated sequence must be served from the cache");

    const auto reload = tool.execute(readInputs(frames, true));
    REQUIRE(testview::status(reload).code == 0, testview::status(reload).message);
    REQUIRE(testing::decodeCount.load() == afterCold + 1, "reload must bypass the cache exactly once");
    const auto afterReload = testing::decodeCount.load();
    runPass(tool, frames, kFrames);
    REQUIRE(testing::decodeCount.load() == afterReload, "a reloaded frame must be cached again");

    const auto missing = tool.execute(readInputs(directory.path / "missing"));
    REQUIRE(testview::status(missing).code != 0, "reading a missing directory must fail");
    const auto beforeRetry = testing::decodeCount.load();
    runPass(tool, frames, kFrames);
    REQUIRE(testing::decodeCount.load() == beforeRetry, "a failed read must not drop the resident frames");

    const auto   decodesBeforeSecond = testing::decodeCount.load();
    ReadImageTool second(ToolServices{PathResolver(std::filesystem::current_path())});
    runPass(second, frames, kFrames);
    const auto secondCold = testing::decodeCount.load() - decodesBeforeSecond;
    runPass(second, frames, kFrames);
    REQUIRE(secondCold == static_cast<std::size_t>(kFrames),
            "a second instance must not reuse the first instance's cache");
    REQUIRE(testing::decodeCount.load() - decodesBeforeSecond == secondCold,
            "the second instance must cache its own frames");

    std::cout << "[OK] image cache: " << kFrames << " small frames, cold decodes=" << (afterCold - start)
              << ", warm pass and failure retry decoded 0\n";
}

// Explicit batch shape and loop count: the numbers are only comparable when the caller states them.
void readerResidencyBenchmark(int imageCount, int width, int height, int rounds,
                              const std::filesystem::path &evidenceDir)
{
    TemporaryDirectory directory;
    const auto         frames = directory.path / "frames";
    std::filesystem::create_directories(evidenceDir);
    writeFrames(frames, imageCount, width, height);

    ReadImageTool tool(ToolServices{PathResolver(std::filesystem::current_path())});
    const double  baselineMiB    = workingSetMiB();
    const auto    decodesAtStart = testing::decodeCount.load();

    runPass(tool, frames, 2); // warm-up: catalog scan and the first decoded images
    const auto   coldDecodes  = testing::decodeCount.load() - decodesAtStart;
    const double coldMs       = runPass(tool, frames, imageCount - 2);
    const double afterColdMiB = workingSetMiB();
    const auto   decodesAfterCold = testing::decodeCount.load();

    const double warmMs = runPass(tool, frames, imageCount);
    const auto   decodesAfterWarm = testing::decodeCount.load();
    const double afterWarmMiB = workingSetMiB();

    REQUIRE(coldDecodes == 2, "the warm-up must decode exactly the images it was asked for");
    REQUIRE(decodesAfterCold - decodesAtStart == static_cast<std::size_t>(imageCount),
            "the cold pass must decode every image exactly once");
    REQUIRE(decodesAfterWarm == decodesAfterCold, "a repeated sequence must be served from the cache");

    const double loopMs       = runPass(tool, frames, rounds);
    const auto   loopDecodes  = testing::decodeCount.load();
    const double afterLoopMiB = workingSetMiB();
    REQUIRE(loopDecodes == decodesAfterWarm, "a long loop must stay on cached frames");

    const double retainedMiB   = afterColdMiB - baselineMiB;
    const double loopGrowthMiB = afterLoopMiB - afterWarmMiB;
    const double expectedMiB   = static_cast<double>(imageCount) * width * height * 3.0 / (1024.0 * 1024.0);
    // Working-set assertions only make sense for the stated large batch: a 96x64 fixture's delta is
    // below the measurement noise, so a small explicit run reports numbers without pretending.
    if (afterColdMiB > 0.0 && expectedMiB >= 20.0)
    {
        REQUIRE(retainedMiB >= expectedMiB * 0.4, "decoded frames must show up in the working set");
        REQUIRE(retainedMiB <= 256.0, "retention must stay far below the cache budget for this batch");
        REQUIRE(loopGrowthMiB <= 16.0, "a long loop must not grow the working set");
    }

    const auto decodesBeforeSecond = testing::decodeCount.load();
    ReadImageTool second(ToolServices{PathResolver(std::filesystem::current_path())});
    runPass(second, frames, imageCount);
    const auto secondCold = testing::decodeCount.load() - decodesBeforeSecond;
    runPass(second, frames, imageCount);
    const auto secondWarm = testing::decodeCount.load() - decodesBeforeSecond;
    REQUIRE(secondCold == static_cast<std::size_t>(imageCount),
            "a second instance must not reuse the first instance's cache");
    REQUIRE(secondWarm == secondCold, "the second instance must cache its own frames");

    std::ofstream evidence(evidenceDir / "read_cache_evidence.json");
    REQUIRE(evidence.is_open(), "cannot open read_cache_evidence.json");
    evidence.precision(6);
    evidence << std::fixed;
    evidence << "{\n"
             << "  \"cache_budget_bytes\": " << ReadImageTool::kCacheBudgetBytes << ",\n"
             << "  \"images\": " << imageCount << ",\n"
             << "  \"width\": " << width << ",\n"
             << "  \"height\": " << height << ",\n"
             << "  \"rounds\": " << rounds << ",\n"
             << "  \"decoded_bytes_per_image\": " << static_cast<std::size_t>(width) * height * 3 << ",\n"
             << "  \"cold_pass_ms\": " << coldMs << ",\n"
             << "  \"warm_pass_ms\": " << warmMs << ",\n"
             << "  \"cold_ms_per_image\": " << coldMs / (imageCount - 2) << ",\n"
             << "  \"warm_ms_per_image\": " << warmMs / imageCount << ",\n"
             << "  \"loop_ms\": " << loopMs << ",\n"
             << "  \"decodes_cold\": " << (decodesAfterCold - decodesAtStart) << ",\n"
             << "  \"decodes_warm\": " << (decodesAfterWarm - decodesAfterCold) << ",\n"
             << "  \"working_set_baseline_mib\": " << baselineMiB << ",\n"
             << "  \"working_set_after_cold_mib\": " << afterColdMiB << ",\n"
             << "  \"working_set_after_loop_mib\": " << afterLoopMiB << ",\n"
             << "  \"second_instance_cold_decodes\": " << secondCold << ",\n"
             << "  \"second_instance_warm_decodes\": " << (secondWarm - secondCold) << "\n"
             << "}\n";
    REQUIRE(!evidence.bad(), "error writing read_cache_evidence.json");

    std::cout << "[OK] image cache residency: " << imageCount << " x " << width << "x" << height
              << ", cold=" << coldMs << " ms for " << (imageCount - 2) << " frames, warm=" << warmMs
              << " ms for " << imageCount << " frames, loop=" << loopMs << " ms for " << rounds << " frames\n";
    std::cout << "[OK] working set: baseline=" << baselineMiB << " MiB, after cold=" << afterColdMiB
              << " MiB (+" << retainedMiB << "), after " << rounds << "-frame loop=" << afterLoopMiB << " MiB\n";
}

int parseCount(const char *value, const char *name)
{
    try
    {
        const auto parsed = std::stoi(value);
        if (parsed > 0)
            return parsed;
    }
    catch (const std::exception &)
    {
    }
    throw std::runtime_error(std::string(name) + " must be a positive integer");
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        bool                      perf = false;
        int                       images = 12;
        int                       width = 1600;
        int                       height = 900;
        int                       rounds = 120;
        std::filesystem::path     evidenceDir = ".";
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument = argv[index];
            if (argument == "--perf")
                perf = true;
            else if (argument == "--images" && index + 1 < argc)
                images = parseCount(argv[++index], "--images");
            else if (argument == "--width" && index + 1 < argc)
                width = parseCount(argv[++index], "--width");
            else if (argument == "--height" && index + 1 < argc)
                height = parseCount(argv[++index], "--height");
            else if (argument == "--rounds" && index + 1 < argc)
                rounds = parseCount(argv[++index], "--rounds");
            else if (argument == "--evidence" && index + 1 < argc)
                evidenceDir = argv[++index];
            else
                throw std::runtime_error("unknown argument: " + argument);
        }

        cacheBudgetContract();
        readerCacheContract();
        if (perf)
        {
            // The cold pass starts after a two-frame warm-up, so the batch has to be larger than it.
            if (images < 3)
                throw std::runtime_error("--images must be at least 3 for the residency benchmark");
            readerResidencyBenchmark(images, width, height, rounds, evidenceDir);
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 1;
    }
}
