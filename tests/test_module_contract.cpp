#include <visionrt/contracts/Module.h>
#include "result_view.h"
#include "TestRegistry.h"
#include <visionrt/contracts/Tool.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>
#include <vector>
#include <set>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
// clang-format off
#    include <windows.h>
#    include <psapi.h>
// clang-format on
#    pragma comment(lib, "psapi.lib")

inline std::size_t getProcessRSS()
{
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    {
        return pmc.WorkingSetSize;
    }
    return 0;
}
#else
inline std::size_t getProcessRSS()
{
    return 0;
}
#endif

#define REQUIRE(cond, msg)                                                                                      \
    do                                                                                                          \
    {                                                                                                           \
        if (!(cond))                                                                                            \
        {                                                                                                       \
            std::cerr << "Assertion failed: " << #cond << " (" << msg << ") at line " << __LINE__ << std::endl; \
            std::exit(1);                                                                                       \
        }                                                                                                       \
    }                                                                                                           \
    while (0)

int main()
{
    std::cout << "[TEST] Running test_module_contract..." << std::endl;

    const visionrt::ModuleApi *api = visiontoolsModuleApi();
    REQUIRE(api != nullptr, "visiontoolsModuleApi must return non-null pointer");
    REQUIRE(api->abi_major == visionrt::MODULE_ABI_MAJOR, "ABI major must match");
    REQUIRE(api->struct_bytes == sizeof(visionrt::ModuleApi), "struct_bytes must match");
    REQUIRE(api->register_tools != nullptr, "register_tools must be present");
    TestRegistry registry;
    REQUIRE(api->register_tools(registry).code == 0, "Registration must succeed");
    const std::set<std::string> expected{
        "visionrt.image.read", "visionrt.roi.projection", "visionrt.image.crop",
        "visionrt.image.preprocess", "visionrt.detection.filter", "visiontools.preprocess.adaptive_binary",
        "visiontools.image.rotate", "visiontools.image.brightness"};
    REQUIRE(registry.factories.size() == expected.size(), "Every declared CPU factory must be registered");
    for (const auto &id : expected)
        REQUIRE(registry.factories.contains(id), "Every stable ID must be registered");
    REQUIRE(!registry.factories.contains("AdaptiveBinaryTool"), "Legacy class-name alias must not be registered");
    struct RejectingRegistry final : visionrt::IToolRegistry
    {
        int calls = 0;
        visionrt::ToolStatus add(const visionrt::ToolFactoryDescriptor &) noexcept override
        {
            ++calls;
            return {17, "注册已拒绝"};
        }
    } rejected;
    REQUIRE(api->register_tools(rejected).code == 17 && rejected.calls == 1,
            "Registration must stop and preserve the first rejected add");
    const auto &factory = registry.factories.at("visiontools.preprocess.adaptive_binary");
    visionrt::ToolServices services;
    visionrt::ToolStatus status;
    REQUIRE(factory.create(nullptr, &status) == nullptr && status.code == -3,
            "Missing services must be rejected without creating an instance");
    factory.destroy(nullptr);
    auto *uninitialized = factory.create(&services, nullptr);
    REQUIRE(uninitialized != nullptr, "A null status sink must be supported");
    REQUIRE(uninitialized->release().code == 0, "An uninitialized instance must release safely");
    REQUIRE(uninitialized->release().code == 0, "Release must be idempotent");
    factory.destroy(uninitialized);
    // 5. Lifecycle and Snapshot persistence across execute/release/destroy (VS-05 / NS-05)
    cv::Mat snapshotImage;
    {
        visionrt::IToolInstance *inst = factory.create(&services, &status);
        REQUIRE(inst != nullptr, "Known registry must return instance");

        cv::Mat          gray(60, 60, CV_8UC1, cv::Scalar(150));
        visionrt::Values in1;
        in1.emplace("images", visionrt::DataRef<visionrt::Image>(visionrt::Image{(gray).clone()}));
        in1.emplace("block_size", std::int64_t(5));
        in1.emplace("constant_offset", 2.0);

        auto res1 = inst->execute(in1);
        REQUIRE(testview::status(res1).code == 0, "Execute 1 must succeed");
        REQUIRE(testview::status(res1).message == "运行成功", "Execute 1 success message must be shared");
        const auto *outBatch1 = std::get_if<visionrt::DataRef<visionrt::Image>>(&res1.outputs["images"]);
        REQUIRE(outBatch1 != nullptr && !outBatch1->get().empty(), "Execute 1 must return valid image");
        snapshotImage = outBatch1->get()[0];

        // Execute a second time with different non-uniform data
        cv::Mat gray2(40, 40, CV_8UC1);
        for (int r = 0; r < 40; ++r)
        {
            for (int c = 0; c < 40; ++c)
            {
                gray2.at<unsigned char>(r, c) = static_cast<unsigned char>((r * 5 + c * 3) % 256);
            }
        }
        visionrt::Values in2;
        in2.emplace("images", visionrt::DataRef<visionrt::Image>(visionrt::Image{(gray2).clone()}));
        in2.emplace("block_size", std::int64_t(3));
        in2.emplace("constant_offset", 1.0);
        auto res2 = inst->execute(in2);
        REQUIRE(testview::status(res2).code == 0, "Execute 2 must succeed");
        REQUIRE(testview::status(res2).message == "运行成功", "Execute 2 success message must be shared");

        // Verify snapshot from execute 1 is still completely valid and all pixels match
        const auto &snapMat = snapshotImage;
        REQUIRE(snapMat.rows == 60 && snapMat.cols == 60, "Snapshot 1 dimensions must remain valid");
        for (int r = 0; r < 60; ++r)
        {
            for (int c = 0; c < 60; ++c)
            {
                REQUIRE(snapMat.at<unsigned char>(r, c) == 255, "Every pixel of snapshot 1 must remain 255");
            }
        }

        // Release idempotency: release twice succeeds (NS-05)
        REQUIRE(inst->release().code == 0, "Release 1 must succeed");
        REQUIRE(inst->release().code == 0, "Repeated release must succeed (idempotent)");

        // Snapshot must remain valid after release
        REQUIRE(snapshotImage.rows == 60 && snapMat.at<unsigned char>(0, 0) == 255,
                "Snapshot must remain valid after release");

        // Destroy instance
        factory.destroy(inst);
    }
    // Snapshot must remain valid after instance destruction
    REQUIRE(snapshotImage.rows == 60 && snapshotImage.at<unsigned char>(0, 0) == 255,
            "Snapshot must remain valid after destroy");

    // 6. Concurrency: 2 instances running concurrently with independent states (VS-05 / NS-05)
    // Verify concurrent execution outputs are 100% equal to serial execution (RS-05)
    {
        auto runWorker = [&](int factor, int size) -> std::vector<cv::Mat>
        {
            visionrt::ToolStatus     s;
            visionrt::IToolInstance *inst = factory.create(&services, &s);
            REQUIRE(inst != nullptr, "Create must succeed");

            cv::Mat img(size, size, CV_8UC1);
            for (int r = 0; r < size; ++r)
            {
                for (int c = 0; c < size; ++c)
                {
                    img.at<unsigned char>(r, c) = static_cast<unsigned char>((r * factor + c * 7 + 13) % 256);
                }
            }
            visionrt::Values inputs;
            inputs.emplace("images", visionrt::DataRef<visionrt::Image>(visionrt::Image{(img).clone()}));
            inputs.emplace("block_size", std::int64_t(7));
            inputs.emplace("constant_offset", 3.0);

            std::vector<cv::Mat> roundOutputs;
            for (int i = 0; i < 10; ++i)
            {
                auto res = inst->execute(inputs);
                REQUIRE(testview::status(res).code == 0, "Execute must succeed");
                const auto *b = std::get_if<visionrt::DataRef<visionrt::Image>>(&res.outputs["images"]);
                REQUIRE(b != nullptr && !b->get().empty(), "Output images must be non-empty");
                REQUIRE(b->get()[0].rows == size && b->get()[0].cols == size, "Concurrent output size match");
                roundOutputs.push_back(b->get()[0].clone());
            }

            inst->release();
            factory.destroy(inst);
            return roundOutputs;
        };

        // 1. Run serially to capture baseline outputs
        std::vector<cv::Mat> serial1 = runWorker(3, 32);
        std::vector<cv::Mat> serial2 = runWorker(11, 48);

        // 2. Run concurrently across two independent threads
        auto f1 = std::async(std::launch::async, runWorker, 3, 32);
        auto f2 = std::async(std::launch::async, runWorker, 11, 48);

        std::vector<cv::Mat> conc1 = f1.get();
        std::vector<cv::Mat> conc2 = f2.get();

        // 3. Bitwise full equality comparison across all rounds between serial and concurrent outputs (RS-05)
        REQUIRE(conc1.size() == 10 && conc2.size() == 10, "Must have 10 rounds of output");
        for (int i = 0; i < 10; ++i)
        {
            REQUIRE(cv::countNonZero(conc1[i] != serial1[i]) == 0,
                    "Concurrent output 1 round must match serial round completely");
            REQUIRE(cv::countNonZero(conc2[i] != serial2[i]) == 0,
                    "Concurrent output 2 round must match serial round completely");
            REQUIRE(cv::countNonZero(conc1[i]) > 0 && cv::countNonZero(conc1[i]) < 32 * 32,
                    "Round 1 output must be non-trivial pattern");
            REQUIRE(cv::countNonZero(conc2[i]) > 0 && cv::countNonZero(conc2[i]) < 48 * 48,
                    "Round 2 output must be non-trivial pattern");
        }
    }

    // 7. Lifecycle: 100 consecutive executions on the same instance with multi-point RSS sampling (NS-05 / RS-05)
    {
        std::size_t              rssBaseline = getProcessRSS();
        visionrt::ToolStatus     s;
        visionrt::IToolInstance *inst = factory.create(&services, &s);
        REQUIRE(inst != nullptr, "Create for 100 consecutive runs must succeed");

        cv::Mat nonUniform(64, 64, CV_8UC1);
        for (int r = 0; r < 64; ++r)
        {
            for (int c = 0; c < 64; ++c)
            {
                nonUniform.at<unsigned char>(r, c) = static_cast<unsigned char>((r * 7 + c * 13) % 256);
            }
        }
        visionrt::Values inputs;
        inputs.emplace("images", visionrt::DataRef<visionrt::Image>(visionrt::Image{(nonUniform).clone()}));
        inputs.emplace("block_size", std::int64_t(5));
        inputs.emplace("constant_offset", 1.0);

        // Warmup 5 runs
        for (int w = 0; w < 5; ++w)
        {
            auto res = inst->execute(inputs);
            REQUIRE(testview::status(res).code == 0, "Warmup execute must succeed");
        }
        std::size_t rssWarmup = getProcessRSS();

        std::vector<double> latencies_us;
        latencies_us.reserve(100);
        std::size_t rss25 = 0, rss50 = 0, rss75 = 0, rss100 = 0;

        for (int i = 0; i < 100; ++i)
        {
            auto t0  = std::chrono::high_resolution_clock::now();
            auto res = inst->execute(inputs);
            auto t1  = std::chrono::high_resolution_clock::now();
            REQUIRE(testview::status(res).code == 0, "100 consecutive execute must succeed");
            latencies_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());

            if (i == 24)
                rss25 = getProcessRSS();
            else if (i == 49)
                rss50 = getProcessRSS();
            else if (i == 74)
                rss75 = getProcessRSS();
            else if (i == 99)
                rss100 = getProcessRSS();
        }

        std::sort(latencies_us.begin(), latencies_us.end());
        double p50 = latencies_us[50];
        double p95 = latencies_us[95];

        inst->release();
        std::size_t rssPostRelease = getProcessRSS();
        factory.destroy(inst);
        std::size_t rssPostDestroy = getProcessRSS();

        std::cout << "[METRICS] 100-run: p50=" << p50 << "us, p95=" << p95 << "us" << std::endl;
        if (rssBaseline == 0)
        {
            std::cout << "[METRICS] RSS sequence: unavailable" << std::endl;
        }
        else
        {
            std::cout << "[METRICS] RSS sequence (bytes): baseline=" << rssBaseline << ", post_warmup=" << rssWarmup
                      << ", loop25=" << rss25 << ", loop50=" << rss50 << ", loop75=" << rss75 << ", loop100=" << rss100
                      << ", post_release=" << rssPostRelease << ", post_destroy=" << rssPostDestroy << std::endl;
            // Plateau check: verify memory stabilization across sample windows (F79 / YR-05 / BS-03)
            const auto evaluateRssPlateau
                = [](std::size_t r25, std::size_t r50, std::size_t r75, std::size_t r100) -> bool
            {
                const bool monotonicGrowthLeak
                    = (r50 >= r25 + 256 * 1024) && (r75 >= r50 + 256 * 1024) && (r100 >= r75 + 256 * 1024);
                if (monotonicGrowthLeak)
                    return false;
                std::size_t maxDiff = 0;
                if (r50 >= r25 && (r50 - r25) > maxDiff)
                    maxDiff = r50 - r25;
                if (r75 >= r50 && (r75 - r50) > maxDiff)
                    maxDiff = r75 - r50;
                if (r100 >= r75 && (r100 - r75) > maxDiff)
                    maxDiff = r100 - r75;
                return maxDiff < 1024 * 1024;
            };

            // 1. Counterexample verification: continuous +512KiB and +256KiB/window growth must be rejected
            REQUIRE(
                !evaluateRssPlateau(10000000, 10000000 + 512 * 1024, 10000000 + 1024 * 1024, 10000000 + 1536 * 1024),
                "Plateau check must reject continuous +512KiB/window growth counterexample");
            REQUIRE(!evaluateRssPlateau(10000000, 10000000 + 256 * 1024, 10000000 + 512 * 1024, 10000000 + 768 * 1024),
                    "Plateau check must reject continuous +256KiB/window growth counterexample");

            // 2. Real working set verification
            REQUIRE(evaluateRssPlateau(rss25, rss50, rss75, rss100), "Working set must reach plateau after warmup");
            std::cout << "[INFO] RSS sequence analysis: working set reached plateau after warmup." << std::endl;
        }
    }

    // 8. 20 separate create/destroy cycles (NS-05)
    {
        for (int cycle = 0; cycle < 20; ++cycle)
        {
            visionrt::ToolStatus     s;
            visionrt::IToolInstance *inst = factory.create(&services, &s);
            REQUIRE(inst != nullptr, "Cycle create must succeed");
            inst->release();
            factory.destroy(inst);
        }
    }

    std::cout << "[PASS] test_module_contract passed." << std::endl;
    return 0;
}
