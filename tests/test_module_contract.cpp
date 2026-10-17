#include <visionrt/sdk/Module.h>
#include <visionrt/sdk/Tool.h>

#include <iostream>
#include <cassert>
#include <chrono>
#include <future>
#include <thread>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
inline std::size_t getProcessRSS() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
    return 0;
}
#else
inline std::size_t getProcessRSS() { return 0; }
#endif

#define REQUIRE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: " << #cond << " (" << msg << ") at line " << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while(0)

int main() {
    std::cout << "[TEST] Running test_module_contract..." << std::endl;

    const visionrt::ModuleApi* api = visiontoolsModuleApi();
    REQUIRE(api != nullptr, "visiontoolsModuleApi must return non-null pointer");
    REQUIRE(api->abi_major == visionrt::MODULE_ABI_MAJOR, "ABI major must match");
    REQUIRE(api->struct_bytes == sizeof(visionrt::ModuleApi), "struct_bytes must match");
    REQUIRE(api->create != nullptr, "create pointer must be non-null");
    REQUIRE(api->destroy != nullptr, "destroy pointer must be non-null");

    visionrt::ToolServices services;
    visionrt::ToolStatus status;

    // 1. Unknown registry_name -> returns nullptr and status code -3
    {
        visionrt::IToolInstance* inst = api->create("non.existent.registry", &services, &status);
        REQUIRE(inst == nullptr, "Unknown registry must return nullptr");
        REQUIRE(status.code == -3, "Unknown registry must set status code -3");
        REQUIRE(!status.message.empty(), "Status message must explain error");
    }

    // 2. Null registry_name -> returns nullptr and status code -3
    {
        visionrt::IToolInstance* inst = api->create(nullptr, &services, &status);
        REQUIRE(inst == nullptr, "Null registry must return nullptr");
        REQUIRE(status.code == -3, "Null registry must set status code -3");
    }

    // 3. Null services pointer -> safely handled without crash
    {
        visionrt::IToolInstance* inst = api->create("visiontools.preprocess.adaptive_binary", nullptr, &status);
        REQUIRE(inst != nullptr, "Null services must be safely tolerated with default services");
        api->destroy(inst);
    }

    // 4. destroy(nullptr) must be safe and no-op (NS-05)
    {
        api->destroy(nullptr);
    }

    // 5. Lifecycle and Snapshot persistence across execute/release/destroy (VS-05 / NS-05)
    visionrt::OwnedImage snapshotImage;
    {
        visionrt::IToolInstance* inst = api->create("visiontools.preprocess.adaptive_binary", &services, &status);
        REQUIRE(inst != nullptr, "Known registry must return instance");

        cv::Mat gray(60, 60, CV_8UC1, cv::Scalar(150));
        visionrt::Values in1;
        in1.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(gray)}));
        in1.emplace("block_size", std::int64_t(5));
        in1.emplace("constant_offset", 2.0);

        auto res1 = inst->execute(in1);
        REQUIRE(res1.status.code == 0, "Execute 1 must succeed");
        const auto* outBatch1 = std::get_if<visionrt::ImageBatch>(&res1.outputs["images"]);
        REQUIRE(outBatch1 != nullptr && !outBatch1->images().empty(), "Execute 1 must return valid image");
        snapshotImage = outBatch1->images()[0];

        // Execute a second time with different non-uniform data
        cv::Mat gray2(40, 40, CV_8UC1);
        for (int r = 0; r < 40; ++r) {
            for (int c = 0; c < 40; ++c) {
                gray2.at<unsigned char>(r, c) = static_cast<unsigned char>((r * 5 + c * 3) % 256);
            }
        }
        visionrt::Values in2;
        in2.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(gray2)}));
        in2.emplace("block_size", std::int64_t(3));
        in2.emplace("constant_offset", 1.0);
        auto res2 = inst->execute(in2);
        REQUIRE(res2.status.code == 0, "Execute 2 must succeed");

        // Verify snapshot from execute 1 is still completely valid and all pixels match
        const auto& snapMat = snapshotImage.readOnlyMat();
        REQUIRE(snapMat.rows == 60 && snapMat.cols == 60, "Snapshot 1 dimensions must remain valid");
        for (int r = 0; r < 60; ++r) {
            for (int c = 0; c < 60; ++c) {
                REQUIRE(snapMat.at<unsigned char>(r, c) == 255, "Every pixel of snapshot 1 must remain 255");
            }
        }

        // Release idempotency: release twice succeeds (NS-05)
        REQUIRE(inst->release().code == 0, "Release 1 must succeed");
        REQUIRE(inst->release().code == 0, "Repeated release must succeed (idempotent)");

        // Snapshot must remain valid after release
        REQUIRE(snapshotImage.rows() == 60 && snapMat.at<unsigned char>(0, 0) == 255, "Snapshot must remain valid after release");

        // Destroy instance
        api->destroy(inst);
    }
    // Snapshot must remain valid after instance destruction
    REQUIRE(snapshotImage.rows() == 60 && snapshotImage.readOnlyMat().at<unsigned char>(0, 0) == 255, "Snapshot must remain valid after destroy");

    // 6. Concurrency: 2 instances running concurrently with independent states (VS-05 / NS-05)
    {
        auto worker = [&](int val, int size) {
            visionrt::ToolStatus s;
            visionrt::IToolInstance* inst = api->create("visiontools.preprocess.adaptive_binary", &services, &s);
            REQUIRE(inst != nullptr, "Concurrent create must succeed");

            cv::Mat img(size, size, CV_8UC1, cv::Scalar(val));
            visionrt::Values inputs;
            inputs.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(img)}));
            inputs.emplace("block_size", std::int64_t(5));
            inputs.emplace("constant_offset", 0.0);

            for (int i = 0; i < 10; ++i) {
                auto res = inst->execute(inputs);
                REQUIRE(res.status.code == 0, "Concurrent execute must succeed");
                const auto* b = std::get_if<visionrt::ImageBatch>(&res.outputs["images"]);
                REQUIRE(b->images()[0].rows() == size, "Concurrent output size match");
            }

            inst->release();
            api->destroy(inst);
            return true;
        };

        auto f1 = std::async(std::launch::async, worker, 100, 32);
        auto f2 = std::async(std::launch::async, worker, 200, 48);
        REQUIRE(f1.get() && f2.get(), "Both concurrent worker threads must complete successfully");
    }

    // 7. Lifecycle: 100 consecutive executions on the same instance (NS-05)
    {
        std::size_t rssBefore = getProcessRSS();
        visionrt::ToolStatus s;
        visionrt::IToolInstance* inst = api->create("visiontools.preprocess.adaptive_binary", &services, &s);
        REQUIRE(inst != nullptr, "Create for 100 consecutive runs must succeed");

        cv::Mat nonUniform(64, 64, CV_8UC1);
        for (int r = 0; r < 64; ++r) {
            for (int c = 0; c < 64; ++c) {
                nonUniform.at<unsigned char>(r, c) = static_cast<unsigned char>((r * 7 + c * 13) % 256);
            }
        }
        visionrt::Values inputs;
        inputs.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(nonUniform)}));
        inputs.emplace("block_size", std::int64_t(5));
        inputs.emplace("constant_offset", 1.0);

        // Warmup 5 runs
        for (int w = 0; w < 5; ++w) {
            auto res = inst->execute(inputs);
            REQUIRE(res.status.code == 0, "Warmup execute must succeed");
        }

        std::vector<double> latencies_us;
        latencies_us.reserve(100);

        for (int i = 0; i < 100; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = inst->execute(inputs);
            auto t1 = std::chrono::high_resolution_clock::now();
            REQUIRE(res.status.code == 0, "100 consecutive execute must succeed");
            latencies_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }

        std::sort(latencies_us.begin(), latencies_us.end());
        double p50 = latencies_us[50];
        double p95 = latencies_us[95];

        inst->release();
        api->destroy(inst);
        std::size_t rssAfter = getProcessRSS();

        std::cout << "[METRICS] 100-run: p50=" << p50 << "us, p95=" << p95
                  << "us, RSS_before=" << rssBefore << "B, RSS_after=" << rssAfter << "B" << std::endl;
    }

    // 8. 20 separate create/destroy cycles (NS-05)
    {
        for (int cycle = 0; cycle < 20; ++cycle) {
            visionrt::ToolStatus s;
            visionrt::IToolInstance* inst = api->create("visiontools.preprocess.adaptive_binary", &services, &s);
            REQUIRE(inst != nullptr, "Cycle create must succeed");
            inst->release();
            api->destroy(inst);
        }
    }

    std::cout << "[PASS] test_module_contract passed." << std::endl;
    return 0;
}
