#include <visionrt/sdk/Module.h>
#include <visionrt/sdk/Tool.h>

#include <iostream>
#include <cassert>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

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

    // 4. destroy(nullptr) must be safe and no-op
    {
        api->destroy(nullptr);
    }

    // 5. Lifecycle and Snapshot persistence across execute/release/destroy (VS-05)
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

        // Execute a second time with different data
        cv::Mat gray2(40, 40, CV_8UC1, cv::Scalar(50));
        visionrt::Values in2;
        in2.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(gray2)}));
        in2.emplace("block_size", std::int64_t(3));
        in2.emplace("constant_offset", 1.0);
        auto res2 = inst->execute(in2);
        REQUIRE(res2.status.code == 0, "Execute 2 must succeed");

        // Verify snapshot from execute 1 is still valid and readable
        REQUIRE(snapshotImage.rows() == 60 && snapshotImage.cols() == 60, "Snapshot 1 dimensions must remain valid");
        REQUIRE(snapshotImage.readOnlyMat().at<unsigned char>(0, 0) == 255, "Snapshot 1 pixels must remain readable");

        // Release idempotency
        REQUIRE(inst->release().code == 0, "Release 1 must succeed");
        REQUIRE(inst->release().code == 0, "Repeated release must succeed (idempotent)");

        // Snapshot must remain valid after release
        REQUIRE(snapshotImage.rows() == 60 && snapshotImage.readOnlyMat().at<unsigned char>(0, 0) == 255, "Snapshot must remain valid after release");

        // Destroy instance
        api->destroy(inst);
    }
    // Snapshot must remain valid after instance destruction
    REQUIRE(snapshotImage.rows() == 60 && snapshotImage.readOnlyMat().at<unsigned char>(0, 0) == 255, "Snapshot must remain valid after destroy");

    // 6. Concurrency: 2 instances running concurrently with independent states (VS-05)
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

    // 7. Stress loop: 100 executions and 20 create/destroy cycles (VS-05)
    {
        for (int cycle = 0; cycle < 20; ++cycle) {
            visionrt::ToolStatus s;
            visionrt::IToolInstance* inst = api->create("visiontools.preprocess.adaptive_binary", &services, &s);
            REQUIRE(inst != nullptr, "Cycle create must succeed");

            cv::Mat img(32, 32, CV_8UC1, cv::Scalar(128));
            visionrt::Values inputs;
            inputs.emplace("images", visionrt::ImageBatch({visionrt::OwnedImage::copyOf(img)}));
            inputs.emplace("block_size", std::int64_t(5));
            inputs.emplace("constant_offset", 1.0);

            for (int i = 0; i < 5; ++i) { // 20 cycles * 5 = 100 executions
                auto res = inst->execute(inputs);
                REQUIRE(res.status.code == 0, "Stress loop execute must succeed");
            }

            inst->release();
            api->destroy(inst);
        }
    }

    std::cout << "[PASS] test_module_contract passed." << std::endl;
    return 0;
}
