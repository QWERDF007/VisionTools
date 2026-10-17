#include <visionrt/sdk/Module.h>
#include <iostream>
#include <cassert>

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

    // 2. Known registry_name -> returns instance and status code 0
    {
        visionrt::IToolInstance* inst = api->create("visiontools.preprocess.adaptive_binary", &services, &status);
        REQUIRE(inst != nullptr, "Known registry must return instance");
        REQUIRE(status.code == 0, "Known registry must set status code 0");

        // Lifecycle: initialize -> execute -> release -> destroy
        visionrt::Values emptyInit;
        visionrt::ToolStatus initStatus = inst->initialize(emptyInit);
        REQUIRE(initStatus.code == 0, "Empty init must succeed");

        visionrt::Values emptyInputs;
        emptyInputs.emplace("images", visionrt::ImageBatch{});
        visionrt::ToolResult execResult = inst->execute(emptyInputs);
        REQUIRE(execResult.status.code == 0, "B=0 execute must succeed");

        visionrt::ToolStatus relStatus = inst->release();
        REQUIRE(relStatus.code == 0, "Release must succeed");

        api->destroy(inst);
    }

    // 3. destroy(nullptr) must be safe and no-op
    {
        api->destroy(nullptr);
    }

    std::cout << "[PASS] test_module_contract passed." << std::endl;
    return 0;
}
