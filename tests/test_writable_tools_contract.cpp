#include "TestRegistry.h"
#include "result_view.h"
#include <visionrt/contracts/Module.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>

int main()
try
{
    using namespace visionrt;
    static_assert(MODULE_ABI_MAJOR == 6);
    const auto *api = visiontoolsModuleApi();
    assert(api && api->abi_major == MODULE_ABI_MAJOR);
    TestRegistry registry;
    assert(api->register_tools(registry).code == 0);
    const auto  &factory = registry.factories.at("visionrt.image.crop");
    ToolServices services;
    ToolStatus   status;
    auto        *instance = factory.create(&services, &status);
    assert(instance && status.code == 0);
    assert(instance->initialize({}).code == 0);

    int           released = 0;
    std::uint8_t *original = nullptr;
    ToolResult    result;
    {
        auto storage
            = std::shared_ptr<std::array<std::uint8_t, 8>>(new std::array<std::uint8_t, 8>{1, 2, 3, 4, 5, 6, 7, 8},
                                                           [&](auto *pixels)
                                                           {
                                                               delete pixels;
                                                               ++released;
                                                           });
        original = storage->data();
        const cv::Mat header(2, 4, CV_8UC1, original);
        const Values  inputs{
             {"images", DataRef<Image>(Image{header, header}, {storage})},
             {"rect", DataRef<ROI>(ROI{{{2, 1, 0, 2, 2, 0}}})}
        };
        result = instance->execute(inputs);
        assert(testview::status(result).code == 0);
        const auto &images = std::get<DataRef<Image>>(result.outputs.at("images")).get();
        assert(images.size() == 2);
        assert(images[0].data == original + 1 && images[1].data == original + 1);
        assert(images[0].step[0] == 4 && images[0].cols == 2 && images[0].rows == 2);
        const auto &origins = std::get<Array>(result.outputs.at("origins"));
        assert((origins.type() == TypeSpec{Type::Array, DType::Float64, 2}));
        assert((origins.get<NestedVector<double, 2>>()
                == NestedVector<double, 2>{
                    {1, 0},
                    {1, 0}
        }));
    }
    assert(instance->release().code == 0);
    factory.destroy(instance);
    assert(released == 0);
    {
        const auto output                      = std::get<DataRef<Image>>(result.outputs.at("images"));
        output.get()[0].at<std::uint8_t>(1, 0) = 91;
        assert(original[5] == 91);
        assert(output.get()[1].at<std::uint8_t>(1, 0) == 91);
        result.outputs.clear();
        assert(released == 0);
    }
    assert(released == 1);
    std::cout << "ABI4 crop views retain external owners and share writable pixels\n";
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
