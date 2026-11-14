#include "TestRegistry.h"
#include "ToolResultAssertions.h"
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
    static_assert(MODULE_ABI_MAJOR == 10);
    const auto *api = visiontoolsModuleApi();
    assert(api && api->abi_major == MODULE_ABI_MAJOR);
    TestRegistry registry;
    assert(api->register_tools(registry).code == 0);
    const auto  &factory = registry.factories.at("CropImage");
    ToolStatus   status;
    auto        *instance = factory.create(&status);
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
             {1, DataRef<Image>(Image{header, header}, {storage})},
             {2, DataRef<ROI>(ROI{{{2, 1, 0, 2, 2, 0}}})}
        };
        result = instance->execute(inputs);
        assert(testview::status(result).code == 0);
        const auto &images = std::get<DataRef<Image>>(result.outputs.at(2)).get();
        assert(images.size() == 2);
        assert(images[0].data == original + 1 && images[1].data == original + 1);
        assert(images[0].step[0] == 4 && images[0].cols == 2 && images[0].rows == 2);
        const auto &origins = std::get<Array>(result.outputs.at(3));
        assert((origins.type() == TypeSpec{Type::Array, DType::Float64, 2}));
        assert((nestedOf<NestedVector<double, 2>>(origins)
                == NestedVector<double, 2>{
                    {1, 0},
                    {1, 0}
        }));
    }
    assert(instance->release().code == 0);
    factory.destroy(instance);
    assert(released == 0);
    {
        const auto output                      = std::get<DataRef<Image>>(result.outputs.at(2));
        output.get()[0].at<std::uint8_t>(1, 0) = 91;
        assert(original[5] == 91);
        assert(output.get()[1].at<std::uint8_t>(1, 0) == 91);
        result.outputs.clear();
        assert(released == 0);
    }
    assert(released == 1);

    {
        // T41-02: testview helpers must be strict: at() + std::get(), no default fallbacks
        ToolResult incompleteCode{{{kStatusMessagePort, std::string("ok")}}, "detail"};
        bool caught = false;
        try { testview::code(incompleteCode); }
        catch (const std::out_of_range &) { caught = true; }
        assert(caught);

        ToolResult wrongTypeCode{{{kStatusCodePort, std::string("bad")}, {kStatusMessagePort, std::string("ok")}}, "detail"};
        caught = false;
        try { testview::code(wrongTypeCode); }
        catch (const std::bad_variant_access &) { caught = true; }
        assert(caught);

        ToolResult incompleteMsg{{{kStatusCodePort, std::int64_t(0)}}, "detail"};
        caught = false;
        try { testview::message(incompleteMsg); }
        catch (const std::out_of_range &) { caught = true; }
        assert(caught);

        ToolResult wrongTypeMsg{{{kStatusCodePort, std::int64_t(0)}, {kStatusMessagePort, std::int64_t(123)}}, "detail"};
        caught = false;
        try { testview::message(wrongTypeMsg); }
        catch (const std::bad_variant_access &) { caught = true; }
        assert(caught);

        ToolResult valid{{{kStatusCodePort, std::int64_t(7)}, {kStatusMessagePort, std::string("fail")}}, "d"};
        const auto st = testview::status(valid);
        assert(st.code == 7);
        assert(st.message == "fail");
        assert(st.detail == "d");

        ToolResult withExtra{{{kStatusCodePort, std::int64_t(0)}, {kStatusMessagePort, std::string("")}, {9, std::int64_t(1)}, {7, std::int64_t(2)}}, ""};
        assert(testview::businessCount(withExtra) == 2);
    }

    std::cout << "ABI9 crop views retain external owners and share writable pixels\n";
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
