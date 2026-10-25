#include "src/preprocess/ReadImageTool.h"
#include "TestSupport.h"
#include "MatAllocationProbe.h"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

using namespace visionrt;
using namespace visiontools::preprocess;
using namespace visiontools::tests;

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
    inputs.emplace(read_ports::in::MODE, std::string("file"));
    inputs.emplace(read_ports::in::PATH, std::string(utf8(path)));
    inputs.emplace(read_ports::in::COLOR, std::string("bgr"));
    inputs.emplace(read_ports::in::LIMIT, std::int64_t(0));
    inputs.emplace(read_ports::in::MAX_BATCH_BYTES, std::int64_t(0));
    inputs.emplace(read_ports::in::PATHS_IN, Array(DataRef<NestedVector<std::string, 1>>(NestedVector<std::string, 1>{})));
    inputs.emplace(read_ports::in::READ_MODE, std::string("all"));
    inputs.emplace(read_ports::in::RECURSIVE, false);
    inputs.emplace(read_ports::in::EXTENSIONS,
                   Array(DataRef<NestedVector<std::string, 1>>(NestedVector<std::string, 1>{"png"})));
    inputs.emplace(read_ports::in::END_POLICY, std::string("wrap"));
    return inputs;
}

void writePng(const std::filesystem::path &path, const cv::Mat &image)
{
    std::vector<unsigned char> encoded;
    REQUIRE(cv::imencode(".png", image, encoded), "Image fixture must encode");
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
    REQUIRE(stream.good(), "Image fixture must be written");
}

} // namespace

void testReadImageTool()
{
    std::cout << "[TEST] Running testReadImageTool..." << std::endl;
    ToolServices  services{PathResolver(std::filesystem::current_path())};
    ReadImageTool tool(services);

    static_assert(std::string_view(read_ports::in::MAX_BATCH_BYTES) == "max_batch_bytes");
    static_assert(std::string_view(read_ports::out::PATHS) == "paths");

    const auto directory       = std::filesystem::current_path() / "test_read_directory";
    const auto files           = directory / "files";
    const auto budgetDirectory = directory / "budget";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(files);
    std::filesystem::create_directories(budgetDirectory);

    // 1. Missing / empty / nonexistent path
    {
        Values inputs = readInputs("");
        auto   res    = tool.execute(inputs);
        requireFailure(res, "文件路径输入为空", "Empty path must fail");
        REQUIRE(res.status.message.find("为空") != std::string::npos, "Empty path must be Chinese");
    }
    {
        Values inputs = readInputs(files / "missing.png");
        auto   res    = tool.execute(inputs);
        requireFailure(res, "文件不存在", "Non-existent path must fail");
        REQUIRE(res.status.message.find("missing.png") != std::string::npos, "Error must include the file path");
    }

    // 2. Valid read
    {
        const auto path = files / "valid.png";
        writePng(path, cv::Mat(24, 24, CV_8UC3, cv::Scalar(10, 20, 30)));
        Values inputs = readInputs(path);
        auto   res    = tool.execute(inputs);
        requireSuccess(res, "Valid image read must succeed");
        const auto *batch = std::get_if<DataRef<Image>>(&res.outputs[read_ports::out::IMAGES]);
        REQUIRE(batch && batch->get().size() == 1, "Must read 1 image");
        REQUIRE(batch->get()[0].cols == 24 && batch->get()[0].rows == 24, "Image dims must match 24x24");
        const auto paths = std::get<Array>(res.outputs[read_ports::out::PATHS]).get<std::vector<std::string>>();
        REQUIRE(paths[0].ends_with(utf8(path)), "Returned path must be the resolved file path");
    }

    // 3. Negative limit / budget
    {
        Values inputs = readInputs(files / "valid.png");
        inputs.insert_or_assign(read_ports::in::LIMIT, std::int64_t(-1));
        auto res = tool.execute(inputs);
        requireFailure(res, "limit", "Negative image limit must be rejected");
        REQUIRE(res.status.message.find("非负") != std::string::npos, "Limit error must state non-negative");
    }
    {
        Values inputs = readInputs(files / "valid.png");
        inputs.insert_or_assign(read_ports::in::MAX_BATCH_BYTES, std::int64_t(-1));
        auto res = tool.execute(inputs);
        requireFailure(res, "max_batch_bytes", "Negative budget must be rejected");
        REQUIRE(res.status.message.find("非负") != std::string::npos, "Budget error must state non-negative");
    }

    // 4. Empty file
    {
        const auto path = files / "empty.png";
        std::ofstream(path, std::ios::binary).close();
        Values inputs = readInputs(path);
        auto   res    = tool.execute(inputs);
        requireFailure(res, "空", "Empty image file must fail");
        REQUIRE(res.status.message.find("empty.png") != std::string::npos, "Error must include the file path");
    }

    // 5. Controlled short read: truncate after the size check, before stream.read
    {
        const auto path = files / "short.png";
        writePng(path, cv::Mat(24, 24, CV_8UC3, cv::Scalar(1, 2, 3)));
        testing::beforeRead = [](const std::filesystem::path &target)
        {
            std::ofstream truncated(target, std::ios::binary | std::ios::trunc);
            truncated << "x";
        };
        Values inputs       = readInputs(path);
        auto   res          = tool.execute(inputs);
        testing::beforeRead = nullptr;
        requireFailure(res, "读取图像文件不完整", "Short read must not decode as a normal image");
        REQUIRE(res.status.message.find("short.png") != std::string::npos,
                "Short-read error must include the file path");
        REQUIRE(isMissing(res.outputs[read_ports::out::IMAGES]), "Short read must not publish partial images");
    }

    // 6. Directory order and explicit limit
    {
        writePng(directory / "b.png", cv::Mat(8, 8, CV_8UC1, cv::Scalar(2)));
        writePng(directory / "a.png", cv::Mat(8, 8, CV_8UC1, cv::Scalar(1)));
        writePng(directory / "c.png", cv::Mat(8, 8, CV_8UC1, cv::Scalar(3)));
        Values inputs = readInputs(directory);
        inputs.insert_or_assign(read_ports::in::MODE, std::string("directory"));
        inputs.insert_or_assign(read_ports::in::COLOR, std::string("gray"));
        inputs.insert_or_assign(read_ports::in::LIMIT, std::int64_t(2));
        auto res = tool.execute(inputs);
        requireSuccess(res, "Directory read must succeed");
        const auto paths = std::get<Array>(res.outputs[read_ports::out::PATHS]).get<std::vector<std::string>>();
        REQUIRE(paths.size() == 2, "Explicit limit must be honored");
        REQUIRE(paths[0].ends_with("a.png") && paths[1].ends_with("b.png"), "Directory order must be lexical");
    }

    // 7. Cumulative decoded-byte budget
    {
        writePng(budgetDirectory / "a.png", cv::Mat(24, 24, CV_8UC3, cv::Scalar(1, 1, 1)));
        writePng(budgetDirectory / "b.png", cv::Mat(32, 16, CV_8UC3, cv::Scalar(2, 2, 2)));
        Values inputs = readInputs(budgetDirectory);
        inputs.insert_or_assign(read_ports::in::MODE, std::string("directory"));
        inputs.insert_or_assign(read_ports::in::COLOR, std::string("bgr"));
        inputs.insert_or_assign(read_ports::in::MAX_BATCH_BYTES, std::int64_t(2000));
        MatAllocationProbe memory;
        auto res = tool.execute(inputs);
        requireFailure(res, "预算超限", "Cumulative decoded-byte budget must fail the batch");
        REQUIRE(res.status.message.find("单张解码瞬时峰值") != std::string::npos,
                "Budget error must explain the single-image peak limit");
        REQUIRE(res.status.message.find("b.png") != std::string::npos, "Budget error must name the failing file");
        REQUIRE(isMissing(res.outputs[read_ports::out::IMAGES]), "Budget failure must not publish a truncated batch");
        const auto failed = memory.snapshot(CV_8UC3);
        REQUIRE(failed.allocations == 2 && failed.releases == 2 && failed.liveBytes == 0,
                "Budget rejection must release both decoded images");
        REQUIRE(failed.peakBytes == 3264,
                "Mixed-resolution decode peak must include the retained image and the next decoded image");
        inputs.insert_or_assign(read_ports::in::MAX_BATCH_BYTES, std::int64_t(3264));
        res = tool.execute(inputs);
        requireSuccess(res, "Exact cumulative decoded-byte budget must succeed");
        const auto &decoded = std::get<DataRef<Image>>(res.outputs.at(read_ports::out::IMAGES)).get();
        REQUIRE(decoded.size() == 2 && decoded[0].size() == cv::Size(24, 24) && decoded[1].size() == cv::Size(16, 32),
                "Mixed image resolutions must be preserved");
        const auto retained = memory.snapshot(CV_8UC3);
        REQUIRE(retained.allocations == 4 && retained.releases == 2 && retained.liveBytes == 3264,
                "Successful read must retain exactly its decoded pixels without an extra payload copy");
        res                 = ToolResult{};
        const auto released = memory.snapshot(CV_8UC3);
        REQUIRE(released.allocations == released.releases && released.liveBytes == 0,
                "Dropping the read result must free every decoded allocation");
        inputs.insert_or_assign(read_ports::in::MAX_BATCH_BYTES, std::int64_t(3263));
        res = tool.execute(inputs);
        requireFailure(res, "预算超限", "One byte below the cumulative decoded budget must fail");
        const auto final = memory.snapshot(CV_8UC3);
        REQUIRE(final.allocations == 6 && final.releases == 6 && final.liveBytes == 0,
                "Repeated exact-budget rejection must not accumulate decoded payloads");
        std::cout << "[ALLOCATIONS] decode mixed=24x24+32x16 BGR budget=3264 peak_decoded_bytes=" << final.peakBytes
                  << " allocations=" << final.allocations << '/' << final.releases << " live_bytes=" << final.liveBytes
                  << std::endl;
    }

    // 8. Oversized file is rejected before allocation (Windows sparse file)
#ifdef _WIN32
    {
        const auto path   = files / "oversized.png";
        HANDLE     handle = CreateFileW(path.wstring().c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                        FILE_ATTRIBUTE_NORMAL, nullptr);
        REQUIRE(handle != INVALID_HANDLE_VALUE, "Oversized fixture must be created");
        LARGE_INTEGER size{};
        size.QuadPart = (static_cast<LONGLONG>(std::numeric_limits<int>::max()) + 4096);
        REQUIRE(SetFilePointerEx(handle, size, nullptr, FILE_BEGIN), "Oversized fixture must seek");
        REQUIRE(SetEndOfFile(handle), "Oversized fixture must set its length");
        CloseHandle(handle);
        Values inputs = readInputs(path);
        auto   res    = tool.execute(inputs);
        requireFailure(res, "超出可表示范围", "Oversized image must be rejected before allocation");
        std::filesystem::remove(path);
    }
#endif

    std::filesystem::remove_all(directory);
    std::cout << "[PASS] testReadImageTool passed." << std::endl;
}
