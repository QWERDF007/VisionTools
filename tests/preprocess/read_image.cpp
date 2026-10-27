#include "src/preprocess/ReadImageTool.h"
#include "TestSupport.h"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

Values defaultInputs(const std::filesystem::path &path)
{
    Values inputs;
    inputs.emplace(read_ports::in::PATH, std::string(utf8(path)));
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
    REQUIRE(cv::imencode(".png", image, encoded), "Image fixture must encode");
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
    REQUIRE(stream.good(), "Image fixture must be written");
}

} // namespace

void testReadImageTool()
{
    std::cout << "[TEST] Running testReadImageTool (S-READ-01 ~ S-READ-05)..." << std::endl;
    ToolServices services{PathResolver(std::filesystem::current_path())};
    ReadImageTool tool(services);

    const auto directory = std::filesystem::current_path() / "test_read_image_suite";
    const auto filesDir  = directory / "files";
    const auto subDir    = filesDir / "sub";
    const auto emptyDir  = directory / "empty";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(filesDir);
    std::filesystem::create_directories(subDir);
    std::filesystem::create_directories(emptyDir);

    // ==========================================
    // 1. S-READ-01: 基础契约、空路径与无效参数
    // ==========================================
    {
        // 1.1 空路径必须失败
        Values inputs = defaultInputs("");
        auto   res    = tool.execute(inputs);
        requireFailure(res, "图像路径输入为空", "Empty path must fail");

        // 1.2 不存在的路径必须报错
        inputs = defaultInputs(filesDir / "nonexistent.png");
        res    = tool.execute(inputs);
        requireFailure(res, "图像路径不存在", "Non-existent path must fail");

        // 1.3 count < 1 拒绝
        inputs = defaultInputs(filesDir);
        inputs.insert_or_assign(read_ports::in::COUNT, std::int64_t(0));
        res = tool.execute(inputs);
        requireFailure(res, "count", "count < 1 must be rejected");

        // 1.4 无效遍历模式拒绝
        inputs = defaultInputs(filesDir);
        inputs.insert_or_assign(read_ports::in::TRAVERSAL, std::string("random"));
        res = tool.execute(inputs);
        requireFailure(res, "traversal", "Invalid traversal must be rejected");

        // 1.5 空目录明确失败
        inputs = defaultInputs(emptyDir);
        res    = tool.execute(inputs);
        requireFailure(res, "没有可用图像文件", "Empty directory must fail");
    }

    // ==========================================
    // 2. S-READ-02: 计数探针与两层缓存
    // ==========================================
    {
        const auto singleImg = filesDir / "single.png";
        writePng(singleImg, cv::Mat(16, 16, CV_8UC3, cv::Scalar(10, 20, 30)));

        testing::fileReadCount = 0;
        testing::decodeCount   = 0;

        ReadImageTool probeTool(services);
        Values inputs = defaultInputs(singleImg);

        // 第一次读取: reload=false -> read=1, decode=1
        auto res1 = probeTool.execute(inputs);
        requireSuccess(res1, "First read must succeed");
        REQUIRE(testing::fileReadCount == 1, "First read must read from disk once");
        REQUIRE(testing::decodeCount == 1, "First read must decode once");
        const auto &batch1 = std::get<DataRef<Image>>(res1.outputs.at(read_ports::out::IMAGES)).get();
        REQUIRE(batch1.size() == 1, "Batch 1 size must be 1");
        uchar *pixels1 = batch1[0].data;

        // 第二次读取: reload=false -> 命中缓存, read=1, decode=1 (不增加)
        auto res2 = probeTool.execute(inputs);
        requireSuccess(res2, "Second read must succeed");
        REQUIRE(testing::fileReadCount == 1, "Second read with reload=false must NOT read disk");
        REQUIRE(testing::decodeCount == 1, "Second read with reload=false must NOT decode");
        const auto &batch2 = std::get<DataRef<Image>>(res2.outputs.at(read_ports::out::IMAGES)).get();
        REQUIRE(batch2.size() == 1, "Batch 2 size must be 1");
        REQUIRE(batch2[0].data == pixels1, "Cached read must share pixel memory");

        // 第三次读取: reload=true -> 强制重读, read=2, decode=2
        inputs.insert_or_assign(read_ports::in::RELOAD, true);
        auto res3 = probeTool.execute(inputs);
        requireSuccess(res3, "Third read (reload=true) must succeed");
        REQUIRE(testing::fileReadCount == 2, "reload=true must read disk again");
        REQUIRE(testing::decodeCount == 2, "reload=true must decode again");
        const auto &batch3 = std::get<DataRef<Image>>(res3.outputs.at(read_ports::out::IMAGES)).get();
        REQUIRE(batch3[0].data != pixels1, "reload=true must produce new decoded buffer");

        // 旧输出持有的像素仍然有效
        REQUIRE(batch1[0].data == pixels1, "Old output Owner must keep pixels alive");

        // 第四次读取: reload=true -> 继续重读, read=3, decode=3
        auto res4 = probeTool.execute(inputs);
        requireSuccess(res4, "Fourth read (reload=true) must succeed");
        REQUIRE(testing::fileReadCount == 3, "Continuing reload=true must read disk");
        REQUIRE(testing::decodeCount == 3, "Continuing reload=true must decode");
    }

    // ==========================================
    // 3. S-READ-03: 目录遍历、顺序、循环跨尾、逆序与原子失败
    // ==========================================
    {
        const auto dirA = directory / "dir_traversal";
        std::filesystem::create_directories(dirA);
        writePng(dirA / "img_1.png", cv::Mat(8, 8, CV_8UC3, cv::Scalar(1, 1, 1)));
        writePng(dirA / "img_2.png", cv::Mat(8, 8, CV_8UC3, cv::Scalar(2, 2, 2)));
        writePng(dirA / "img_3.png", cv::Mat(8, 8, CV_8UC3, cv::Scalar(3, 3, 3)));

        // 3.1 顺序遍历 + 循环 + count=2
        ReadImageTool dirTool(services);
        Values inputs = defaultInputs(dirA);
        inputs.insert_or_assign(read_ports::in::COUNT, std::int64_t(2));
        inputs.insert_or_assign(read_ports::in::TRAVERSAL, std::string("循环"));
        inputs.insert_or_assign(read_ports::in::ORDER, std::string("顺序（下一张）"));

        // 第 1 轮: 取 2 张 (img_1, img_2)
        auto res = dirTool.execute(inputs);
        requireSuccess(res, "Loop round 1 must succeed");
        auto paths = std::get<Array>(res.outputs.at(read_ports::out::PATHS)).get<std::vector<std::string>>();
        REQUIRE(paths.size() == 2, "Round 1 must take 2 images");
        REQUIRE(paths[0].ends_with("img_1.png") && paths[1].ends_with("img_2.png"), "Round 1: img_1, img_2");

        // 第 2 轮: 跨尾循环取 2 张 (img_3, img_1)
        res = dirTool.execute(inputs);
        requireSuccess(res, "Loop round 2 must succeed");
        paths = std::get<Array>(res.outputs.at(read_ports::out::PATHS)).get<std::vector<std::string>>();
        REQUIRE(paths.size() == 2, "Round 2 must take 2 images");
        REQUIRE(paths[0].ends_with("img_3.png") && paths[1].ends_with("img_1.png"), "Round 2: img_3, img_1 (wrapped)");

        // 3.2 count > total: 单次输出超过总文件数，同一文件出现多次
        inputs.insert_or_assign(read_ports::in::COUNT, std::int64_t(5));
        res = dirTool.execute(inputs);
        requireSuccess(res, "Count > total in loop must succeed");
        paths = std::get<Array>(res.outputs.at(read_ports::out::PATHS)).get<std::vector<std::string>>();
        REQUIRE(paths.size() == 5, "Count 5 must return 5 images");
        const auto &mats = std::get<DataRef<Image>>(res.outputs.at(read_ports::out::IMAGES)).get();
        REQUIRE(mats.size() == 5, "Images size must be 5");

        // 3.3 逆序遍历
        inputs.insert_or_assign(read_ports::in::ORDER, std::string("逆序（上一张）"));
        inputs.insert_or_assign(read_ports::in::COUNT, std::int64_t(2));
        res = dirTool.execute(inputs);
        requireSuccess(res, "Reverse order must succeed");
        paths = std::get<Array>(res.outputs.at(read_ports::out::PATHS)).get<std::vector<std::string>>();
        REQUIRE(paths.size() == 2, "Reverse round 1 must return 2 images");
        REQUIRE(paths[0].ends_with("img_3.png") && paths[1].ends_with("img_2.png"), "Reverse starts from end: img_3, img_2");

        // 3.4 不循环模式: 序列末尾与结束报错
        ReadImageTool onceTool(services);
        Values onceInputs = defaultInputs(dirA);
        onceInputs.insert_or_assign(read_ports::in::TRAVERSAL, std::string("不循环"));
        onceInputs.insert_or_assign(read_ports::in::COUNT, std::int64_t(2));

        // 第 1 轮: img_1, img_2
        res = onceTool.execute(onceInputs);
        requireSuccess(res, "Once round 1 must succeed");
        REQUIRE(!std::get<bool>(res.outputs.at(read_ports::out::END_REACHED)), "End not reached yet");

        // 第 2 轮: 仅剩 1 张 (img_3)
        res = onceTool.execute(onceInputs);
        requireSuccess(res, "Once round 2 must succeed with remaining 1 image");
        paths = std::get<Array>(res.outputs.at(read_ports::out::PATHS)).get<std::vector<std::string>>();
        REQUIRE(paths.size() == 1, "Remaining count must be 1");
        REQUIRE(paths[0].ends_with("img_3.png"), "Last image must be img_3");
        REQUIRE(std::get<bool>(res.outputs.at(read_ports::out::END_REACHED)), "End reached must be true");

        // 第 3 轮: 已结束，必须报错
        res = onceTool.execute(onceInputs);
requireFailure(res, "图像序列已结束", "Exhausted non-loop sequence must fail");

        // 3.5 损坏文件失败原子性: 中途坏图游标不推进，修复后从原位成功
        ReadImageTool atomicTool(services);
        const auto corruptFile = dirA / "img_corrupt.png";
        std::ofstream(corruptFile, std::ios::binary).write("corrupted_data", 14);

        Values atomicInputs = defaultInputs(dirA);
        atomicInputs.insert_or_assign(read_ports::in::COUNT, std::int64_t(1));
        atomicInputs.insert_or_assign(read_ports::in::TRAVERSAL, std::string("不循环"));

        // img_1 成功
        res = atomicTool.execute(atomicInputs);
        requireSuccess(res, "img_1 read must succeed");
        REQUIRE(std::get<std::int64_t>(res.outputs.at(read_ports::out::CURRENT_INDEX)) == 0, "Index must be 0");

        // img_2 成功
        res = atomicTool.execute(atomicInputs);
        requireSuccess(res, "img_2 read must succeed");
        REQUIRE(std::get<std::int64_t>(res.outputs.at(read_ports::out::CURRENT_INDEX)) == 1, "Index must be 1");

        // img_3 成功
        res = atomicTool.execute(atomicInputs);
        requireSuccess(res, "img_3 read must succeed");
        REQUIRE(std::get<std::int64_t>(res.outputs.at(read_ports::out::CURRENT_INDEX)) == 2, "Index must be 2");

        // img_corrupt 失败
        res = atomicTool.execute(atomicInputs);
        requireFailure(res, "图像解码失败", "Corrupt image decode must fail");
        REQUIRE(isMissing(res.outputs.at(read_ports::out::IMAGES)), "Images output must be Missing on failure");

        // 再次执行，游标未推进，依然在 corrupt 位置报错
        res = atomicTool.execute(atomicInputs);
        requireFailure(res, "图像解码失败", "Retry without fix must fail at the same index");

        // 修复损坏文件
        writePng(corruptFile, cv::Mat(8, 8, CV_8UC3, cv::Scalar(4, 4, 4)));
        res = atomicTool.execute(atomicInputs);
        requireSuccess(res, "After fix, read must succeed from original position");
        paths = std::get<Array>(res.outputs.at(read_ports::out::PATHS)).get<std::vector<std::string>>();
        REQUIRE(paths[0].ends_with("img_corrupt.png"), "Resumed from corrupt index");
        std::filesystem::remove(corruptFile);
    }

    // ==========================================
    // 4. S-READ-04: 可写共享与原地修改
    // ==========================================
    {
        const auto writeImg = directory / "writable.png";
        writePng(writeImg, cv::Mat(4, 4, CV_8UC3, cv::Scalar(10, 20, 30)));

        ReadImageTool writeTool(services);
        Values inputs = defaultInputs(writeImg);

        auto res1 = writeTool.execute(inputs);
        requireSuccess(res1, "Writeable test read 1 must succeed");
        auto &mat1 = std::get<DataRef<Image>>(res1.outputs.at(read_ports::out::IMAGES)).get()[0];
        uchar *origPtr = mat1.data;

        // 下游原地修改像素
        mat1.at<cv::Vec3b>(0, 0) = cv::Vec3b(99, 99, 99);

        // 再次读取 (reload=false): 看到下游修改后的内容
        auto res2 = writeTool.execute(inputs);
        requireSuccess(res2, "Writeable test read 2 must succeed");
        auto &mat2 = std::get<DataRef<Image>>(res2.outputs.at(read_ports::out::IMAGES)).get()[0];
        REQUIRE(mat2.data == origPtr, "Zero-copy shared memory");
        REQUIRE(mat2.at<cv::Vec3b>(0, 0) == cv::Vec3b(99, 99, 99), "Downstream in-place modification is visible");

        // 再次读取 (reload=true): 重新从磁盘解码，恢复原始磁盘数据
        inputs.insert_or_assign(read_ports::in::RELOAD, true);
        auto res3 = writeTool.execute(inputs);
        requireSuccess(res3, "Reload=true must succeed");
        auto &mat3 = std::get<DataRef<Image>>(res3.outputs.at(read_ports::out::IMAGES)).get()[0];
        REQUIRE(mat3.data != origPtr, "Reload produces new memory buffer");
        REQUIRE(mat3.at<cv::Vec3b>(0, 0) == cv::Vec3b(10, 20, 30), "Reload restores original disk data");

        // 旧引用仍保留修改数据
        REQUIRE(mat1.at<cv::Vec3b>(0, 0) == cv::Vec3b(99, 99, 99), "Old output reference remains valid");

        // release() 测试
        REQUIRE(bool(writeTool.release()), "release must succeed");
    }

    // ==========================================
    // 5. S-READ-05: 递归、相对路径与中文路径
    // ==========================================
    // 5. S-READ-05: 递归、相对路径与中文路径
    // ==========================================
    {
        const auto cnDir = directory / L"中文目录";
        std::filesystem::create_directories(cnDir);
        const auto subCn = cnDir / L"子目录";
        std::filesystem::create_directories(subCn);
        writePng(cnDir / L"根图.png", cv::Mat(4, 4, CV_8UC3, cv::Scalar(1, 1, 1)));
        writePng(subCn / L"深层图.png", cv::Mat(4, 4, CV_8UC3, cv::Scalar(2, 2, 2)));

        ReadImageTool cnTool(services);

        // 不递归: 只有 1 张 (根图.png)
        Values inputsNoRec = defaultInputs(cnDir);
        inputsNoRec.insert_or_assign(read_ports::in::RECURSIVE, false);
        inputsNoRec.insert_or_assign(read_ports::in::TRAVERSAL, std::string("不循环"));
        inputsNoRec.insert_or_assign(read_ports::in::COUNT, std::int64_t(10));
        auto res = cnTool.execute(inputsNoRec);
        requireSuccess(res, "Non-recursive CN dir read must succeed");
        REQUIRE(std::get<std::int64_t>(res.outputs.at(read_ports::out::TOTAL_COUNT)) == 1, "Non-recursive total_count must be 1");
        auto paths = std::get<Array>(res.outputs.at(read_ports::out::PATHS)).get<std::vector<std::string>>();
        REQUIRE(paths.size() == 1, "Non-recursive must find exactly 1 image");

        // 递归: 包含子目录, 共 2 张
        Values inputsRec = defaultInputs(cnDir);
        inputsRec.insert_or_assign(read_ports::in::RECURSIVE, true);
        inputsRec.insert_or_assign(read_ports::in::TRAVERSAL, std::string("不循环"));
        inputsRec.insert_or_assign(read_ports::in::COUNT, std::int64_t(10));
        res = cnTool.execute(inputsRec);
        requireSuccess(res, "Recursive CN dir read must succeed");
        REQUIRE(std::get<std::int64_t>(res.outputs.at(read_ports::out::TOTAL_COUNT)) == 2, "Recursive total_count must be 2");
        paths = std::get<Array>(res.outputs.at(read_ports::out::PATHS)).get<std::vector<std::string>>();
        REQUIRE(paths.size() == 2, "Recursive must find 2 images");
    }

    // ==========================================
    // 6. T35-12: 历史缓存是严格有界的 LRU
    // ==========================================
    {
        const auto oneImage = [](uchar value) { return cv::Mat(10, 10, CV_8UC1, cv::Scalar(value)); };
        const std::size_t imageBytes = 10 * 10;
        DecodedImageCache cache{imageBytes * 3};

        REQUIRE(cache.insert("a", oneImage(1)), "first image must be retained");
        REQUIRE(cache.insert("b", oneImage(2)), "second image must be retained");
        REQUIRE(cache.insert("c", oneImage(3)), "third image must be retained");
        REQUIRE(cache.bytes() <= cache.budget(), "history exceeded its byte budget");

        // A hit is a use: after touching `a`, the coldest entry is `b`, so `b` is the one evicted.
        const cv::Mat held = *cache.find("a");
        REQUIRE(held.data != nullptr, "a retained image must be readable");
        REQUIRE(cache.insert("d", oneImage(4)), "a fourth image must be retained by evicting the coldest");
        REQUIRE(cache.find("a") != nullptr, "a recently used image must survive eviction");
        REQUIRE(cache.find("b") == nullptr, "the coldest image must be the one evicted");
        REQUIRE(cache.bytes() <= cache.budget(), "history exceeded its byte budget after eviction");

        // An image larger than the whole budget is returned once by the tool but never retained.
        DecodedImageCache tight{imageBytes};
        const cv::Mat     oversized(10, 11, CV_8UC1, cv::Scalar(9));
        REQUIRE(!tight.insert("big", oversized), "an oversized image must be refused by the history");
        REQUIRE(tight.bytes() == 0 && tight.size() == 0, "a refused image must not consume budget");

        // A protected batch (the last successful output) is never evicted, and the budget still holds.
        DecodedImageCache guarded{imageBytes * 2};
        REQUIRE(guarded.insert("keep", oneImage(5)), "protected candidate must be retained");
        guarded.protect({"keep"});
        REQUIRE(guarded.insert("one", oneImage(6)), "a second image fits next to the protected batch");
        REQUIRE(guarded.insert("two", oneImage(7)), "a third image must evict the unprotected one");
        REQUIRE(guarded.find("keep") != nullptr, "the protected batch was evicted");
        REQUIRE(guarded.bytes() <= guarded.budget(), "protected history exceeded its budget");

        // Eviction releases only the cache's own reference.
        REQUIRE(held.data != nullptr && held.at<uchar>(0, 0) == 1,
                "an evicted image kept alive by an external Owner must stay readable");
    }

    // ==========================================
    // 7. T35-13: 失败不提前清空上次成功批次
    // ==========================================
    {
        const auto atomicDir = directory / "dir_atomic";
        std::filesystem::create_directories(atomicDir);
        writePng(atomicDir / "a.png", cv::Mat(8, 8, CV_8UC3, cv::Scalar(11, 11, 11)));
        writePng(atomicDir / "b.png", cv::Mat(8, 8, CV_8UC3, cv::Scalar(22, 22, 22)));
        writePng(atomicDir / "c.png", cv::Mat(8, 8, CV_8UC3, cv::Scalar(33, 33, 33)));

        ReadImageTool atomicTool(services);
        Values inputs = defaultInputs(atomicDir);
        inputs.insert_or_assign(read_ports::in::COUNT, std::int64_t(2));
        inputs.insert_or_assign(read_ports::in::TRAVERSAL, std::string("不循环"));

        auto first = atomicTool.execute(inputs);
        requireSuccess(first, "the first batch must succeed");
        const auto firstBatch = std::get<DataRef<Image>>(first.outputs.at(read_ports::out::IMAGES));
        const auto firstPixels = firstBatch.get()[0].data;
        const auto firstPaths = std::get<Array>(first.outputs.at(read_ports::out::PATHS)).get<std::vector<std::string>>();
        REQUIRE(firstPaths.size() == 2, "the first batch must contain two images");

        // A reload batch whose second file is corrupt: the decoded first file must not reach the
        // cache, the cursor must not advance and the previous batch must stay alive.
        std::filesystem::remove(atomicDir / "c.png");
        std::ofstream(atomicDir / "c.png", std::ios::binary).write("broken_image", 12);
        inputs.insert_or_assign(read_ports::in::RELOAD, true);
        auto failed = atomicTool.execute(inputs);
        requireFailure(failed, "图像解码失败", "a corrupt second image must fail the whole batch");
        REQUIRE(isMissing(failed.outputs.at(read_ports::out::IMAGES)), "a failed batch publishes no images");
        REQUIRE(firstBatch.get()[0].data == firstPixels && firstBatch.get()[0].at<cv::Vec3b>(0, 0) == cv::Vec3b(11, 11, 11),
                "a failed batch invalidated the previous Owner");

        // Retry without reload after the file is repaired: the cursor still points at c.png and the
        // cache still holds the pixels of the last successful attempt, not the failed one.
        writePng(atomicDir / "c.png", cv::Mat(8, 8, CV_8UC3, cv::Scalar(33, 33, 33)));
        inputs.insert_or_assign(read_ports::in::RELOAD, false);
        auto resumed = atomicTool.execute(inputs);
        requireSuccess(resumed, "the repaired sequence must resume");
        auto resumedPaths = std::get<Array>(resumed.outputs.at(read_ports::out::PATHS)).get<std::vector<std::string>>();
        REQUIRE(resumedPaths.size() == 1 && resumedPaths[0].ends_with("c.png"),
                "the resumed batch must start at the file that failed");
        const auto &resumedImages = std::get<DataRef<Image>>(resumed.outputs.at(read_ports::out::IMAGES)).get();
        REQUIRE(resumedImages[0].at<cv::Vec3b>(0, 0) == cv::Vec3b(33, 33, 33), "the resumed batch read the repaired file");
    }

    // ==========================================
    // 8. T35-14: 目录尾部被删除后不得重复旧图
    // ==========================================
    {
        const auto tailDir = directory / "dir_tail";
        const auto rebuildTail = [&]
        {
            std::filesystem::remove_all(tailDir);
            std::filesystem::create_directories(tailDir);
            writePng(tailDir / "a.png", cv::Mat(4, 4, CV_8UC3, cv::Scalar(1, 1, 1)));
            writePng(tailDir / "b.png", cv::Mat(4, 4, CV_8UC3, cv::Scalar(2, 2, 2)));
            writePng(tailDir / "c.png", cv::Mat(4, 4, CV_8UC3, cv::Scalar(3, 3, 3)));
        };
        const auto readOne = [&](ReadImageTool &tool, const Values &inputs)
        {
            auto res = tool.execute(inputs);
            requireSuccess(res, "single-step read must succeed");
            return std::get<Array>(res.outputs.at(read_ports::out::PATHS)).get<std::vector<std::string>>()[0];
        };

        // 8.1 forward, non-looping: the next file is deleted and the last delivered file is the tail.
        {
            rebuildTail();
            ReadImageTool tool(services);
            Values inputs = defaultInputs(tailDir);
            inputs.insert_or_assign(read_ports::in::TRAVERSAL, std::string("不循环"));
            REQUIRE(readOne(tool, inputs).ends_with("a.png"), "step 1 must deliver a.png");
            REQUIRE(readOne(tool, inputs).ends_with("b.png"), "step 2 must deliver b.png");
            std::filesystem::remove(tailDir / "c.png");
            auto res = tool.execute(inputs);
            requireFailure(res, "图像序列已结束", "a deleted tail must end a non-looping sequence instead of restarting");
        }

        // 8.2 forward, non-looping: both the next and the last delivered file disappear.
        {
            rebuildTail();
            ReadImageTool tool(services);
            Values inputs = defaultInputs(tailDir);
            inputs.insert_or_assign(read_ports::in::TRAVERSAL, std::string("不循环"));
            REQUIRE(readOne(tool, inputs).ends_with("a.png"), "step 1 must deliver a.png");
            REQUIRE(readOne(tool, inputs).ends_with("b.png"), "step 2 must deliver b.png");
            std::filesystem::remove(tailDir / "b.png");
            std::filesystem::remove(tailDir / "c.png");
            auto res = tool.execute(inputs);
            requireFailure(res, "图像序列已结束", "losing both anchors must end the sequence, not restart it");
        }

        // 8.3 forward, non-looping: the whole tail after the delivered file disappears.
        {
            rebuildTail();
            writePng(tailDir / "d.png", cv::Mat(4, 4, CV_8UC3, cv::Scalar(4, 4, 4)));
            ReadImageTool tool(services);
            Values inputs = defaultInputs(tailDir);
            inputs.insert_or_assign(read_ports::in::TRAVERSAL, std::string("不循环"));
            REQUIRE(readOne(tool, inputs).ends_with("a.png"), "step 1 must deliver a.png");
            std::filesystem::remove(tailDir / "b.png");
            std::filesystem::remove(tailDir / "c.png");
            std::filesystem::remove(tailDir / "d.png");
            auto res = tool.execute(inputs);
            requireFailure(res, "图像序列已结束", "deleting the whole tail must end the sequence");
        }

        // 8.4 reverse, non-looping: deleting the previous file at the head ends the sequence.
        {
            rebuildTail();
            ReadImageTool tool(services);
            Values inputs = defaultInputs(tailDir);
            inputs.insert_or_assign(read_ports::in::TRAVERSAL, std::string("不循环"));
            inputs.insert_or_assign(read_ports::in::ORDER, std::string("逆序（上一张）"));
            REQUIRE(readOne(tool, inputs).ends_with("c.png"), "reverse step 1 must deliver c.png");
            REQUIRE(readOne(tool, inputs).ends_with("b.png"), "reverse step 2 must deliver b.png");
            std::filesystem::remove(tailDir / "a.png");
            auto res = tool.execute(inputs);
            requireFailure(res, "图像序列已结束", "a deleted head must end the reverse sequence");
        }

        // 8.5 looping mode still wraps, but by an explicit request rather than by falling back.
        {
            rebuildTail();
            ReadImageTool tool(services);
            Values inputs = defaultInputs(tailDir);
            inputs.insert_or_assign(read_ports::in::TRAVERSAL, std::string("循环"));
            REQUIRE(readOne(tool, inputs).ends_with("a.png"), "loop step 1 must deliver a.png");
            REQUIRE(readOne(tool, inputs).ends_with("b.png"), "loop step 2 must deliver b.png");
            std::filesystem::remove(tailDir / "c.png");
            REQUIRE(readOne(tool, inputs).ends_with("a.png"), "a looping sequence must wrap to the first file");
        }
    }

    // ==========================================
    // 9. S-CORE-02: the whole result is built before the cursor/catalog move; cache maintenance is
    //    best effort and never turns a valid batch into a failure.
    // ==========================================
    {
        const auto atomicDir = directory / "atomic";
        std::filesystem::create_directories(atomicDir);
        writePng(atomicDir / "a.png", cv::Mat(4, 4, CV_8UC1, cv::Scalar(1)));
        writePng(atomicDir / "b.png", cv::Mat(4, 4, CV_8UC1, cv::Scalar(2)));
        writePng(atomicDir / "c.png", cv::Mat(4, 4, CV_8UC1, cv::Scalar(3)));

        const auto nextPath = [&](ReadImageTool &tool, const Values &inputs)
        {
            auto res = tool.execute(inputs);
            requireSuccess(res, "atomic read must succeed");
            return std::get<Array>(res.outputs.at(read_ports::out::PATHS)).get<std::vector<std::string>>()[0];
        };

        ReadImageTool tool(services);
        Values        inputs = defaultInputs(atomicDir);
        inputs.insert_or_assign(read_ports::in::TRAVERSAL, std::string("不循环"));
        REQUIRE(nextPath(tool, inputs).ends_with("a.png"), "atomic step 1 must deliver a.png");

        bool fired = false;
        testing::beforeResultBuild = [&]
        {
            if (fired)
                return;
            fired = true;
            throw std::bad_alloc();
        };
        auto failed = tool.execute(inputs);
        testing::beforeResultBuild = {};
        requireFailure(failed, "图像读取失败", "a result packaging failure must be reported");
        REQUIRE(nextPath(tool, inputs).ends_with("b.png"),
                "a failed result packaging must not skip b.png");

        bool cacheFailed = false;
        testing::beforeCacheInsert = [&]
        {
            if (cacheFailed)
                return;
            cacheFailed = true;
            throw std::bad_alloc();
        };
        REQUIRE(nextPath(tool, inputs).ends_with("c.png"),
                "a cache insert failure must not fail an already valid batch");
        testing::beforeCacheInsert = {};
    }

    std::filesystem::remove_all(directory);
    std::cout << "[PASS] testReadImageTool passed successfully!" << std::endl;
}
