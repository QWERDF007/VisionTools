#include <cuda_runtime.h>
#include "../../tests/result_view.h"
#include <visionrt/contracts/Module.h>
#include <visionrt/contracts/ROI.h>
#include <visionrt/contracts/Tool.h>
#include <visionrt/contracts/Value.h>


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <span>
#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <winver.h>
#endif
#include <stdexcept>
#include <string>
#include <vector>

using namespace visionrt;

namespace {

#ifdef _WIN32
using LibHandle = HMODULE;

inline LibHandle loadSharedLib(const std::filesystem::path &path)
{
    return LoadLibraryW(path.wstring().c_str());
}

inline void *getProcAddr(LibHandle h, const char *name)
{
    return reinterpret_cast<void *>(GetProcAddress(h, name));
}

inline void freeSharedLib(LibHandle h)
{
    FreeLibrary(h);
}
#else
#    include <dlfcn.h>
using LibHandle = void *;

inline LibHandle loadSharedLib(const std::filesystem::path &path)
{
    return dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
}

inline void *getProcAddr(LibHandle h, const char *name)
{
    return dlsym(h, name);
}

inline void freeSharedLib(LibHandle h)
{
    dlclose(h);
}
#endif


using ModuleApiFn = const ModuleApi *(VT_CALL *)() noexcept;

class ProbeModule final : public IToolRegistry
{
public:
    explicit ProbeModule(const std::filesystem::path &path)
    {
        library_ = loadSharedLib(std::filesystem::absolute(path));
        if (!library_)
            throw std::runtime_error("无法加载推理模块：" + path.string());
        try
        {
            const auto entry = reinterpret_cast<ModuleApiFn>(getProcAddr(library_, "visiontoolsModuleApi"));
            if (!entry)
                throw std::runtime_error("推理模块缺少 visiontoolsModuleApi");
            const auto *api = entry();
            if (!api || api->abi_major != MODULE_ABI_MAJOR || api->struct_bytes < sizeof(ModuleApi) || !api->register_tools)
                throw std::runtime_error("推理模块 ABI3 接口无效");
            const auto status = api->register_tools(*this);
            if (!status)
                throw std::runtime_error(status.message + "：" + status.detail);
        }
        catch (...)
        {
            freeSharedLib(library_);
            throw;
        }
    }
    ~ProbeModule() override { freeSharedLib(library_); }
    ProbeModule(const ProbeModule &) = delete;
    ProbeModule &operator=(const ProbeModule &) = delete;

    ToolStatus add(const ToolFactoryDescriptor &descriptor) noexcept override
    {
        try
        {
            if (!descriptor.tool_id || !*descriptor.tool_id || !descriptor.create || !descriptor.destroy)
                return {-3, "无效推理工厂"};
            auto [position, inserted] = factories.emplace(descriptor.tool_id, descriptor);
            if (!inserted)
                return {-3, "推理工厂 ID 重复"};
            position->second.tool_id = position->first.c_str();
            return {};
        }
        catch (const std::exception &error)
        {
            return {-3, "推理工厂注册失败", error.what()};
        }
    }
    std::map<std::string, ToolFactoryDescriptor, std::less<>> factories;
private:
    LibHandle library_ = nullptr;
};

const ToolFactoryDescriptor &getFactory(const std::filesystem::path &path = {})
{
    static const ProbeModule module(path);
    return module.factories.at("visionrt.dl.yolo");
}

struct DetectionParameters
{
    double       confidence = 0.25;
    double       nms        = 0.45;
    std::int64_t maximum    = 300;
};

void validateParameters(const DetectionParameters &p)
{
    if (p.confidence < 0.0 || p.confidence > 1.0)
        throw std::runtime_error("conf_threshold out of range [0, 1]");
    if (p.nms < 0.0 || p.nms > 1.0)
        throw std::runtime_error("nms_threshold out of range [0, 1]");
    if (p.maximum <= 0)
        throw std::runtime_error("max_detections must be positive");
}

struct Arguments
{
    std::string              weights, engine, report, precision = "fp32", modulePath, packageDir;
    std::vector<std::string> images;
    int                      device = 0, width = 640, height = 640, classes = 80;
    DetectionParameters      detection;
    bool                     syntheticOnly = false;
};

Arguments arguments(int argc, char **argv)
{
    Arguments result;
    for (int i = 1; i < argc; ++i)
    {
        const std::string key = argv[i];
        if (key == "--synthetic-test")
        {
            result.syntheticOnly = true;
            continue;
        }
        if (i + 1 == argc)
            throw std::runtime_error("Missing argument after " + key);
        const std::string value = argv[++i];
        if (key == "--wts")
            result.weights = value;
        else if (key == "--engine")
            result.engine = value;
        else if (key == "--report")
            result.report = value;
        else if (key == "--module")
            result.modulePath = value;
        else if (key == "--package-dir")
            result.packageDir = value;
        else if (key == "--image")
            result.images.push_back(value);
        else if (key == "--precision")
            result.precision = value;
        else if (key == "--device")
            result.device = std::stoi(value);
        else if (key == "--width")
            result.width = std::stoi(value);
        else if (key == "--height")
            result.height = std::stoi(value);
        else if (key == "--classes")
            result.classes = std::stoi(value);
        else if (key == "--conf")
            result.detection.confidence = std::stod(value);
        else if (key == "--nms")
            result.detection.nms = std::stod(value);
        else if (key == "--max")
            result.detection.maximum = std::stoll(value);
        else
            throw std::runtime_error("Unknown argument: " + key);
    }
    if (result.syntheticOnly)
    {
        if (result.report.empty())
            result.report = "probe_report.json";
        return result;
    }
    if (result.weights.empty() || result.engine.empty() || result.report.empty() || result.images.size() < 2)
    {
        throw std::runtime_error(
            "Missing required arguments for non-synthetic run (requires --wts, --engine, --report, and at least 2 "
            "--image arguments)");
    }
    validateParameters(result.detection);
    return result;
}

void require(bool value, const std::string &message)
{
    if (!value)
        throw std::runtime_error(message);
}

void success(const ToolStatus &status)
{
    require(bool(status), status.message);
}

void allMissing(const ToolResult &result)
{
    require(!testview::status(result) && testview::businessCount(result) == 5,
            "Expected whole-invocation failure with five business outputs");
    for (const auto &[name, value] : result.outputs)
        require(isMissing(value), "Partial output escaped after failure: " + name);
}

std::string utf8(const std::filesystem::path &path)
{
    const auto encoded = path.u8string();
    return {reinterpret_cast<const char *>(encoded.data()), encoded.size()};
}

std::string absolute(const std::string &path)
{
    return utf8(std::filesystem::absolute(std::filesystem::path(path)));
}

struct ProbeEnvironment
{
    std::uint32_t sdkAbiMajor      = MODULE_ABI_MAJOR;
    std::string cudaRuntimeVersion = "unavailable";
    std::string cudaDriverVersion  = "unavailable";
    std::string registeredModels   = "visionrt.dl.yolo";

    ProbeEnvironment()
    {
        int rt = 0;
        if (cudaRuntimeGetVersion(&rt) == cudaSuccess && rt > 0)
        {
            cudaRuntimeVersion = std::to_string(rt / 1000) + "." + std::to_string((rt % 100) / 10);
        }
        int drv = 0;
        if (cudaDriverGetVersion(&drv) == cudaSuccess && drv > 0)
        {
            cudaDriverVersion = std::to_string(drv / 1000) + "." + std::to_string((drv % 100) / 10);
        }
    }
};

struct GpuDevice
{
    int         index = 0;
    std::string name;
};

struct GpuQueryResult
{
    ToolStatus             status;
    std::vector<GpuDevice> devices;
};

GpuQueryResult queryDevices()
{
    GpuQueryResult result;
    int            count = 0;
    cudaError_t    err   = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0)
    {
        result.status = {-1, "CUDA device query unavailable"};
        result.devices.push_back({0, "unavailable"});
        return result;
    }
    result.status = {0, ""};
    for (int i = 0; i < count; ++i)
    {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
        {
            result.devices.push_back({i, prop.name});
        }
        else
        {
            result.devices.push_back({i, "unavailable"});
        }
    }
    return result;
}

using Tool = std::unique_ptr<IToolInstance, void (*)(IToolInstance *)>;

Tool create(const std::filesystem::path &modulePath = "")
{
    const ToolServices services{PathResolver(std::filesystem::current_path())};
    ToolStatus         status;
    const auto        &factory = getFactory(modulePath);
    auto              *raw = factory.create(&services, &status);
    Tool instance(raw,
                [](IToolInstance *p)
                {
                    if (p)
                    {
                        try { p->release(); } catch (...) {}
                        getFactory().destroy(p);
                    }
                });
    success(status);
    require(raw != nullptr, "Module factory returned null for visionrt.dl.yolo");
    return instance;
}

Values configuration(const Arguments &args, bool build)
{
    return {
        {        "model_name",                                    std::string("yolov8n")},
        {         "load_mode",               std::string(build ? "build_wts" : "engine")},
        {        "model_path", std::string(absolute(build ? args.weights : args.engine))},
        {"engine_output_path",                        std::string(absolute(args.engine))},
        {         "precision",                               std::string(args.precision)},
        {       "labels_path",                                          std::string("")},
        {         "gpu_index",                          std::int64_t(args.device)},
        {       "input_width",                           std::int64_t(args.width)},
        {      "input_height",                          std::int64_t(args.height)},
        {       "num_classes",                         std::int64_t(args.classes)}
    };
}

Values inputs(const DataRef<Image> &images, const DetectionParameters &parameters)
{
    return {
        {        "images",                images},
        {"conf_threshold", parameters.confidence},
        { "nms_threshold",        parameters.nms},
        {"max_detections",    parameters.maximum}
    };
}

DataRef<Image> loadImages(const Arguments &args)
{
    Image images;
    for (const auto &name : args.images)
    {
        cv::Mat image = cv::imread(name);
        require(!image.empty(), "Cannot load test image: " + name);
        images.push_back(std::move(image));
    }
    images.push_back(cv::Mat(197, 321, CV_8UC3, cv::Scalar(0, 0, 0)));
    return DataRef<Image>(std::move(images));
}

struct PublicDetection
{
    std::array<float, 4> box{};
    float                score   = 0.0f;
    std::int64_t         classId = 0;
    std::string          className;
};

std::vector<std::vector<PublicDetection>> outputContract(const ToolResult &result, const DataRef<Image> &images)
{
    success(testview::status(result));
    require(testview::businessCount(result) == 5, "Public detection output contract must expose 5 business outputs");
    const auto *rois     = std::get_if<DataRef<ROI>>(&result.outputs.at("rois"));
    const auto *scores   = std::get_if<Array>(&result.outputs.at("scores"));
    const auto *classes  = std::get_if<Array>(&result.outputs.at("classes"));
    const auto *names    = std::get_if<Array>(&result.outputs.at("class_names"));
    const auto *returned = std::get_if<DataRef<Image>>(&result.outputs.at("images"));
    require(rois && scores && classes && names && returned, "Invalid public output types");

    const auto &roiBatch   = rois->get();
    const auto &scoreData  = scores->get<std::vector<std::vector<float>>>();
    const auto &classData  = classes->get<std::vector<std::vector<std::int64_t>>>();
    const auto &nameData   = names->get<std::vector<std::vector<std::string>>>();
    const auto  count      = images.get().size();
    require(roiBatch.size() == count && scoreData.size() == count && classData.size() == count
                && nameData.size() == count && returned->get().size() == count,
            "Mismatched batch size in outputs");

    std::vector<std::vector<PublicDetection>> batch(count);
    for (std::size_t b = 0; b < count; ++b)
    {
        const auto &boxes      = roiBatch[b];
        const auto  n          = boxes.size();
        const auto &scoreRow   = scoreData[b];
        const auto &classRow   = classData[b];
        const auto &nameRow    = nameData[b];
        require(scoreRow.size() == n && classRow.size() == n && nameRow.size() == n,
                "Ragged element mismatch across outputs in batch item");
        for (std::size_t i = 0; i < n; ++i)
        {
            PublicDetection d;
            const auto      corners = rectangleCorners(std::span<const double>(boxes[i]));
            d.box                   = {static_cast<float>(corners[0].x), static_cast<float>(corners[0].y),
                                       static_cast<float>(corners[2].x), static_cast<float>(corners[2].y)};
            d.score                 = scoreRow[i];
            d.classId               = classRow[i];
            d.className             = nameRow[i];
            batch[b].push_back(d);
        }
    }
    return batch;
}

inline float intersectionOverUnion(const std::array<float, 4> &a, const std::array<float, 4> &b)
{
    float x1        = std::max(a[0], b[0]);
    float y1        = std::max(a[1], b[1]);
    float x2        = std::min(a[2], b[2]);
    float y2        = std::min(a[3], b[3]);
    float interArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float aArea     = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    float bArea     = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    float unionArea = aArea + bArea - interArea;
    if (unionArea <= 0.0f)
        return 0.0f;
    return interArea / unionArea;
}

inline float edgeError(const std::array<float, 4> &a, const std::array<float, 4> &b)
{
    return std::max({std::abs(a[0] - b[0]), std::abs(a[1] - b[1]), std::abs(a[2] - b[2]), std::abs(a[3] - b[3])});
}

struct ComparisonResult
{
    bool passed         = true;
    int  unmatchedLeft  = 0;
    int  unmatchedRight = 0;
};

ComparisonResult compareDetections(const std::vector<PublicDetection> &a, const std::vector<PublicDetection> &b,
                                   bool exact)
{
    ComparisonResult                      result;
    std::vector<std::vector<std::size_t>> edges(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const auto &left = a[i];
        for (std::size_t j = 0; j < b.size(); ++j)
        {
            const auto &right = b[j];
            if (left.classId != right.classId)
                continue;
            const bool small = std::min(left.box[2] - left.box[0], left.box[3] - left.box[1]) < 16.0f;
            const auto iou   = intersectionOverUnion(left.box, right.box);
            const auto score = std::abs(left.score - right.score);
            const auto edge  = edgeError(left.box, right.box);
            if ((exact && score == 0.0f && edge == 0.0f)
                || (!exact && score <= 0.01f && (small ? edge <= 0.5f : iou >= 0.98f)))
                edges[i].push_back(j);
        }
        std::sort(edges[i].begin(), edges[i].end(), [&](auto x, auto y)
                  { return intersectionOverUnion(left.box, b[x].box) > intersectionOverUnion(left.box, b[y].box); });
    }
    std::vector<int>                                      owners(b.size(), -1);
    std::function<bool(std::size_t, std::vector<bool> &)> augment = [&](std::size_t i, std::vector<bool> &visited)
    {
        for (const auto j : edges[i])
        {
            if (visited[j])
                continue;
            visited[j] = true;
            if (owners[j] < 0 || augment(static_cast<std::size_t>(owners[j]), visited))
            {
                owners[j] = static_cast<int>(i);
                return true;
            }
        }
        return false;
    };
    int matches = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        std::vector<bool> visited(b.size(), false);
        if (augment(i, visited))
            ++matches;
    }
    result.unmatchedLeft  = static_cast<int>(a.size()) - matches;
    result.unmatchedRight = static_cast<int>(b.size()) - matches;
    result.passed         = (result.unmatchedLeft == 0) && (result.unmatchedRight == 0);
    return result;
}

void writeJsonReport(const std::filesystem::path &reportPath, const std::string &overallStatus,
                     const ProbeEnvironment &env, const GpuQueryResult &devices, const std::string &buildStatus,
                     const std::string &buildReason, const std::string &baselineStatus,
                     const std::string &baselineReason, const std::string &memoryStatus,
                     const std::string &memoryReason, const Arguments *args = nullptr,
                     const std::vector<std::string>                  *imagePaths = nullptr,
                     const std::vector<std::vector<PublicDetection>> *buildDets  = nullptr)
{

    std::filesystem::create_directories(std::filesystem::absolute(reportPath).parent_path());
    cv::FileStorage out(reportPath.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!out.isOpened())
        throw std::runtime_error("Cannot write report file: " + reportPath.string());
    out << "scope" << "gpu_release" << "overall_status" << overallStatus;
    out << "environment" << "{" << "sdk_abi_major" << static_cast<int>(env.sdkAbiMajor)
        << "cuda_runtime_version" << env.cudaRuntimeVersion << "cuda_driver_version" << env.cudaDriverVersion
        << "devices" << "[";
    for (const auto &device : devices.devices)
        out << "{" << "index" << device.index << "name" << device.name << "}";
    out << "]" << "}" << "target_tool_id" << "visionrt.dl.yolo";
    if (args)
        out << "parameters" << "{" << "width" << args->width << "height" << args->height
            << "classes" << args->classes << "precision" << args->precision
            << "confidence" << args->detection.confidence << "nms" << args->detection.nms
            << "max_detections" << std::to_string(args->detection.maximum) << "}";
    out << "images" << "[";
    if (imagePaths)
        for (std::size_t image = 0; image < imagePaths->size(); ++image)
        {
            out << "{" << "path" << (*imagePaths)[image] << "detections" << "[";
            if (buildDets && image < buildDets->size())
                for (const auto &detection : (*buildDets)[image])
                    out << "{" << "class_id" << std::to_string(detection.classId) << "score" << detection.score
                        << "box" << "[:" << detection.box[0] << detection.box[1] << detection.box[2] << detection.box[3]
                        << "]" << "}";
            out << "]" << "}";
        }
    out << "]" << "checks" << "{";
    out << "model_build_and_reload" << "{" << "status" << buildStatus << "reason" << buildReason << "}";
    out << "approved_baseline_comparison" << "{" << "status" << baselineStatus << "reason" << baselineReason << "}";
    out << "memory_lifecycle" << "{" << "status" << memoryStatus << "reason" << memoryReason << "}" << "}";
}


void testSyntheticRoundtrip(const std::filesystem::path &tempReportPath)
{
    ProbeEnvironment env;
    GpuQueryResult   devices = queryDevices();
    writeJsonReport(tempReportPath, "not_run", env, devices, "not_run", "未提供真实权重及运行环境", "not_run",
                    "未提供批准基准与实测输出", "not_run", "未运行");

    // Read back and verify structure (YS-04)

    cv::FileStorage document(tempReportPath.string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    require(document.isOpened(), "Report JSON must reopen");
    require(static_cast<std::string>(document["scope"]) == "gpu_release", "Invalid scope");
    require(static_cast<std::string>(document["overall_status"]) == "not_run", "Invalid overall_status");
    require(static_cast<int>(document["environment"]["sdk_abi_major"]) == static_cast<int>(MODULE_ABI_MAJOR), "Invalid environment");
    require(static_cast<std::string>(document["target_tool_id"]) == "visionrt.dl.yolo", "Invalid target_tool_id");
    require(!document["checks"]["model_build_and_reload"].empty(), "Missing checks");

    // Synthetic comparison test
    PublicDetection d1{
        {10.0f, 10.0f, 50.0f, 50.0f},
        0.9f,
        0,
        "person"
    };
    PublicDetection d2{
        {10.0f, 10.0f, 50.0f, 50.0f},
        0.9f,
        0,
        "person"
    };
    auto exactRes = compareDetections({d1}, {d2}, true);
    require(exactRes.passed, "Exact match synthetic test failed");

    PublicDetection d3{
        {100.0f, 100.0f, 150.0f, 150.0f},
        0.8f,
        1,
        "car"
    };
    auto unmatchRes = compareDetections({d1}, {d3}, true);
    require(!unmatchRes.passed && unmatchRes.unmatchedLeft == 1, "Unmatched synthetic test failed");
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        auto             args = arguments(argc, argv);
        ProbeEnvironment environment;
        const auto       devices = queryDevices();

        if (args.syntheticOnly)
        {
            std::cout << "[INFO] Running synthetic JSON report test..." << std::endl;
            testSyntheticRoundtrip(args.report);
            std::cout << "[SUCCESS] Synthetic probe report written and verified at: " << args.report << std::endl;
            return 0;
        }

        if (args.modulePath.empty() && !args.packageDir.empty())
        {

            const auto package = std::filesystem::weakly_canonical(args.packageDir);
            const auto indexPath = package / "module_index.json";
            cv::FileStorage index(indexPath.string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
            if (!index.isOpened() || static_cast<int>(index["schema_version"]) != 1 || !index["modules"].isSeq())
                throw std::runtime_error("Invalid module_index.json: " + indexPath.string());
            std::filesystem::path relative;
            for (const auto &module : index["modules"])
                if (static_cast<std::string>(module["id"]) == "visiontools_infer")
                {
                    if (!relative.empty())
                        throw std::runtime_error("Duplicate inference module entry");
                    relative = static_cast<std::string>(module["library"]);
                }
            if (relative.empty() || relative.is_absolute())
                throw std::runtime_error("Inference module must declare a relative library path");
            const auto library = std::filesystem::weakly_canonical(package / relative);
            const auto contained = library.lexically_relative(package);
            if (contained.empty() || *contained.begin() == "..")
                throw std::runtime_error("Inference module escapes package root");
            args.modulePath = library.string();
        }

        std::filesystem::create_directories(
            std::filesystem::absolute(std::filesystem::path(args.engine)).parent_path());
        std::filesystem::create_directories(
            std::filesystem::absolute(std::filesystem::path(args.report)).parent_path());
        const auto images = loadImages(args);

        auto built = create(args.modulePath);
        success(built->initialize(configuration(args, true)));
        success(devices.status);

        const auto batchInputs     = inputs(images, args.detection);
        const auto buildResult     = built->execute(batchInputs);
        const auto buildDetections = outputContract(buildResult, images);

        const auto emptyInputs = inputs(DataRef<Image>(Image{}), args.detection);
        const auto emptyResult = built->execute(emptyInputs);
        outputContract(emptyResult, std::get<DataRef<Image>>(emptyInputs.at("images")));

        auto bad      = batchInputs;
        bad["images"] = Missing{};
        allMissing(built->execute(bad));
        bad                  = batchInputs;
        bad["nms_threshold"] = std::numeric_limits<double>::infinity();
        allMissing(built->execute(bad));
        bad           = batchInputs;
        bad["images"] = DataRef<Image>(Image{images.get()[0], cv::Mat(4, 4, CV_32FC1, cv::Scalar(0))});
        allMissing(built->execute(bad));

        bool thresholdEqualKeeps = false, thresholdAboveExcludes = false;
        for (std::size_t b = 0; b < buildDetections.size(); ++b)
        {
            if (buildDetections[b].empty())
                continue;
            const auto &target = buildDetections[b].front();
            if (target.score >= 1.0f)
                continue;
            const double boundaryScore = target.score;
            const auto   single        = DataRef<Image>(Image{images.get()[b]});
            auto         boundary      = inputs(single, args.detection);
            boundary["conf_threshold"] = boundaryScore;
            const auto equalResult     = built->execute(boundary);
            const auto equalDets       = outputContract(equalResult, single);
            thresholdEqualKeeps        = !equalDets.empty() && !equalDets[0].empty();

            boundary["conf_threshold"] = std::nextafter(boundaryScore, 1.0);
            const auto aboveResult     = built->execute(boundary);
            const auto aboveDets       = outputContract(aboveResult, single);
            thresholdAboveExcludes
                = aboveDets.empty() || aboveDets[0].empty()
               || std::none_of(aboveDets[0].begin(), aboveDets[0].end(), [&](const auto &d)
                               { return d.classId == target.classId && std::abs(d.score - target.score) < 1e-4; });
            break;
        }

        auto stricter              = batchInputs;
        stricter["conf_threshold"] = 1.0;
        success(testview::status(built->execute(stricter)));

        // Snapshot persistence across release and destroy
        success(built->release());
        built.reset();
        outputContract(buildResult, images);

        auto loaded = create(args.modulePath);
        success(loaded->initialize(configuration(args, false)));
        const auto loadResult     = loaded->execute(batchInputs);
        const auto loadDetections = outputContract(loadResult, images);

        bool passed = thresholdEqualKeeps && thresholdAboveExcludes;
        for (std::size_t b = 0; b < images.get().size(); ++b)
        {
            const auto single           = DataRef<Image>(Image{images.get()[b]});
            const auto singleInputs     = inputs(single, args.detection);
            const auto singleResult     = loaded->execute(singleInputs);
            const auto singleDetections = outputContract(singleResult, single);

            const auto singleDets = singleDetections.empty() ? std::vector<PublicDetection>{} : singleDetections[0];
            auto       cmp1       = compareDetections(loadDetections[b], singleDets, true);
            auto       cmp2       = compareDetections(buildDetections[b], loadDetections[b], false);
            passed                = cmp1.passed && cmp2.passed && passed;
        }

        const std::string buildStatus   = passed ? "passed" : "failed";
        const std::string overallStatus = passed ? "not_run" : "failed";

        std::vector<std::string> allImagePaths = args.images;
        allImagePaths.push_back("generated:black:321x197");

        writeJsonReport(args.report, overallStatus, environment, devices, buildStatus, "", "not_run",
                        "未提供批准基准对比结果", "not_run", "需独立内存采样", &args, &allImagePaths, &buildDetections);

        if (!passed)
            return 2;
        return 3; // not_run exit code for incomplete scope
    }
    catch (const std::exception &error)
    {
        std::cerr << "Probe error: " << error.what() << std::endl;
        return 1;
    }
}
