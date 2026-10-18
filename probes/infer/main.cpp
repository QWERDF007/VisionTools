#include <visionrt/sdk/Module.h>
#include <visionrt/sdk/Tool.h>
#include <visionrt/sdk/Contract.h>
#include <visionrt/sdk/Value.h>
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>

#ifndef VISIONRT_SDK_VERSION
#define VISIONRT_SDK_VERSION "0.1.0"
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winver.h>
#endif
#include <stdexcept>
#include <string>
#include <vector>

using namespace visionrt;

namespace {

#ifdef _WIN32
using LibHandle = HMODULE;
inline LibHandle loadSharedLib(const std::filesystem::path& path) { return LoadLibraryW(path.wstring().c_str()); }
inline void* getProcAddr(LibHandle h, const char* name) { return reinterpret_cast<void*>(GetProcAddress(h, name)); }
inline void freeSharedLib(LibHandle h) { FreeLibrary(h); }
#else
#include <dlfcn.h>
using LibHandle = void*;
inline LibHandle loadSharedLib(const std::filesystem::path& path) { return dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL); }
inline void* getProcAddr(LibHandle h, const char* name) { return dlsym(h, name); }
inline void freeSharedLib(LibHandle h) { dlclose(h); }
#endif

using ModuleApiFn = const visionrt::ModuleApi*(*)();

inline const visionrt::ModuleApi* getModuleApi(const std::filesystem::path& modulePath = "") {
    static LibHandle s_lib = nullptr;
    static const visionrt::ModuleApi* s_api = nullptr;
    if (s_api) return s_api;

    if (modulePath.empty()) {
        throw std::runtime_error("Module path must be explicitly provided via --module or --package-dir");
    }

    if (!std::filesystem::exists(modulePath)) {
        throw std::runtime_error("Module file does not exist: " + modulePath.string());
    }

    s_lib = loadSharedLib(std::filesystem::absolute(modulePath));
    if (!s_lib) {
        throw std::runtime_error("Failed to load module library: " + modulePath.string());
    }

    auto* fn = reinterpret_cast<ModuleApiFn>(getProcAddr(s_lib, "visiontoolsModuleApi"));
    if (!fn) {
        throw std::runtime_error("visiontoolsModuleApi symbol not found in " + modulePath.string());
    }
    s_api = fn();
    if (!s_api) {
        throw std::runtime_error("visiontoolsModuleApi() returned null");
    }
    if (s_api->abi_major != visionrt::MODULE_ABI_MAJOR) {
        throw std::runtime_error("Module ABI major mismatch");
    }
    if (s_api->struct_bytes != sizeof(visionrt::ModuleApi)) {
        throw std::runtime_error("Module struct bytes mismatch");
    }
    if (!s_api->create) {
        throw std::runtime_error("Module create function pointer is null");
    }
    return s_api;
}

struct DetectionParameters {
    double confidence = 0.25;
    double nms = 0.45;
    std::int64_t maximum = 300;
};

void validateParameters(const DetectionParameters& p) {
    if (p.confidence < 0.0 || p.confidence > 1.0) throw std::runtime_error("conf_threshold out of range [0, 1]");
    if (p.nms < 0.0 || p.nms > 1.0) throw std::runtime_error("nms_threshold out of range [0, 1]");
    if (p.maximum <= 0) throw std::runtime_error("max_detections must be positive");
}

struct Arguments {
    std::string weights, engine, report, precision = "fp32", modulePath, packageDir;
    std::vector<std::string> images;
    int device = 0, width = 640, height = 640, classes = 80;
    DetectionParameters detection;
    bool syntheticOnly = false;
};

Arguments arguments(int argc, char** argv) {
    Arguments result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--synthetic-test") {
            result.syntheticOnly = true;
            continue;
        }
        if (i + 1 == argc) throw std::runtime_error("Missing argument after " + key);
        const std::string value = argv[++i];
        if (key == "--wts") result.weights = value;
        else if (key == "--engine") result.engine = value;
        else if (key == "--report") result.report = value;
        else if (key == "--module") result.modulePath = value;
        else if (key == "--package-dir") result.packageDir = value;
        else if (key == "--image") result.images.push_back(value);
        else if (key == "--precision") result.precision = value;
        else if (key == "--device") result.device = std::stoi(value);
        else if (key == "--width") result.width = std::stoi(value);
        else if (key == "--height") result.height = std::stoi(value);
        else if (key == "--classes") result.classes = std::stoi(value);
        else if (key == "--conf") result.detection.confidence = std::stod(value);
        else if (key == "--nms") result.detection.nms = std::stod(value);
        else if (key == "--max") result.detection.maximum = std::stoll(value);
        else throw std::runtime_error("Unknown argument: " + key);
    }
    if (result.syntheticOnly) {
        if (result.report.empty()) result.report = "probe_report.yaml";
        return result;
    }
    if (result.weights.empty() || result.engine.empty() || result.report.empty() || result.images.size() < 2) {
        throw std::runtime_error("Missing required arguments for non-synthetic run (requires --wts, --engine, --report, and at least 2 --image arguments)");
    }
    validateParameters(result.detection);
    return result;
}

void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

void success(const ToolStatus& status) { require(bool(status), status.message); }

void allMissing(const ToolResult& result) {
    require(!result.status && result.outputs.size() == 5, "Expected whole-invocation failure with five outputs");
    for (const auto& [name, value] : result.outputs) require(isMissing(value), "Partial output escaped after failure: " + name);
}

std::string utf8(const std::filesystem::path& path) {
    const auto encoded = path.u8string();
    return {reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}

std::string absolute(const std::string& path) {
    return utf8(std::filesystem::absolute(std::filesystem::path(path)));
}

struct ProbeEnvironment {
    std::string sdkVersion = VISIONRT_SDK_VERSION;
    std::string cudaRuntimeVersion = "unavailable";
    std::string cudaDriverVersion = "unavailable";
    std::string registeredModels = "visionrt.dl.yolo";

    ProbeEnvironment() {
        int rt = 0;
        if (cudaRuntimeGetVersion(&rt) == cudaSuccess && rt > 0) {
            cudaRuntimeVersion = std::to_string(rt / 1000) + "." + std::to_string((rt % 100) / 10);
        }
        int drv = 0;
        if (cudaDriverGetVersion(&drv) == cudaSuccess && drv > 0) {
            cudaDriverVersion = std::to_string(drv / 1000) + "." + std::to_string((drv % 100) / 10);
        }
    }
};

struct GpuDevice {
    int index = 0;
    std::string name;
};

struct GpuQueryResult {
    ToolStatus status;
    std::vector<GpuDevice> devices;
};

GpuQueryResult queryDevices() {
    GpuQueryResult result;
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        result.status = {-1, "CUDA device query unavailable"};
        result.devices.push_back({0, "unavailable"});
        return result;
    }
    result.status = {0, ""};
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            result.devices.push_back({i, prop.name});
        } else {
            result.devices.push_back({i, "unavailable"});
        }
    }
    return result;
}

using Tool = std::unique_ptr<IToolInstance, void(*)(IToolInstance*)>;
Tool create(const std::filesystem::path& modulePath = "") {
    const ToolServices services{PathResolver(std::filesystem::current_path())};
    ToolStatus status;
    const auto* api = getModuleApi(modulePath);
    auto* raw = api->create("visionrt.dl.yolo", &services, &status);
    success(status);
    require(raw != nullptr, "Module factory returned null for visionrt.dl.yolo");
    return Tool(raw, [](IToolInstance* p) {
        if (p) {
            const auto* a = getModuleApi();
            if (a && a->destroy) a->destroy(p);
        }
    });
}

Values configuration(const Arguments& args, bool build) {
    return {{"model_name", Utf8("yolov8n")}, {"load_mode", Utf8(build ? "build_wts" : "engine")},
        {"model_path", Utf8(absolute(build ? args.weights : args.engine))},
        {"engine_output_path", Utf8(absolute(args.engine))}, {"precision", Utf8(args.precision)},
        {"gpu_index", std::int64_t(args.device)}, {"input_width", std::int64_t(args.width)},
        {"input_height", std::int64_t(args.height)}, {"num_classes", std::int64_t(args.classes)}};
}

Values inputs(const ImageBatch& images, const DetectionParameters& parameters) {
    return {{"images", images}, {"conf_threshold", parameters.confidence},
        {"nms_threshold", parameters.nms}, {"max_detections", parameters.maximum}};
}

ImageBatch loadImages(const Arguments& args) {
    std::vector<OwnedImage> images;
    for (const auto& name : args.images) {
        cv::Mat image = cv::imread(name);
        require(!image.empty(), "Cannot load test image: " + name);
        images.push_back(OwnedImage::copyOf(image));
    }
    images.push_back(OwnedImage::take(cv::Mat(197, 321, CV_8UC3, cv::Scalar(0, 0, 0))));
    return ImageBatch(std::move(images));
}

struct PublicDetection {
    std::array<float, 4> box{};
    float score = 0.0f;
    std::int64_t classId = 0;
    std::string className;
};

std::vector<std::vector<PublicDetection>> outputContract(const ToolResult& result, const ImageBatch& images) {
    success(result.status);
    require(result.outputs.size() == 5, "Public detection output contract must expose 5 outputs");
    const auto* rois = std::get_if<RoiBatch>(&result.outputs.at("rois"));
    const auto* scores = std::get_if<NumericArray>(&result.outputs.at("scores"));
    const auto* classes = std::get_if<NumericArray>(&result.outputs.at("classes"));
    const auto* names = std::get_if<StringArray>(&result.outputs.at("class_names"));
    const auto* returned = std::get_if<ImageBatch>(&result.outputs.at("images"));
    require(rois && scores && classes && names && returned, "Invalid public output types");

    const auto count = images.images().size();
    require(rois->images().size() == count && scores->buffers().size() == count && classes->buffers().size() == count &&
            names->buffers().size() == count && returned->images().size() == count, "Mismatched batch size in outputs");

    std::vector<std::vector<PublicDetection>> batch(count);
    for (std::size_t b = 0; b < count; ++b) {
        const auto& boxes = rois->images()[b];
        const auto n = boxes.size();
        const auto& scoreBuffer = scores->buffers()[b];
        const auto& classBuffer = classes->buffers()[b];
        const auto& nameBuffer = names->buffers()[b];
        const auto rawScores = scoreBuffer.values<float>();
        const auto rawClasses = classBuffer.values<std::int64_t>();
        const auto rawNames = nameBuffer.values();
        require(rawScores.size() == n && rawClasses.size() == n && rawNames.size() == n,
                "Ragged element mismatch across outputs in batch item");
        for (std::size_t i = 0; i < n; ++i) {
            PublicDetection d;
            const auto corners = boxes[i].rectangleCorners();
            d.box = {static_cast<float>(corners[0].x), static_cast<float>(corners[0].y),
                     static_cast<float>(corners[2].x), static_cast<float>(corners[2].y)};
            d.score = rawScores[i];
            d.classId = rawClasses[i];
            d.className = rawNames[i];
            batch[b].push_back(d);
        }
    }
    return batch;
}

inline float intersectionOverUnion(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    float x1 = std::max(a[0], b[0]);
    float y1 = std::max(a[1], b[1]);
    float x2 = std::min(a[2], b[2]);
    float y2 = std::min(a[3], b[3]);
    float interArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float aArea = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    float bArea = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    float unionArea = aArea + bArea - interArea;
    if (unionArea <= 0.0f) return 0.0f;
    return interArea / unionArea;
}

inline float edgeError(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    return std::max({std::abs(a[0] - b[0]), std::abs(a[1] - b[1]), std::abs(a[2] - b[2]), std::abs(a[3] - b[3])});
}

struct ComparisonResult {
    bool passed = true;
    int unmatchedLeft = 0;
    int unmatchedRight = 0;
};

ComparisonResult compareDetections(const std::vector<PublicDetection>& a, const std::vector<PublicDetection>& b, bool exact) {
    ComparisonResult result;
    std::vector<std::vector<std::size_t>> edges(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto& left = a[i];
        for (std::size_t j = 0; j < b.size(); ++j) {
            const auto& right = b[j];
            if (left.classId != right.classId) continue;
            const bool small = std::min(left.box[2] - left.box[0], left.box[3] - left.box[1]) < 16.0f;
            const auto iou = intersectionOverUnion(left.box, right.box);
            const auto score = std::abs(left.score - right.score);
            const auto edge = edgeError(left.box, right.box);
            if ((exact && score == 0.0f && edge == 0.0f) || (!exact && score <= 0.01f && (small ? edge <= 0.5f : iou >= 0.98f)))
                edges[i].push_back(j);
        }
        std::sort(edges[i].begin(), edges[i].end(), [&](auto x, auto y) {
            return intersectionOverUnion(left.box, b[x].box) > intersectionOverUnion(left.box, b[y].box);
        });
    }
    std::vector<int> owners(b.size(), -1);
    std::function<bool(std::size_t, std::vector<bool>&)> augment = [&](std::size_t i, std::vector<bool>& visited) {
        for (const auto j : edges[i]) {
            if (visited[j]) continue;
            visited[j] = true;
            if (owners[j] < 0 || augment(static_cast<std::size_t>(owners[j]), visited)) {
                owners[j] = static_cast<int>(i);
                return true;
            }
        }
        return false;
    };
    int matches = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::vector<bool> visited(b.size(), false);
        if (augment(i, visited)) ++matches;
    }
    result.unmatchedLeft = static_cast<int>(a.size()) - matches;
    result.unmatchedRight = static_cast<int>(b.size()) - matches;
    result.passed = (result.unmatchedLeft == 0) && (result.unmatchedRight == 0);
    return result;
}

void writeYamlReport(
    const std::filesystem::path& reportPath,
    const std::string& overallStatus,
    const ProbeEnvironment& env,
    const GpuQueryResult& devices,
    const std::string& buildStatus,
    const std::string& buildReason,
    const std::string& baselineStatus,
    const std::string& baselineReason,
    const std::string& memoryStatus,
    const std::string& memoryReason,
    const std::string& hostStatus,
    const std::string& hostReason,
    const Arguments* args = nullptr,
    const std::vector<std::string>* imagePaths = nullptr,
    const std::vector<std::vector<PublicDetection>>* buildDets = nullptr
) {
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "scope" << YAML::Value << "gpu_release";
    out << YAML::Key << "overall_status" << YAML::Value << overallStatus;

    out << YAML::Key << "environment" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "sdk_version" << YAML::Value << env.sdkVersion;
    out << YAML::Key << "cuda_runtime_version" << YAML::Value << env.cudaRuntimeVersion;
    out << YAML::Key << "cuda_driver_version" << YAML::Value << env.cudaDriverVersion;
    out << YAML::Key << "devices" << YAML::Value << YAML::BeginSeq;
    for (const auto& dev : devices.devices) {
        out << YAML::BeginMap;
        out << YAML::Key << "index" << YAML::Value << dev.index;
        out << YAML::Key << "name" << YAML::Value << dev.name;
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;
    out << YAML::EndMap; // environment

    out << YAML::Key << "target_registry" << YAML::Value << "visionrt.dl.yolo";

    if (args) {
        out << YAML::Key << "parameters" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "width" << YAML::Value << args->width;
        out << YAML::Key << "height" << YAML::Value << args->height;
        out << YAML::Key << "classes" << YAML::Value << args->classes;
        out << YAML::Key << "precision" << YAML::Value << args->precision;
        out << YAML::Key << "confidence" << YAML::Value << args->detection.confidence;
        out << YAML::Key << "nms" << YAML::Value << args->detection.nms;
        out << YAML::Key << "max_detections" << YAML::Value << args->detection.maximum;
        out << YAML::EndMap;
    }

    out << YAML::Key << "images" << YAML::Value << YAML::BeginSeq;
    if (imagePaths && buildDets) {
        for (std::size_t i = 0; i < imagePaths->size(); ++i) {
            out << YAML::BeginMap;
            out << YAML::Key << "path" << YAML::Value << (*imagePaths)[i];
            out << YAML::Key << "detections" << YAML::Value << YAML::BeginSeq;
            if (i < buildDets->size()) {
                for (const auto& det : (*buildDets)[i]) {
                    out << YAML::BeginMap;
                    out << YAML::Key << "class_id" << YAML::Value << det.classId;
                    out << YAML::Key << "score" << YAML::Value << det.score;
                    out << YAML::Key << "box" << YAML::Value << YAML::Flow << YAML::BeginSeq
                        << det.box[0] << det.box[1] << det.box[2] << det.box[3] << YAML::EndSeq;
                    out << YAML::EndMap;
                }
            }
            out << YAML::EndSeq;
            out << YAML::EndMap;
        }
    }
    out << YAML::EndSeq;

    out << YAML::Key << "checks" << YAML::Value << YAML::BeginMap;

    out << YAML::Key << "model_build_and_reload" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "status" << YAML::Value << buildStatus;
    if (!buildReason.empty()) out << YAML::Key << "reason" << YAML::Value << buildReason;
    out << YAML::EndMap;

    out << YAML::Key << "approved_baseline_comparison" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "status" << YAML::Value << baselineStatus;
    if (!baselineReason.empty()) out << YAML::Key << "reason" << YAML::Value << baselineReason;
    out << YAML::EndMap;

    out << YAML::Key << "memory_lifecycle" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "status" << YAML::Value << memoryStatus;
    if (!memoryReason.empty()) out << YAML::Key << "reason" << YAML::Value << memoryReason;
    out << YAML::EndMap;

    out << YAML::Key << "host_integration" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "status" << YAML::Value << hostStatus;
    if (!hostReason.empty()) out << YAML::Key << "reason" << YAML::Value << hostReason;
    out << YAML::EndMap;

    out << YAML::EndMap; // checks
    out << YAML::EndMap; // root

    std::filesystem::create_directories(std::filesystem::absolute(reportPath).parent_path());
    std::ofstream fout(reportPath);
    if (!fout.is_open()) {
        throw std::runtime_error("Cannot write report file: " + reportPath.string());
    }
    fout << out.c_str() << std::endl;
}

void testSyntheticRoundtrip(const std::filesystem::path& tempReportPath) {
    ProbeEnvironment env;
    GpuQueryResult devices = queryDevices();
    writeYamlReport(tempReportPath, "not_run", env, devices,
                    "not_run", "未提供真实权重及运行环境",
                    "not_run", "未提供批准基准与实测输出",
                    "not_run", "未运行",
                    "not_run", "未提供宿主联合运行环境");

    // Read back and verify structure (YS-04)
    YAML::Node doc = YAML::LoadFile(tempReportPath.string());
    require(doc["scope"] && doc["scope"].as<std::string>() == "gpu_release", "Invalid scope");
    require(doc["overall_status"] && doc["overall_status"].as<std::string>() == "not_run", "Invalid overall_status");
    require(doc["environment"] && doc["environment"]["sdk_version"], "Invalid environment");
    require(doc["target_registry"] && doc["target_registry"].as<std::string>() == "visionrt.dl.yolo", "Invalid target_registry");
    require(doc["checks"] && doc["checks"]["model_build_and_reload"], "Missing checks");

    // Synthetic comparison test
    PublicDetection d1{{10.0f, 10.0f, 50.0f, 50.0f}, 0.9f, 0, "person"};
    PublicDetection d2{{10.0f, 10.0f, 50.0f, 50.0f}, 0.9f, 0, "person"};
    auto exactRes = compareDetections({d1}, {d2}, true);
    require(exactRes.passed, "Exact match synthetic test failed");

    PublicDetection d3{{100.0f, 100.0f, 150.0f, 150.0f}, 0.8f, 1, "car"};
    auto unmatchRes = compareDetections({d1}, {d3}, true);
    require(!unmatchRes.passed && unmatchRes.unmatchedLeft == 1, "Unmatched synthetic test failed");
}

} // namespace

int main(int argc, char** argv) {
    try {
        auto args = arguments(argc, argv);
        ProbeEnvironment environment;
        const auto devices = queryDevices();

        if (args.syntheticOnly) {
            std::cout << "[INFO] Running synthetic YAML report test..." << std::endl;
            testSyntheticRoundtrip(args.report);
            std::cout << "[SUCCESS] Synthetic probe report written and verified at: " << args.report << std::endl;
            return 0;
        }

        if (args.modulePath.empty() && !args.packageDir.empty()) {
            const auto idxPath = std::filesystem::path(args.packageDir) / "module_index.yaml";
            if (!std::filesystem::exists(idxPath)) {
                throw std::runtime_error("module_index.yaml not found in package directory: " + args.packageDir);
            }
            YAML::Node idx = YAML::LoadFile(idxPath.string());
            if (!idx["modules"] || !idx["modules"]["visiontools_infer"] || !idx["modules"]["visiontools_infer"]["library"]) {
                throw std::runtime_error("visiontools_infer library not declared in " + idxPath.string());
            }
            const auto libRel = idx["modules"]["visiontools_infer"]["library"].as<std::string>();
            args.modulePath = (std::filesystem::path(args.packageDir) / libRel).string();
        }

        std::filesystem::create_directories(std::filesystem::absolute(std::filesystem::path(args.engine)).parent_path());
        std::filesystem::create_directories(std::filesystem::absolute(std::filesystem::path(args.report)).parent_path());
        const auto images = loadImages(args);

        auto built = create(args.modulePath);
        success(built->initialize(configuration(args, true)));
        success(devices.status);

        const auto batchInputs = inputs(images, args.detection);
        const auto buildResult = built->execute(batchInputs);
        const auto buildDetections = outputContract(buildResult, images);

        const auto emptyInputs = inputs(ImageBatch{}, args.detection);
        const auto emptyResult = built->execute(emptyInputs);
        outputContract(emptyResult, std::get<ImageBatch>(emptyInputs.at("images")));

        auto bad = batchInputs;
        bad["images"] = Missing{};
        allMissing(built->execute(bad));
        bad = batchInputs;
        bad["nms_threshold"] = std::numeric_limits<double>::infinity();
        allMissing(built->execute(bad));
        bad = batchInputs;
        bad["images"] = ImageBatch({images.images()[0], OwnedImage::take(cv::Mat(4, 4, CV_32FC1, cv::Scalar(0)))});
        allMissing(built->execute(bad));

        bool thresholdEqualKeeps = false, thresholdAboveExcludes = false;
        for (std::size_t b = 0; b < buildDetections.size(); ++b) {
            if (buildDetections[b].empty()) continue;
            const auto& target = buildDetections[b].front();
            if (target.score >= 1.0f) continue;
            const double boundaryScore = target.score;
            const auto single = ImageBatch({images.images()[b]});
            auto boundary = inputs(single, args.detection);
            boundary["conf_threshold"] = boundaryScore;
            const auto equalResult = built->execute(boundary);
            const auto equalDets = outputContract(equalResult, single);
            thresholdEqualKeeps = !equalDets.empty() && !equalDets[0].empty();

            boundary["conf_threshold"] = std::nextafter(boundaryScore, 1.0);
            const auto aboveResult = built->execute(boundary);
            const auto aboveDets = outputContract(aboveResult, single);
            thresholdAboveExcludes = aboveDets.empty() || aboveDets[0].empty() ||
                std::none_of(aboveDets[0].begin(), aboveDets[0].end(),
                    [&](const auto& d) { return d.classId == target.classId && std::abs(d.score - target.score) < 1e-4; });
            break;
        }

        auto stricter = batchInputs;
        stricter["conf_threshold"] = 1.0;
        success(built->execute(stricter).status);

        // Snapshot persistence across release and destroy
        success(built->release());
        built.reset();
        outputContract(buildResult, images);

        auto loaded = create(args.modulePath);
        success(loaded->initialize(configuration(args, false)));
        const auto loadResult = loaded->execute(batchInputs);
        const auto loadDetections = outputContract(loadResult, images);

        bool passed = thresholdEqualKeeps && thresholdAboveExcludes;
        for (std::size_t b = 0; b < images.images().size(); ++b) {
            const auto single = ImageBatch({images.images()[b]});
            const auto singleInputs = inputs(single, args.detection);
            const auto singleResult = loaded->execute(singleInputs);
            const auto singleDetections = outputContract(singleResult, single);

            const auto singleDets = singleDetections.empty() ? std::vector<PublicDetection>{} : singleDetections[0];
            auto cmp1 = compareDetections(loadDetections[b], singleDets, true);
            auto cmp2 = compareDetections(buildDetections[b], loadDetections[b], false);
            passed = cmp1.passed && cmp2.passed && passed;
        }

        const std::string buildStatus = passed ? "passed" : "failed";
        const std::string overallStatus = passed ? "not_run" : "failed";

        writeYamlReport(
            args.report,
            overallStatus,
            environment,
            devices,
            buildStatus,
            "",
            "not_run",
            "未提供批准基准对比结果",
            "not_run",
            "需独立内存采样",
            "not_run",
            "未提供宿主联合运行环境",
            &args,
            &args.images,
            &buildDetections
        );

        if (!passed) return 2;
        return 3; // not_run exit code for incomplete scope
    } catch (const std::exception& error) {
        std::cerr << "Probe error: " << error.what() << std::endl;
        return 1;
    }
}
