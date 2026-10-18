#include <visionrt/sdk/Module.h>
#include <visionrt/sdk/Tool.h>
#include <visionrt/sdk/Contract.h>
#include <visionrt/sdk/Value.h>
#include <visionrt/common/VersionDef.h>
#include <cuda_runtime.h>

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
#define NOMINMAX
#include <windows.h>
#include <winver.h>
#endif
#include <stdexcept>
#include <string>
#include <vector>

extern "C" VT_EXPORT const visionrt::ModuleApi* VT_CALL visiontoolsModuleApi() noexcept;

using namespace visionrt;

namespace {

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
    std::string weights, engine, report, precision = "fp32";
    std::vector<std::string> images;
    int device = 0, width = 640, height = 640, classes = 80;
    DetectionParameters detection;
};

Arguments arguments(int argc, char** argv) {
    Arguments result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 == argc) throw std::runtime_error("Missing argument after " + key);
        const std::string value = argv[++i];
        if (key == "--wts") result.weights = value;
        else if (key == "--engine") result.engine = value;
        else if (key == "--report") result.report = value;
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
    if (result.weights.empty() || result.engine.empty() || result.report.empty() || result.images.size() < 2)
        throw std::runtime_error("Required: --wts FILE --engine FILE --report FILE.xml --image FILE --image FILE; "
                                 "an actual 321x197 black image is appended automatically");
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

std::string absolute(const std::string& path) { return utf8(std::filesystem::absolute(std::filesystem::u8path(path))); }

struct ProbeEnvironment {
    std::string sdkVersion = VISIONRT_VERSION_STRING;
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
Tool create() {
    const ToolServices services{PathResolver(std::filesystem::current_path())};
    ToolStatus status;
    const auto* api = visiontoolsModuleApi();
    require(api != nullptr, "visiontoolsModuleApi returned null");
    auto* raw = api->create("visionrt.dl.yolo", &services, &status);
    success(status);
    require(raw != nullptr, "Module factory returned null for visionrt.dl.yolo");
    return Tool(raw, [](IToolInstance* p) {
        if (p) {
            const auto* a = visiontoolsModuleApi();
            if (a) a->destroy(p);
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
        std::ifstream file(std::filesystem::u8path(name), std::ios::binary | std::ios::ate);
        require(bool(file), "Cannot open image: " + name);
        const auto length = file.tellg();
        require(length > 0, "Empty image file: " + name);
        std::vector<unsigned char> bytes(static_cast<std::size_t>(length));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        require(bool(file), "Cannot read image: " + name);
        auto image = cv::imdecode(bytes, cv::IMREAD_UNCHANGED);
        require(!image.empty(), "Cannot decode image: " + name);
        images.push_back(OwnedImage::take(std::move(image)));
    }
    images.push_back(OwnedImage::take(cv::Mat(197, 321, CV_8UC3, cv::Scalar::all(0))));
    return ImageBatch(std::move(images));
}

struct PublicDetection {
    int classId = 0;
    float score = 0.0f;
    std::array<float, 4> box = {0, 0, 0, 0}; // x1, y1, x2, y2
};

std::vector<std::vector<PublicDetection>> outputContract(const ToolResult& result, const ImageBatch& input) {
    success(result.status);
    require(result.outputs.size() == 5, "YOLO output field count mismatch");
    require(std::get<ImageBatch>(result.outputs.at("images")).owner() == input.owner(), "YOLO copied/replaced input images");
    const auto rois = std::get<RoiBatch>(result.outputs.at("rois")).images();
    const auto& scores = std::get<NumericArray>(result.outputs.at("scores"));
    const auto& classes = std::get<NumericArray>(result.outputs.at("classes"));
    const auto& names = std::get<StringArray>(result.outputs.at("class_names"));
    validateValue({Type::NumericArray, DType::Float32, 2, Layout::BatchRagged}, scores);
    validateValue({Type::NumericArray, DType::Int64, 2, Layout::BatchRagged}, classes);
    validateValue({Type::StringArray, {}, 2, Layout::BatchRagged}, names);
    const auto batch = input.images().size();
    require(rois.size() == batch && scores.buffers().size() == batch && classes.buffers().size() == batch &&
            names.buffers().size() == batch, "Detection batch dimensions disagree");

    std::vector<std::vector<PublicDetection>> batchDetections(batch);
    for (std::size_t b = 0; b < batch; ++b) {
        const auto s = scores.buffers()[b].values<float>();
        const auto c = classes.buffers()[b].values<std::int64_t>();
        const auto n = names.buffers()[b].values();
        require(rois[b].size() == s.size() && c.size() == s.size() && n.size() == s.size(),
                "Five-field detection correspondence failed");
        for (std::size_t i = 0; i < s.size(); ++i) {
            const auto box = rois[b][i].encoded();
            require(box[0] == 2 && box[5] == 0, "Invalid ROI bounding box encoding");
            require(box[3] >= 0.0f && box[4] >= 0.0f, "Invalid ROI dimensions");
            require(s[i] >= 0.0f && s[i] <= 1.0f, "Detection score out of range [0, 1]");
            require(c[i] >= 0, "Class ID must be non-negative");
            require(n[i] == std::to_string(c[i]), "Class name should correspond to class ID string");
            PublicDetection d;
            d.classId = static_cast<int>(c[i]);
            d.score = s[i];
            d.box = {box[1], box[2], box[1] + box[3], box[2] + box[4]};
            batchDetections[b].push_back(d);
        }
    }
    return batchDetections;
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

bool compareDetections(cv::FileStorage& file, const std::vector<PublicDetection>& a, const std::vector<PublicDetection>& b, bool exact) {
    bool passed = true;
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
    file << "{" << "matches" << "[";
    for (std::size_t j = 0; j < owners.size(); ++j) if (owners[j] >= 0) {
        const auto& left = a[static_cast<std::size_t>(owners[j])];
        const auto& right = b[j];
        file << "{" << "left_index" << owners[j] << "right_index" << static_cast<int>(j)
             << "class_id" << static_cast<int>(left.classId) << "iou" << intersectionOverUnion(left.box, right.box)
             << "score_difference" << std::abs(left.score - right.score) << "maximum_edge_error" << edgeError(left.box, right.box) << "}";
    }
    const auto unmatchedA = static_cast<int>(a.size()) - matches;
    const auto unmatchedB = static_cast<int>(b.size()) - matches;
    passed = passed && (unmatchedA == 0) && (unmatchedB == 0);
    file << "]" << "unmatched_left" << unmatchedA << "unmatched_right" << unmatchedB << "passed" << int(passed) << "}";
    return passed;
}

#ifdef _WIN32
void loadedLibraries(cv::FileStorage& report) {
    report << "loaded_libraries" << "[";
    for (const auto* name : {L"visiontools_infer.dll", L"inferrt_model.dll", L"inferrt_core.dll",
                             L"nvinfer_10.dll", L"cudart64_11.dll", L"cudart64_12.dll", L"nvcuda.dll",
                             L"opencv_world480.dll", L"opencv_core480.dll"}) {
        const auto handle = GetModuleHandleW(name);
        if (!handle) continue;
        std::vector<wchar_t> path(32768);
        const auto size = GetModuleFileNameW(handle, path.data(), static_cast<DWORD>(path.size()));
        require(size > 0 && size < path.size(), "Cannot query a loaded module filename");
        const std::filesystem::path filename(std::wstring(path.data(), size));
        report << "{" << "path" << utf8(filename);
        DWORD ignored = 0;
        const auto versionSize = GetFileVersionInfoSizeW(filename.c_str(), &ignored);
        std::vector<std::byte> version(versionSize);
        if (versionSize && GetFileVersionInfoW(filename.c_str(), 0, versionSize, version.data())) {
            VS_FIXEDFILEINFO* information = nullptr;
            UINT informationSize = 0;
            if (VerQueryValueW(version.data(), L"\\", reinterpret_cast<void**>(&information), &informationSize) &&
                information && informationSize >= sizeof(VS_FIXEDFILEINFO)) {
                report << "file_version" << (std::to_string(HIWORD(information->dwFileVersionMS)) + "." +
                    std::to_string(LOWORD(information->dwFileVersionMS)) + "." +
                    std::to_string(HIWORD(information->dwFileVersionLS)) + "." +
                    std::to_string(LOWORD(information->dwFileVersionLS)));
            }
        }
        report << "}";
    }
    report << "]";
}
#endif

} // namespace

int main(int argc, char** argv) {
    try {
        const auto args = arguments(argc, argv);
        std::filesystem::create_directories(std::filesystem::absolute(std::filesystem::u8path(args.engine)).parent_path());
        std::filesystem::create_directories(std::filesystem::absolute(std::filesystem::u8path(args.report)).parent_path());
        const auto images = loadImages(args);

        auto built = create();
        success(built->initialize(configuration(args, true)));

        ProbeEnvironment environment;
        const auto devices = queryDevices();
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

        auto loaded = create();
        success(loaded->initialize(configuration(args, false)));
        const auto loadResult = loaded->execute(batchInputs);
        const auto loadDetections = outputContract(loadResult, images);

        cv::FileStorage report(args.report, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_XML);
        require(report.isOpened(), "Cannot write report: " + args.report);
        report << "schema" << "VisionRT.VR3-002.VR3-027.probe.2"
               << "sdk_version" << environment.sdkVersion << "opencv_version" << CV_VERSION
               << "cuda_runtime_version" << environment.cudaRuntimeVersion << "cuda_driver_api_version" << environment.cudaDriverVersion
               << "precision" << args.precision << "weights" << absolute(args.weights) << "engine" << absolute(args.engine)
               << "confidence" << args.detection.confidence << "nms" << args.detection.nms << "maximum_detections" << int(args.detection.maximum)
               << "width" << args.width << "height" << args.height << "num_classes" << args.classes;

#ifdef _MSC_FULL_VER
        report << "msvc_full_version" << _MSC_FULL_VER << "crt" << "MD";
#ifdef _WIN32
        loadedLibraries(report);
#endif
#endif
        report << "registered_models" << environment.registeredModels << "devices" << "[";
        for (const auto& device : devices.devices) report << "{" << "index" << device.index << "name" << device.name << "}";
        report << "]" << "images" << "[";

        bool passed = thresholdEqualKeeps && thresholdAboveExcludes;
        for (std::size_t b = 0; b < images.images().size(); ++b) {
            const auto single = ImageBatch({images.images()[b]});
            const auto singleInputs = inputs(single, args.detection);
            const auto singleResult = loaded->execute(singleInputs);
            const auto singleDetections = outputContract(singleResult, single);

            report << "{" << "index" << static_cast<int>(b);
            report << "batch_vs_single_exact";
            const auto singleDets = singleDetections.empty() ? std::vector<PublicDetection>{} : singleDetections[0];
            passed = compareDetections(report, loadDetections[b], singleDets, true) && passed;
            report << "build_vs_load";
            passed = compareDetections(report, buildDetections[b], loadDetections[b], false) && passed;
            report << "}";
        }
        report << "]" << "passed" << int(passed);
        return passed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "Probe error: " << error.what() << std::endl;
        return 1;
    }
}
