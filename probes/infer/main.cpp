#include "infer/InferProtocol.h"
#include "infer/YoloModel.h"

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
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winver.h>
#endif
#include <stdexcept>

using namespace visionrt;
using namespace visionrt::infer;
namespace {
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
using Tool = std::unique_ptr<IToolInstance, decltype(&visionrtInferDestroy)>;
Tool create() {
    const ToolServices services{PathResolver(std::filesystem::current_path())};
    ToolStatus status;
    Tool tool(visionrtInferCreate(&services, &status), visionrtInferDestroy);
    success(status);
    require(bool(tool), "Module factory returned null");
    return tool;
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
void outputContract(const ToolResult& result, const ImageBatch& input, const std::vector<ImageTrace>& traces) {
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
            names.buffers().size() == batch && traces.size() == batch, "Detection batch dimensions disagree");
    for (std::size_t b = 0; b < batch; ++b) {
        const auto s = scores.buffers()[b].values<float>();
        const auto c = classes.buffers()[b].values<std::int64_t>();
        const auto n = names.buffers()[b].values();
        require(rois[b].size() == traces[b].detections.size() && s.size() == rois[b].size() &&
                c.size() == s.size() && n.size() == s.size(), "Five-field detection correspondence failed");
        for (std::size_t i = 0; i < s.size(); ++i) {
            const auto& detection = traces[b].detections[i];
            const auto box = rois[b][i].encoded();
            require(box[0] == 2 && box[5] == 0 && box[1] == float(detection.box.x1) && box[2] == float(detection.box.y1) &&
                    box[3] == float(detection.box.x2 - detection.box.x1) && box[4] == float(detection.box.y2 - detection.box.y1) &&
                    s[i] == detection.score && c[i] == detection.classId && n[i] == std::to_string(c[i]),
                    "Detection fields refer to different targets");
        }
    }
}
void box(cv::FileStorage& file, const Box& value) {
    file << "[" << value.x1 << value.y1 << value.x2 << value.y2 << "]";
}
void geometry(cv::FileStorage& file, const LetterboxGeometry& g) {
    file << "{" << "original" << "[" << g.originalWidth << g.originalHeight << "]"
         << "input" << "[" << g.inputWidth << g.inputHeight << "]"
         << "resized" << "[" << g.resizedWidth << g.resizedHeight << "]"
         << "padding_ltrb" << "[" << g.padLeft << g.padTop << g.inputWidth - g.resizedWidth - g.padLeft
         << g.inputHeight - g.resizedHeight - g.padTop << "]"
         << "sdk_scale" << g.sdkScale << "sx" << double(g.resizedWidth) / g.originalWidth
         << "sy" << double(g.resizedHeight) / g.originalHeight << "landmarks" << "[";
    for (const auto y : {0.0, g.originalHeight / 2.0, double(g.originalHeight)})
        for (const auto x : {0.0, g.originalWidth / 2.0, double(g.originalWidth)}) {
            const auto nx = x * g.resizedWidth / g.originalWidth + g.padLeft;
            const auto ny = y * g.resizedHeight / g.originalHeight + g.padTop;
            const auto back = toSource({nx, ny, nx, ny}, g);
            const auto error = std::max(std::abs(x - back.x1), std::abs(y - back.y1));
            file << "{" << "source" << "[" << x << y << "]" << "network" << "[" << nx << ny << "]"
                 << "roundtrip_error" << error << "}";
            require(error < 1e-9, "Actual letterbox boundary landmark failed round trip");
        }
    file << "]" << "}";
}
void trace(cv::FileStorage& file, const ImageTrace& value, const Arguments& args) {
    file << "{" << "geometry";
    geometry(file, value.geometry);
    file << "branches" << "[";
    std::size_t offset = 0;
    for (std::size_t branch = 0; branch < 3; ++branch) {
        const int stride = 8 << branch;
        const auto grid = static_cast<std::size_t>(args.width / stride) * (args.height / stride);
        const auto& raw = value.rawOutputs[branch];
        file << "{" << "name" << "output" + std::to_string(branch) << "stride" << stride
             << "raw_tensor" << cv::Mat(1, static_cast<int>(raw.size()), CV_32F, const_cast<float*>(raw.data()))
             << "candidates" << "[";
        // Keep the strongest candidate plus every threshold-near candidate, with
        // original raw values. Complete raw tensors above permit manual audit.
        const auto begin = value.candidates.begin() + static_cast<std::ptrdiff_t>(offset);
        const auto end = begin + static_cast<std::ptrdiff_t>(grid);
        const auto strongest = std::max_element(begin, end, [](const auto& a, const auto& b) { return a.score < b.score; });
        for (auto it = begin; it != end; ++it) {
            if (it != strongest && std::abs(it->score - args.detection.confidence) > 0.02) continue;
            const auto index = it->candidateIndex - offset;
            file << "{" << "candidate_index" << static_cast<int>(it->candidateIndex)
                 << "grid_index" << static_cast<int>(index) << "class_id" << static_cast<int>(it->classId)
                 << "ltrb" << "[" << raw[index] << raw[grid + index] << raw[2 * grid + index] << raw[3 * grid + index] << "]"
                 << "class_logit" << raw[(4 + static_cast<std::size_t>(it->classId)) * grid + index]
                 << "score" << it->score << "network_box";
            box(file, it->box);
            file << "}";
        }
        file << "]" << "}";
        offset += grid;
    }
    file << "]" << "detections" << "[";
    for (const auto& detection : value.detections) {
        file << "{" << "class_id" << static_cast<int>(detection.classId) << "score" << detection.score
             << "candidate_index" << static_cast<int>(detection.candidateIndex) << "box";
        box(file, detection.box);
        file << "}";
    }
    file << "]" << "}";
}
double edgeError(const Box& a, const Box& b) {
    return std::max({std::abs(a.x1 - b.x1), std::abs(a.y1 - b.y1), std::abs(a.x2 - b.x2), std::abs(a.y2 - b.y2)});
}
bool comparison(cv::FileStorage& file, const ImageTrace& a, const ImageTrace& b, bool exact) {
    bool passed = true;
    file << "{" << "raw_max_abs" << "[";
    for (std::size_t branch = 0; branch < 3; ++branch) {
        require(a.rawOutputs[branch].size() == b.rawOutputs[branch].size(), "Raw tensor shape changed");
        double maximum = 0;
        for (std::size_t i = 0; i < a.rawOutputs[branch].size(); ++i)
            maximum = std::max(maximum, std::abs(double(a.rawOutputs[branch][i]) - b.rawOutputs[branch][i]));
        file << maximum;
        if (exact && maximum != 0) passed = false;
    }
    file << "]";
    std::vector<std::vector<std::size_t>> edges(a.detections.size());
    for (std::size_t i = 0; i < a.detections.size(); ++i) {
        const auto& left = a.detections[i];
        for (std::size_t j = 0; j < b.detections.size(); ++j) {
            const auto& right = b.detections[j];
            if (left.classId != right.classId) continue;
            const bool small = std::min(left.box.x2 - left.box.x1, left.box.y2 - left.box.y1) < 16;
            const auto iou = intersectionOverUnion(left.box, right.box);
            const auto score = std::abs(left.score - right.score);
            const auto edge = edgeError(left.box, right.box);
            if ((exact && score == 0 && edge == 0) || (!exact && score <= 0.01 && (small ? edge <= 0.5 : iou >= 0.98)))
                edges[i].push_back(j);
        }
        std::sort(edges[i].begin(), edges[i].end(), [&](auto x, auto y) {
            return intersectionOverUnion(left.box, b.detections[x].box) > intersectionOverUnion(left.box, b.detections[y].box);
        });
    }
    std::vector<int> owners(b.detections.size(), -1);
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
    for (std::size_t i = 0; i < a.detections.size(); ++i) {
        std::vector<bool> visited(b.detections.size());
        if (augment(i, visited)) ++matches;
    }
    file << "matches" << "[";
    for (std::size_t j = 0; j < owners.size(); ++j) if (owners[j] >= 0) {
        const auto& left = a.detections[static_cast<std::size_t>(owners[j])];
        const auto& right = b.detections[j];
#ifdef _WIN32
void loadedLibraries(cv::FileStorage& report) {
    report << "loaded_libraries" << "[";
    for (const auto* name : {L"VisionRTInfer.dll", L"inferrt_model.dll", L"inferrt_core.dll",
                             L"nvinfer_10.dll", L"cudart64_12.dll", L"nvcuda.dll",
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
        file << "{" << "left_index" << owners[j] << "right_index" << static_cast<int>(j)
             << "class_id" << static_cast<int>(left.classId) << "iou" << intersectionOverUnion(left.box, right.box)
             << "score_difference" << std::abs(left.score - right.score) << "maximum_edge_error" << edgeError(left.box, right.box) << "}";
    }
    const auto unmatchedA = static_cast<int>(a.detections.size()) - matches;
    const auto unmatchedB = static_cast<int>(b.detections.size()) - matches;
    passed = passed && unmatchedA == 0 && unmatchedB == 0;
    file << "]" << "unmatched_left" << unmatchedA << "unmatched_right" << unmatchedB << "passed" << int(passed) << "}";
    return passed;
}
}

int main(int argc, char** argv) {
    try {
        const auto args = arguments(argc, argv);
        std::filesystem::create_directories(std::filesystem::absolute(std::filesystem::u8path(args.engine)).parent_path());
        std::filesystem::create_directories(std::filesystem::absolute(std::filesystem::u8path(args.report)).parent_path());
        const auto images = loadImages(args);
        auto built = create();
        success(built->initialize(configuration(args, true)));
        ProbeEnvironment environment;
        std::vector<TensorDescription> tensors;
        ToolStatus status;
        visionrtInferInspect(built.get(), &tensors, &environment, &status);
        success(status);
        GpuQueryResult devices;
        visionrtInferQueryDevices(&devices);
        success(devices.status);
        GeometryProbe pixelProbe;
        visionrtInferProbeGeometry(&pixelProbe, &status);
        success(status);
        const auto batchInputs = inputs(images, args.detection);
        std::vector<ImageTrace> buildTraces;
        const auto buildResult = visionrtInferExecuteWithTrace(built.get(), &batchInputs, &buildTraces);
        outputContract(buildResult, images, buildTraces);
        const auto emptyInputs = inputs(ImageBatch{}, args.detection);
        std::vector<ImageTrace> emptyTraces;
        outputContract(visionrtInferExecuteWithTrace(built.get(), &emptyInputs, &emptyTraces),
                       std::get<ImageBatch>(emptyInputs.at("images")), emptyTraces);
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
        double boundaryScore = 0;
        std::size_t boundaryCandidate = 0;
        for (std::size_t b = 0; b < buildTraces.size(); ++b) {
            if (buildTraces[b].detections.empty()) continue;
            const auto& target = buildTraces[b].detections.front();
            if (target.score >= 1.0F) continue;
            boundaryScore = target.score;
            boundaryCandidate = target.candidateIndex;
            const auto single = ImageBatch({images.images()[b]});
            auto boundary = inputs(single, args.detection);
            boundary["conf_threshold"] = boundaryScore;
            std::vector<ImageTrace> equalTraces, aboveTraces;
            outputContract(visionrtInferExecuteWithTrace(built.get(), &boundary, &equalTraces), single, equalTraces);
            thresholdEqualKeeps = std::any_of(equalTraces[0].detections.begin(), equalTraces[0].detections.end(),
                [&](const auto& value) { return value.candidateIndex == boundaryCandidate; });
            boundary["conf_threshold"] = std::nextafter(boundaryScore, 1.0);
            outputContract(visionrtInferExecuteWithTrace(built.get(), &boundary, &aboveTraces), single, aboveTraces);
            thresholdAboveExcludes = std::none_of(aboveTraces[0].detections.begin(), aboveTraces[0].detections.end(),
                [&](const auto& value) { return value.candidateIndex == boundaryCandidate; });
            break;
        }
        auto stricter = batchInputs;
        stricter["conf_threshold"] = 1.0;
        success(built->execute(stricter).status);
        // The saved snapshot must remain valid after reuse and module-owned destruction.
        success(built->release());
        built.reset();
        outputContract(buildResult, images, buildTraces);
        auto loaded = create();
        success(loaded->initialize(configuration(args, false)));
        std::vector<ImageTrace> loadTraces;
        outputContract(visionrtInferExecuteWithTrace(loaded.get(), &batchInputs, &loadTraces), images, loadTraces);
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
#ifdef VISIONRT_PROBE_SDK_DIR
        report << "sdk_package_dir" << VISIONRT_PROBE_SDK_DIR;
#endif
#endif
        report << "registered_models" << environment.registeredModels << "devices" << "[";
        for (const auto& device : devices.devices) report << "{" << "index" << device.index << "name" << device.name << "}";
        report << "]" << "tensors" << "[";
        for (const auto& tensor : tensors) {
            report << "{" << "name" << tensor.name << "dtype" << tensor.dtype << "layout" << tensor.layout << "shape" << "[";
            for (const auto dimension : tensor.shape) report << static_cast<int>(dimension);
            report << "]" << "}";
        }
        report << "]" << "pixel_center_probe" << "{" << "geometry";
        geometry(report, pixelProbe.geometry);
        report << "maximum_8bit_pixel_error" << pixelProbe.maximumPixelError
               << "padding_8bit_pixel_error" << pixelProbe.paddingPixelError << "}" << "images" << "[";
        bool passed = pixelProbe.maximumPixelError <= 1.0 && pixelProbe.paddingPixelError < 0.001 &&
                      thresholdEqualKeeps && thresholdAboveExcludes;
        for (std::size_t b = 0; b < images.images().size(); ++b) {
            const auto single = ImageBatch({images.images()[b]});
            const auto singleInputs = inputs(single, args.detection);
            std::vector<ImageTrace> singleTraces;
            outputContract(visionrtInferExecuteWithTrace(loaded.get(), &singleInputs, &singleTraces), single, singleTraces);
            report << "{" << "path" << (b < args.images.size() ? absolute(args.images[b]) : "generated:black:321x197") << "build";
            trace(report, buildTraces[b], args);
            report << "load";
            trace(report, loadTraces[b], args);
            report << "build_load_comparison";
            passed = comparison(report, buildTraces[b], loadTraces[b], false) && passed;
            report << "batch_single_comparison";
            passed = comparison(report, loadTraces[b], singleTraces[0], true) && passed;
            report << "}";
        }
        const bool zero = buildTraces.back().detections.empty() && loadTraces.back().detections.empty();
        bool differentSizes = false, differentCounts = false;
        for (std::size_t b = 1; b < args.images.size(); ++b) {
            differentSizes = differentSizes || images.images()[b].cols() != images.images()[0].cols() ||
                                              images.images()[b].rows() != images.images()[0].rows();
            differentCounts = differentCounts || buildTraces[b].detections.size() != buildTraces[0].detections.size();
        }
        passed = passed && zero && differentSizes && differentCounts;
        success(loaded->release());
        report << "]" << "generated_black_zero_targets" << int(zero) << "different_real_image_sizes" << int(differentSizes)
               << "different_real_image_target_counts" << int(differentCounts)
               << "threshold_boundary" << "{" << "score" << boundaryScore
               << "candidate_index" << static_cast<int>(boundaryCandidate)
               << "equal_keeps" << int(thresholdEqualKeeps) << "next_double_excludes" << int(thresholdAboveExcludes) << "}"
               << "independent_python_comparison" << "not_run_use_reference.py" << "passed" << int(passed);
        report.release();
        std::cout << "build/save/load and batch/single report: " << args.report << "; passed=" << passed << '\n';
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Inference probe failed: " << error.what() << '\n';
        return 1;
    }
}
