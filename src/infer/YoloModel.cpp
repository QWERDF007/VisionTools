#include "YoloModel.h"

#include "InferProtocol.h"

#include "common/InputValues.h"
#include "common/UpstreamStatus.h"
#include "common/errors/ToolError.h"

#include "errors/YoloErrors.h"

#include <inferrt/model/Buffers.hpp>
#include <inferrt/model/ModelFactory.hpp>
#include <inferrt/model/Utils.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace visionrt::infer {

// This module keeps its historical visionrt::infer namespace; the shared tool error vocabulary
// lives in visiontools, so the names it uses are bound once instead of at every call site.
using visiontools::CommonToolError;
using visiontools::ToolFailure;
using visiontools::fail;
using visiontools::status;

namespace {
namespace yolo_ports = ports::yolo;
constexpr std::array                  strides{8, 16, 32};
constexpr std::array<const char *, 3> outputNames{"output0", "output1", "output2"};

template<class T>
T requiredParameter(const Values &values, const char *name)
{
    const auto found = values.find(name);
    if (found == values.end() || isMissing(found->second))
        fail(CommonToolError::InputMissing, std::string("初始化参数 ") + name);
    if (const auto *value = std::get_if<T>(&found->second))
        return *value;
    fail(CommonToolError::InputTypeMismatch, std::string("初始化参数 ") + name);
}

int integer(const Values &values, const char *name, int minimum)
{
    const auto value = requiredParameter<std::int64_t>(values, name);
    if (value < minimum || value > std::numeric_limits<int>::max())
        fail(CommonToolError::InputOutOfRange,
             std::string(name) + " 的值 " + std::to_string(value) + " 超出整数范围");
    return static_cast<int>(value);
}

std::string text(const Values &values, const char *name)
{
    return requiredParameter<std::string>(values, name);
}

std::string utf8Path(const std::filesystem::path &path)
{
    const auto value = path.u8string();
    return {reinterpret_cast<const char *>(value.data()), value.size()};
}

irt::PreprocessSpec preprocessingSpec(int width, int height)
{
    irt::PreprocessSpec spec;
    spec.input_width    = width;
    spec.input_height   = height;
    spec.input_channels = spec.source_channels = 3;
    spec.src_color                             = irt::ColorFormat::BGR;
    spec.dst_color                             = irt::ColorFormat::RGB;
    spec.padding_mode                          = irt::PaddingMode::Letterbox;
    spec.pad_value                             = 114.0F;
    spec.mean                                  = {0.0F, 0.0F, 0.0F};
    spec.stddev                                = {1.0F, 1.0F, 1.0F};
    spec.scale                                 = 1.0F / 255.0F;
    return spec;
}

void cudaCheck(cudaError_t code, const char *operation, YoloError failure = YoloError::InferenceFailed)
{
    if (code != cudaSuccess)
        fail(failure, operation, cudaGetErrorString(code));
}

void verifyTensor(const irt::TensorInfo &tensor, const char *name, const irt::Shape &shape, irt::TensorIOMode mode)
{
    if (tensor.name != name || tensor.mode != mode || tensor.desc.data_type != irt::TensorDataType::F32
        || tensor.desc.layout != irt::TensorLayout::Opaque || tensor.desc.shape != shape)
        fail(YoloError::TensorMismatch, name,
             "期望 float32/Opaque " + irt::model::dimsToString(shape) + "，实际 " + tensor.name + " "
                 + irt::model::dataTypeToString(tensor.desc.data_type) + " "
                 + irt::model::dimsToString(tensor.desc.shape));
}

std::vector<std::string> labels(const Values &values, const PathResolver &paths, int count)
{
    std::vector<std::string> result;
    const auto               path = text(values, yolo_ports::init::LABELS_PATH);
    if (!path.empty())
    {
        const auto resolution = paths.tryResolve(path);
        if (!resolution)
            fail(YoloError::LabelFileInvalid, path, resolution.issue().text);
        std::ifstream file(resolution.value(), std::ios::binary);
        if (!file)
            fail(YoloError::LabelFileInvalid, path, "无法打开文件");
        std::string line;
        while (std::getline(file, line))
        {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            if (result.empty() && line.starts_with("\xEF\xBB\xBF"))
                line.erase(0, 3);
            validateUtf8(line);
            result.push_back(std::move(line));
        }
        if (file.bad() || result.size() != static_cast<std::size_t>(count))
            fail(YoloError::LabelFileInvalid, path, "必须包含 num_classes 行 UTF-8 文本");
    }
    else
    {
        result.reserve(count);
        for (int i = 0; i < count; ++i) result.push_back(std::to_string(i));
    }
    return result;
}
} // namespace

struct YoloTool::State
{
    int                                         device = 0, width = 640, height = 640, classes = 80;
    std::vector<std::string>                    names;
    std::unique_ptr<irt::model::IModel>         model;
    std::unique_ptr<irt::ITensorRuntimeSession> session;
    cudaStream_t                                stream = nullptr;
    irt::model::DeviceBuffer                    deviceInput;
    irt::model::PinnedHostBuffer                hostInput;
    std::array<irt::model::DeviceBuffer, 3>     deviceOutputs;
    std::array<irt::model::PinnedHostBuffer, 3> hostOutputs;
    std::array<cv::Mat, 3>                      planes;
    std::vector<irt::BufferView>                views;
    std::vector<irt::TensorInfo>                descriptors;
    irt::PreprocessSpec                         preprocessing;

    ToolStatus close() noexcept
    {
        cudaError_t first    = cudaSuccess;
        const auto  remember = [&first](cudaError_t code)
        {
            if (first == cudaSuccess)
                first = code;
        };
        if (model || stream)
            remember(cudaSetDevice(device));
        if (stream)
            remember(cudaStreamSynchronize(stream));
        // CUDA work has finished before session/buffer/model destruction.
        session.reset();
        views.clear();
        for (auto &plane : planes) plane.release();
        for (auto &buffer : deviceOutputs) buffer.reset();
        for (auto &buffer : hostOutputs) buffer.reset();
        deviceInput.reset();
        hostInput.reset();
        model.reset();
        if (stream)
        {
            remember(cudaStreamDestroy(stream));
            stream = nullptr;
        }
        if (first != cudaSuccess)
        {
            try
            {
                return status(YoloError::ReleaseFailed, {}, cudaGetErrorString(first));
            }
            catch (...)
            {
                return status(YoloError::ReleaseFailed, "无法读取 CUDA 错误文本");
            }
        }
        return {};
    }

    ~State()
    {
        close();
    }
};

YoloTool::YoloTool(const ToolServices &services)
    : paths_(services.paths)
{
}

YoloTool::~YoloTool() = default;

ToolStatus YoloTool::initialize(const Values &parameters)
{
    const auto cleanup = release();
    if (!cleanup)
        return cleanup;
    try
    {
        auto       state      = std::make_unique<State>();
        const auto modelName  = text(parameters, yolo_ports::init::MODEL_NAME);
        const auto mode       = text(parameters, yolo_ports::init::LOAD_MODE);
        const auto precision  = text(parameters, yolo_ports::init::PRECISION);
        const auto modelPath  = text(parameters, yolo_ports::init::MODEL_PATH);
        const auto outputPath = text(parameters, yolo_ports::init::ENGINE_OUTPUT_PATH);
        if (modelName != "yolov8n")
            fail(YoloError::ModelUnsupported, "仅支持原生 yolov8n，实际模型：" + modelName);
        if (mode != "engine" && mode != "build_wts")
            fail(CommonToolError::InputOutOfRange, std::string(yolo_ports::init::LOAD_MODE) + " 只接受 engine/build_wts，实际值：" + mode);
        if (precision != "fp32" && precision != "fp16")
            fail(CommonToolError::InputOutOfRange, std::string(yolo_ports::init::PRECISION) + " 只接受 fp32/fp16，实际值：" + precision);
        if (modelPath.empty())
            fail(CommonToolError::InputMissing, yolo_ports::init::MODEL_PATH);
        state->device  = integer(parameters, yolo_ports::init::GPU_INDEX, 0);
        state->width   = integer(parameters, yolo_ports::init::INPUT_WIDTH, 32);
        state->height  = integer(parameters, yolo_ports::init::INPUT_HEIGHT, 32);
        state->classes = integer(parameters, yolo_ports::init::NUM_CLASSES, 1);
        if (state->width % 32 || state->height % 32 || state->classes > std::numeric_limits<int>::max() - 4)
            fail(CommonToolError::InputOutOfRange, "原生 yolov8n 输入尺寸必须为 32 的倍数；num_classes 必须满足 4+C 的整数范围");
        state->names      = labels(parameters, paths_, state->classes);
        int  deviceCount  = 0;
        auto deviceStatus = cudaGetDeviceCount(&deviceCount);
        if (deviceStatus != cudaSuccess)
            return status(YoloError::DeviceUnavailable, cudaGetErrorString(deviceStatus));
        if (deviceCount == 0 || state->device >= deviceCount)
            return status(YoloError::DeviceInvalid, "gpu_index=" + std::to_string(state->device));
        deviceStatus = cudaSetDevice(state->device);
        if (deviceStatus != cudaSuccess)
            return status(YoloError::DeviceInvalid, cudaGetErrorString(deviceStatus));
        if (!irt::model::isSupportedModel(modelName))
            return status(YoloError::ModelUnsupported, "InferRT 未注册 yolov8n");
        auto config = std::make_unique<irt::model::IModelConfig>();
        config->setRuntime(irt::model::ModelRuntime(irt::model::ModelRuntime::Backend::TensorRT,
                                                    irt::model::ModelRuntime::Device::GPU, state->device));
        config->setInputShape({1, 3, state->height, state->width});
        config->setNumClasses(state->classes);
        config->setDynamicBatch(false);
        config->setInputTensorNames({"input"});
        config->setOutputTensorNames({"output0", "output1", "output2"});
        config->setPrecision(precision == "fp16" ? irt::model::ModelPrecision::FP16 : irt::model::ModelPrecision::FP32);
        state->model = irt::model::CreateModel(modelName, std::move(config));
        if (!state->model)
            return status(YoloError::ModelLoadFailed, "InferRT CreateModel(yolov8n) 返回空");
        const auto modelResolution = paths_.tryResolve(modelPath);
        if (!modelResolution)
            fail(YoloError::ModelLoadFailed, modelPath, modelResolution.issue().text);
        const auto resolved = utf8Path(modelResolution.value());
        if (mode == "engine")
            state->model->load(resolved);
        else
        {
            state->model->build(resolved);
            if (!outputPath.empty())
            {
                const auto outputResolution = paths_.tryResolve(outputPath);
                if (!outputResolution)
                    fail(YoloError::ModelLoadFailed, outputPath, outputResolution.issue().text);
                state->model->save(utf8Path(outputResolution.value()));
            }
        }
        auto inputs  = state->model->inputs();
        auto outputs = state->model->outputs();
        if (inputs.size() != 1 || outputs.size() != 3)
            fail(YoloError::TensorMismatch, "原生 yolov8n 需要 1 个输入和 3 个 stride 输出");
        verifyTensor(inputs[0], "input", {1, 3, state->height, state->width}, irt::TensorIOMode::Input);
        state->descriptors.push_back(inputs[0]);
        for (std::size_t branch = 0; branch < strides.size(); ++branch)
        {
            const auto found = std::find_if(outputs.begin(), outputs.end(),
                                            [branch](const auto &value) { return value.name == outputNames[branch]; });
            if (found == outputs.end())
                fail(YoloError::TensorMismatch, std::string("缺少原生输出：") + outputNames[branch]);
            const auto grid
                = static_cast<std::int64_t>(state->width / strides[branch]) * (state->height / strides[branch]);
            verifyTensor(*found, outputNames[branch], {1, 4 + state->classes, grid}, irt::TensorIOMode::Output);
            state->descriptors.push_back(*found);
        }
        state->session = state->model->createSession();
        if (!state->session)
            fail(YoloError::SessionFailed, "InferRT 返回空执行会话");
        cudaCheck(cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags",
                  YoloError::SessionFailed);
        const auto inputElements = inputs[0].desc.shape.elementCount();
        state->deviceInput.resize(inputElements);
        state->hostInput.resize(inputElements);
        const auto pixels    = static_cast<std::size_t>(state->width) * state->height;
        auto      *inputData = static_cast<float *>(state->hostInput.data());
        for (std::size_t channel = 0; channel < 3; ++channel)
            state->planes[channel] = cv::Mat(state->height, state->width, CV_32FC1, inputData + channel * pixels);
        state->views.reserve(4);
        state->views.push_back(irt::BufferView::fromBytes(state->deviceInput.data(), state->deviceInput.sizeBytes(),
                                                          irt::MemoryKind::DEVICE, "input"));
        for (std::size_t branch = 0; branch < strides.size(); ++branch)
        {
            const auto elements = state->descriptors[branch + 1].desc.shape.elementCount();
            state->deviceOutputs[branch].resize(elements);
            state->hostOutputs[branch].resize(elements);
            state->views.push_back(irt::BufferView::fromBytes(state->deviceOutputs[branch].data(),
                                                              state->deviceOutputs[branch].sizeBytes(),
                                                              irt::MemoryKind::DEVICE, outputNames[branch]));
        }
        state->preprocessing = preprocessingSpec(state->width, state->height);
        state_               = std::move(state);
        return {};
    }
    catch (const visiontools::ToolFailure &error)
    {
        return visiontools::status(error);
    }
    catch (const std::exception &error)
    {
        return visiontools::status(YoloError::ModelLoadFailed, {}, error.what());
    }
    catch (...)
    {
        return visiontools::status(YoloError::ModelLoadFailed, "未知异常");
    }
}

ToolResult YoloTool::execute(const Values &inputs)
{
    return run(inputs, nullptr);
}

ToolResult YoloTool::executeWithTrace(const Values &inputs, std::vector<ImageTrace> &traces)
{
    traces.clear();
    return run(inputs, &traces);
}

ToolResult YoloTool::run(const Values &inputs, std::vector<ImageTrace> *traces)
{
    try
    {
        if (!state_)
            return missingResult(status(YoloError::ModelLoadFailed, "模型尚未初始化"));
        // A branch the condition tool did not select must not load a model or touch the device.
        if (const auto upstream = visiontools::upstreamStatus(inputs); upstream != 0)
            return missingResult({upstream, "上游未选中本分支，工具未执行"});
        const auto found = inputs.find(yolo_ports::in::IMAGES);
        if (found == inputs.end() || isMissing(found->second))
            fail(CommonToolError::InputMissing, yolo_ports::in::IMAGES);
        const auto *imagesPtr = std::get_if<DataRef<Image>>(&found->second);
        if (!imagesPtr)
            fail(CommonToolError::InputTypeMismatch, std::string(yolo_ports::in::IMAGES) + " 必须是图像");
        const auto               &images = *imagesPtr;
        const DetectionParameters parameters{requiredParameter<double>(inputs, yolo_ports::in::CONF_THRESHOLD),
                                             requiredParameter<double>(inputs, yolo_ports::in::NMS_THRESHOLD),
                                             requiredParameter<std::int64_t>(inputs, yolo_ports::in::MAX_DETECTIONS)};
        validateParameters(parameters);
        for (const auto &image : images.get())
        {
            if (image.empty() || image.depth() != CV_8U
                || (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
                fail(CommonToolError::UnsupportedImage, "images 必须包含非空 uint8 灰度/BGR/BGRA 图像");
        }
        auto &state = *state_;
        cudaCheck(cudaSetDevice(state.device), "cudaSetDevice");
        ROI                           rois;
        NestedVector<float, 2>        scores;
        NestedVector<std::int64_t, 2> classes;
        NestedVector<std::string, 2>  names;
        const auto                    batch = images.get().size();
        rois.reserve(batch);
        scores.reserve(batch);
        classes.reserve(batch);
        names.reserve(batch);
        if (traces)
            traces->reserve(batch);
        for (const auto &image : images.get())
        {
            cv::Mat        converted;
            const cv::Mat *bgr = &image;
            if (image.channels() != 3)
            {
                cv::cvtColor(*bgr, converted, image.channels() == 1 ? cv::COLOR_GRAY2BGR : cv::COLOR_BGRA2BGR);
                bgr = &converted;
            }
            const auto preprocessed = irt::model::ImageNetUtil::preprocessWithGeometry(*bgr, state.preprocessing);
            if (preprocessed.image.type() != CV_32FC3 || preprocessed.image.cols != state.width
                || preprocessed.image.rows != state.height)
                fail(YoloError::PreprocessFailed, "InferRT 返回了不兼容的图像");
            cv::split(preprocessed.image, state.planes.data());
            cudaCheck(cudaMemcpyAsync(state.deviceInput.data(), state.hostInput.data(), state.hostInput.sizeBytes(),
                                      cudaMemcpyHostToDevice, state.stream),
                      "YOLO H2D");
            state.model->executeSession(*state.session, state.views,
                                        {reinterpret_cast<std::uintptr_t>(state.stream), true});
            for (std::size_t branch = 0; branch < strides.size(); ++branch)
                cudaCheck(cudaMemcpyAsync(state.hostOutputs[branch].data(), state.deviceOutputs[branch].data(),
                                          state.hostOutputs[branch].sizeBytes(), cudaMemcpyDeviceToHost, state.stream),
                          "D2H 拷贝");
            cudaCheck(cudaStreamSynchronize(state.stream), "stream completion");
            const auto             &actual = preprocessed.geometry;
            const LetterboxGeometry geometry{actual.original_width, actual.original_height, state.width,
                                             state.height,          actual.resized_width,   actual.resized_height,
                                             actual.pad_left,       actual.pad_top,         actual.scale};
            std::array<HeadView, 3> heads;
            for (std::size_t branch = 0; branch < strides.size(); ++branch)
                heads[branch] = {
                    {static_cast<const float *>(state.hostOutputs[branch].data()), state.hostOutputs[branch].size()},
                    strides[branch],
                    state.width / strides[branch],
                    state.height / strides[branch],
                    state.classes
                };
            ImageTrace *trace = nullptr;
            if (traces)
            {
                traces->emplace_back();
                trace           = &traces->back();
                trace->geometry = geometry;
                for (std::size_t branch = 0; branch < strides.size(); ++branch)
                    trace->rawOutputs[branch].assign(heads[branch].values.begin(), heads[branch].values.end());
            }
            const auto detections = postprocess(heads, geometry, parameters, trace ? &trace->candidates : nullptr);
            if (trace)
                trace->detections = detections;
            std::vector<std::vector<double>> imageRois;
            std::vector<float>        imageScores;
            std::vector<std::int64_t> imageClasses;
            std::vector<std::string>  imageNames;
            const auto                count = detections.size();
            imageRois.reserve(count);
            imageScores.reserve(count);
            imageClasses.reserve(count);
            imageNames.reserve(count);
            for (const auto &detection : detections)
            {
                const auto &box = detection.box;
                imageRois.push_back({2, static_cast<double>(box.x1), static_cast<double>(box.y1),
                                     static_cast<double>(box.x2) - box.x1, static_cast<double>(box.y2) - box.y1, 0});
                imageScores.push_back(detection.score);
                imageClasses.push_back(detection.classId);
                imageNames.push_back(state.names[static_cast<std::size_t>(detection.classId)]);
            }
            rois.push_back(std::move(imageRois));
            scores.push_back(std::move(imageScores));
            classes.push_back(std::move(imageClasses));
            names.push_back(std::move(imageNames));
        }
        return toolResult({{yolo_ports::out::IMAGES, images},
                           {yolo_ports::out::ROIS, DataRef<ROI>(std::move(rois))},
                           {yolo_ports::out::SCORES, Array(std::move(scores))},
                           {yolo_ports::out::CLASSES, Array(std::move(classes))},
                           {yolo_ports::out::CLASS_NAMES, Array(std::move(names))}},
                          executionSuccess());
    }
    catch (const visiontools::ToolFailure &error)
    {
        if (state_ && state_->stream)
            cudaStreamSynchronize(state_->stream);
        if (traces)
            traces->clear();
        return missingResult(visiontools::status(error));
    }
    catch (const std::exception &error)
    {
        if (state_ && state_->stream)
            cudaStreamSynchronize(state_->stream);
        if (traces)
            traces->clear();
        return missingResult(visiontools::status(YoloError::InferenceFailed, {}, error.what()));
    }
    catch (...)
    {
        if (state_ && state_->stream)
            cudaStreamSynchronize(state_->stream);
        if (traces)
            traces->clear();
        return missingResult(visiontools::status(YoloError::InferenceFailed, "未知异常"));
    }
}

ToolStatus YoloTool::release()
{
    if (!state_)
        return {};
    auto status = state_->close();
    state_.reset();
    return status;
}

std::vector<TensorDescription> YoloTool::tensorDescriptions() const
{
    std::vector<TensorDescription> descriptions;
    if (state_)
        for (const auto &descriptor : state_->descriptors)
            descriptions.push_back({descriptor.name, irt::model::dataTypeToString(descriptor.desc.data_type), "Opaque",
                                    descriptor.desc.shape.dims});
    return descriptions;
}

GeometryProbe probePreprocessing()
{
    cv::Mat ramp(7, 13, CV_8UC3);
    for (int y = 0; y < ramp.rows; ++y)
        for (int x = 0; x < ramp.cols; ++x)
            ramp.at<cv::Vec3b>(y, x)
                = cv::Vec3b(static_cast<unsigned char>(x * 10), static_cast<unsigned char>(y * 20), 0);
    const auto    transformed = irt::model::ImageNetUtil::preprocessWithGeometry(ramp, preprocessingSpec(640, 640));
    const auto   &actual      = transformed.geometry;
    GeometryProbe result{
        {13, 7, 640, 640, actual.resized_width, actual.resized_height, actual.pad_left, actual.pad_top, actual.scale}
    };
    for (int y = 0; y < actual.resized_height; ++y)
    {
        const auto sourceY = std::clamp((y + 0.5) * 7.0 / actual.resized_height - 0.5, 0.0, 6.0);
        for (int x = 0; x < actual.resized_width; ++x)
        {
            const auto sourceX = std::clamp((x + 0.5) * 13.0 / actual.resized_width - 0.5, 0.0, 12.0);
            const auto pixel   = transformed.image.at<cv::Vec3f>(y + actual.pad_top, x + actual.pad_left);
            result.maximumPixelError
                = std::max({result.maximumPixelError, std::abs(pixel[0] * 255.0),
                            std::abs(pixel[1] * 255.0 - sourceY * 20), std::abs(pixel[2] * 255.0 - sourceX * 10)});
        }
    }
    const auto padding = transformed.image.at<cv::Vec3f>(0, 0);
    for (int channel = 0; channel < 3; ++channel)
        result.paddingPixelError = std::max(result.paddingPixelError, std::abs(padding[channel] * 255.0 - 114));
    return result;
}

} // namespace visionrt::infer
