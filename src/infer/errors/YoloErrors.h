#pragma once

#include "common/errors/ToolError.h"

namespace visionrt::infer {

// The inference tool fails at stages a caller can act on: no usable device, an unsupported or
// unloadable model, a tensor contract that does not match, or a failure inside the runtime.
enum class YoloError : std::int64_t
{
    // The five stages a caller acts on keep the published numbers: device, model identity, load,
    // tensor contract, execution.
    DeviceUnavailable = visiontools::errorCode(visiontools::domain::kInfer, 1),
    ModelUnsupported  = visiontools::errorCode(visiontools::domain::kInfer, 2),
    ModelLoadFailed   = visiontools::errorCode(visiontools::domain::kInfer, 3),
    TensorMismatch    = visiontools::errorCode(visiontools::domain::kInfer, 4),
    InferenceFailed   = visiontools::errorCode(visiontools::domain::kInfer, 5),
    // Second-level detail behind those stages.
    DeviceInvalid     = visiontools::errorCode(visiontools::domain::kInfer, 6),
    SessionFailed     = visiontools::errorCode(visiontools::domain::kInfer, 7),
    PreprocessFailed  = visiontools::errorCode(visiontools::domain::kInfer, 8),
    ReleaseFailed     = visiontools::errorCode(visiontools::domain::kInfer, 9),
    LabelFileInvalid  = visiontools::errorCode(visiontools::domain::kInfer, 10)
};

inline std::string_view message(YoloError code) noexcept
{
    switch (code)
    {
    case YoloError::DeviceUnavailable:
        return "GPU 不可用";
    case YoloError::DeviceInvalid:
        return "GPU 索引无效";
    case YoloError::ModelUnsupported:
        return "模型文件不受支持";
    case YoloError::ModelLoadFailed:
        return "模型加载失败";
    case YoloError::TensorMismatch:
        return "张量形状不匹配";
    case YoloError::SessionFailed:
        return "推理会话创建失败";
    case YoloError::PreprocessFailed:
        return "推理预处理失败";
    case YoloError::InferenceFailed:
        return "CUDA 推理失败";
    case YoloError::ReleaseFailed:
        return "推理资源释放失败";
    case YoloError::LabelFileInvalid:
        return "类别名称文件不合法";
    }
    return "推理失败";
}

} // namespace visionrt::infer
