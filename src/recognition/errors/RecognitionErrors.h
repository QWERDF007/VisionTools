#pragma once

#include "common/errors/ToolError.h"

namespace visiontools::recognition {

// The recognition tools fail for reasons the parameters alone cannot express: the training asset
// that defines the reference geometry or the colour classes is missing or unusable, or the image at
// hand does not contain the trained feature.
enum class RecognitionError : std::int64_t
{
    AssetPathMissing          = errorCode(domain::kRecognition, 1),
    AssetUnreadable           = errorCode(domain::kRecognition, 2),
    AssetFormatInvalid        = errorCode(domain::kRecognition, 3),
    ReferenceSegmentMissing   = errorCode(domain::kRecognition, 4),
    ReferenceSegmentsParallel = errorCode(domain::kRecognition, 5),
    EdgeNotFound              = errorCode(domain::kRecognition, 6),
    ColorSamplesMissing       = errorCode(domain::kRecognition, 7),
    ColorClassMissing         = errorCode(domain::kRecognition, 8),
    ColorSamplesTooSmall      = errorCode(domain::kRecognition, 9),
    UnsupportedColorImage     = errorCode(domain::kRecognition, 10),
    DetectionRegionMissing    = errorCode(domain::kRecognition, 11),
    DetectionRegionNotCircle      = errorCode(domain::kRecognition, 12),
    ColorModelUnreadable          = errorCode(domain::kRecognition, 13),
    ColorModelIncompatible        = errorCode(domain::kRecognition, 14),
    ColorModelPathMissing         = errorCode(domain::kRecognition, 15),
    StraightEdgeModelUnreadable   = errorCode(domain::kRecognition, 16),
    StraightEdgeModelIncompatible = errorCode(domain::kRecognition, 17),
    StraightEdgeModelPathMissing  = errorCode(domain::kRecognition, 18),
    StraightEdgeRegionsInvalid    = errorCode(domain::kRecognition, 19),
    MatchModelPathMissing         = errorCode(domain::kRecognition, 20),
    MatchModelUnreadable          = errorCode(domain::kRecognition, 21),
    MatchModelIncompatible        = errorCode(domain::kRecognition, 22),
    MatchTrainingInvalid          = errorCode(domain::kRecognition, 23),
    MatchNotFound                 = errorCode(domain::kRecognition, 24)
};

inline std::string_view message(RecognitionError code) noexcept
{
    switch (code)
    {
    case RecognitionError::AssetPathMissing:
        return "训练数据路径未配置";
    case RecognitionError::AssetUnreadable:
        return "训练数据无法读取";
    case RecognitionError::AssetFormatInvalid:
        return "训练数据格式不正确";
    case RecognitionError::ReferenceSegmentMissing:
        return "训练数据缺少两条非平行参考边";
    case RecognitionError::ReferenceSegmentsParallel:
        return "两条参考边平行，无法确定姿态";
    case RecognitionError::EdgeNotFound:
        return "未找到与参考边匹配的直边";
    case RecognitionError::ColorSamplesMissing:
        return "训练数据缺少颜色样本";
    case RecognitionError::ColorClassMissing:
        return "颜色训练样本缺少类别名";
    case RecognitionError::ColorSamplesTooSmall:
        return "颜色训练样本的有效像素不足";
    case RecognitionError::UnsupportedColorImage:
        return "颜色识别需要 8 位三通道图像";
    case RecognitionError::DetectionRegionMissing:
        return "检测区域未配置";
    case RecognitionError::DetectionRegionNotCircle:
        return "检测区域必须是圆";
    case RecognitionError::ColorModelUnreadable:
        return "颜色模型无法读取";
    case RecognitionError::ColorModelIncompatible:
        return "颜色模型格式不兼容或已损坏";
    case RecognitionError::ColorModelPathMissing:
        return "颜色模型路径未配置";
    case RecognitionError::StraightEdgeModelUnreadable:
        return "直边模型无法读取";
    case RecognitionError::StraightEdgeModelIncompatible:
        return "直边模型格式不兼容或已损坏";
    case RecognitionError::StraightEdgeModelPathMissing:
        return "直边模型路径未配置";
    case RecognitionError::StraightEdgeRegionsInvalid:
        return "直边训练需要两个有效的搜索矩形";
    case RecognitionError::MatchModelPathMissing:
        return "匹配模型路径未配置";
    case RecognitionError::MatchModelUnreadable:
        return "匹配模型无法读取";
    case RecognitionError::MatchModelIncompatible:
        return "匹配模型格式不兼容或已损坏";
    case RecognitionError::MatchTrainingInvalid:
        return "匹配训练数据无效";
    case RecognitionError::MatchNotFound:
        return "未找到满足分数阈值的匹配目标";
    }
    return "识别失败";
}

} // namespace visiontools::recognition
