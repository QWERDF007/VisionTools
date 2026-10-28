#pragma once

#include "common/errors/ToolError.h"

namespace visiontools::preprocess {

enum class ReadImageError : std::int64_t
{
    PathNotFound  = errorCode(domain::kReadImage, 1),
    NoImagesFound = errorCode(domain::kReadImage, 2),
    FileReadFailed = errorCode(domain::kReadImage, 3),
    DecodeFailed   = errorCode(domain::kReadImage, 4),
    FileTooLarge   = errorCode(domain::kReadImage, 5)
};

inline std::string_view message(ReadImageError code) noexcept
{
    switch (code)
    {
    case ReadImageError::PathNotFound:
        return "图像路径不存在";
    case ReadImageError::NoImagesFound:
        return "没有可用图像";
    case ReadImageError::FileReadFailed:
        return "图像文件读取失败";
    case ReadImageError::DecodeFailed:
        return "图像解码失败";
    case ReadImageError::FileTooLarge:
        return "图像文件超出大小限制";
    }
    return "图像读取失败";
}

} // namespace visiontools::preprocess
