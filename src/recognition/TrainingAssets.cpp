#include "TrainingAssets.h"
#include <visionrt/contracts/ProtocolEnums.h>

#include "common/ProductPaths.h"
#include "common/errors/ToolError.h"

#include "errors/RecognitionErrors.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <string>
#include <string_view>

namespace visiontools::recognition {
namespace {
using Json = nlohmann::json;


std::string utf8(const std::filesystem::path &path)
{
    const auto text = path.u8string();
    return {reinterpret_cast<const char *>(text.data()), text.size()};
}

std::string text(const Json &value, std::string_view context)
{
    if (!value.is_string())
        fail(RecognitionError::AssetFormatInvalid, context, "需要字符串");
    return value.get<std::string>();
}

double number(const Json &value, std::string_view context)
{
    if (!value.is_number())
        fail(RecognitionError::AssetFormatInvalid, context, "需要数值");
    const double result = value.get<double>();
    if (!std::isfinite(result))
        fail(RecognitionError::AssetFormatInvalid, context, "需要有限数值");
    return result;
}

const Json &field(const Json &object, const char *key, std::string_view context)
{
    const auto found = object.find(key);
    if (found == object.end())
        fail(RecognitionError::AssetFormatInvalid, context, std::string("缺少字段 ") + key);
    return *found;
}

cv::Point2d point(const Json &value, std::string_view context)
{
    if (!value.is_array() || value.size() != 2)
        fail(RecognitionError::AssetFormatInvalid, context, "需要两个数值的坐标");
    return {number(value[0], context), number(value[1], context)};
}

SearchRectangle searchRectangle(const Json &value, std::string_view context)
{
    if (!value.is_array() || value.size() != 5)
        fail(RecognitionError::AssetFormatInvalid, context, "需要 [中心X, 中心Y, 宽, 高, 角度] 五个数值");
    SearchRectangle rectangle;
    rectangle.center = {number(value[0], context), number(value[1], context)};
    rectangle.width  = number(value[2], context);
    rectangle.height = number(value[3], context);
    rectangle.angle  = number(value[4], context);
    if (!(rectangle.width > 0.0) || !(rectangle.height > 0.0))
        fail(RecognitionError::AssetFormatInvalid, context, "搜索框宽高必须为正数");
    return rectangle;
}

ColorRegion colorRegion(const Json &value, std::string_view context)
{
    const auto rectangle = searchRectangle(value, context);
    return {rectangle.center, rectangle.width, rectangle.height, rectangle.angle};
}

Json readDocument(const std::filesystem::path &asset)
{
    std::ifstream file(asset, std::ios::binary);
    if (!file)
        fail(RecognitionError::AssetUnreadable, utf8(asset), "无法打开训练数据文件");
    Json document;
    try
    {
        document = Json::parse(file);
    }
    catch (const Json::exception &error)
    {
        fail(RecognitionError::AssetFormatInvalid, utf8(asset), error.what());
    }
    if (!document.is_object())
        fail(RecognitionError::AssetFormatInvalid, utf8(asset), "顶层必须是对象");
    return document;
}

std::filesystem::path resolveImage(const std::string &path, std::string_view context)
{
    if (path.empty())
        fail(RecognitionError::AssetFormatInvalid, context, "训练图路径为空");
    return common::resolveResourcePath(path);
}

void requireSchema(const Json &document, visionrt::TrainingKind kind, std::string_view context)
{
    const auto &version = field(document, "schema_version", context);
    if (!version.is_number_integer() || version.get<int>() != visionrt::kTrainingAssetVersion)
        fail(RecognitionError::AssetFormatInvalid, context, "schema_version 必须为 2");
    if (field(document, "kind", context) != static_cast<int>(kind))
        fail(RecognitionError::AssetFormatInvalid, context, "训练类型不匹配");
}
} // namespace

StraightEdgeTraining loadStraightEdgeTraining(const std::filesystem::path &asset)
{
    const auto context = utf8(asset);
    const auto document = readDocument(asset);
    requireSchema(document, visionrt::TrainingKind::StraightEdges, context);

    StraightEdgeTraining training;
    training.referenceImage
        = resolveImage(text(field(document, "reference_image", context), context), context);

    const auto &segments = field(document, "segments", context);
    if (!segments.is_array() || segments.size() < 2)
        fail(RecognitionError::ReferenceSegmentMissing, context, "至少需要两条参考边");
    for (std::size_t index = 0; index < segments.size(); ++index)
    {
        const auto &item    = segments[index];
        const auto  segment = std::string("segments[") + std::to_string(index) + "]";
        if (!item.is_object())
            fail(RecognitionError::AssetFormatInvalid, context, segment + " 必须是对象");
        ReferenceSegment result;
        if (const auto name = item.find("name"); name != item.end())
            result.name = text(*name, context);
        result.from   = point(field(item, "from", context), context);
        result.to     = point(field(item, "to", context), context);
        result.search = searchRectangle(field(item, "search", context), context);
        if (!(cv::norm(result.to - result.from) > 1.0))
            fail(RecognitionError::AssetFormatInvalid, context, segment + " 的参考边长度不足 1 像素");
        training.segments.push_back(std::move(result));
    }
    if (training.segments.size() != 2)
        fail(RecognitionError::ReferenceSegmentMissing, context, "当前实现需要恰好两条参考边");
    return training;
}

ColorTraining loadColorTraining(const std::filesystem::path &asset)
{
    const auto context = utf8(asset);
    const auto document = readDocument(asset);
    requireSchema(document, visionrt::TrainingKind::ColorSamples, context);

    const auto &records = field(document, "records", context);
    if (!records.is_array() || records.empty())
        fail(RecognitionError::ColorSamplesMissing, context, "records 为空");

    ColorTraining training;
    for (std::size_t index = 0; index < records.size(); ++index)
    {
        const auto &item   = records[index];
        const auto  record = std::string("records[") + std::to_string(index) + "]";
        if (!item.is_object())
            fail(RecognitionError::AssetFormatInvalid, context, record + " 必须是对象");
        ColorSample sample;
        sample.id    = text(field(item, "id", context), context);
        sample.label = text(field(item, "label", context), context);
        if (sample.label.empty())
            fail(RecognitionError::ColorClassMissing, context, record + " 的颜色名称为空");
        sample.image = resolveImage(text(field(item, "image", context), context), context);
        const auto &regions = field(item, "regions", context);
        if (!regions.is_array())
            fail(RecognitionError::AssetFormatInvalid, context, record + " 的 regions 必须是数组");
        for (std::size_t region = 0; region < regions.size(); ++region)
            sample.regions.push_back(colorRegion(regions[region], context));
        if (sample.regions.empty())
            fail(RecognitionError::ColorSamplesMissing, context, record + " 没有训练区域");
        training.samples.push_back(std::move(sample));
    }
    return training;
}

} // namespace visiontools::recognition
