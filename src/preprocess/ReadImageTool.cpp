#include "ReadImageTool.h"

#include "PortConstants.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <vector>

namespace visiontools::preprocess {
namespace {

namespace p = visiontools::ports::visionrt_image_read;

std::string utf8Path(const std::filesystem::path &path)
{
    const auto value = path.generic_u8string();
    return {reinterpret_cast<const char *>(value.data()), value.size()};
}

std::string toLower(std::string_view s)
{
    std::string res;
    res.reserve(s.size());
    for (char c : s) res.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return res;
}

bool hasExtension(const std::filesystem::path &path, const std::vector<std::string> &extensions)
{
    const auto ext = toLower(utf8Path(path.extension()));
    for (const auto &candidate : extensions)
    {
        if (ext == candidate || ext == ("." + candidate))
            return true;
    }
    return false;
}

} // namespace

ReadImageTool::ReadImageTool(const visionrt::ToolServices &services)
    : services_(services)
{
}

visionrt::ToolStatus ReadImageTool::initialize(const visionrt::Values &parameters)
{
    if (!parameters.empty())
    {
        return {1, "ReadImageTool accepts no init parameters"};
    }
    return {};
}

visionrt::ToolResult ReadImageTool::execute(const visionrt::Values &inputs)
{
    try
    {
        auto getStr = [&](const std::string &key, const std::string &defVal) -> std::string
        {
            auto it = inputs.find(key);
            if (it == inputs.end() || visionrt::isMissing(it->second))
                return defVal;
            if (auto *u = std::get_if<visionrt::Utf8>(&it->second))
                return u->str();
            throw visionrt::ContractError("Expected string for input: " + key);
        };

        auto getInt = [&](const std::string &key, std::int64_t defVal) -> std::int64_t
        {
            auto it = inputs.find(key);
            if (it == inputs.end() || visionrt::isMissing(it->second))
                return defVal;
            if (auto *v = std::get_if<std::int64_t>(&it->second))
                return *v;
            throw visionrt::ContractError("Expected integer for input: " + key);
        };

        const auto mode    = getStr(p::in::MODE, "file");
        const auto color   = getStr(p::in::COLOR, "bgr");
        const auto rawPath = getStr(p::in::PATH, "");
        const auto limit   = getInt(p::in::LIMIT, 0);

        int flags = cv::IMREAD_COLOR;
        if (color == "unchanged")
            flags = cv::IMREAD_UNCHANGED;
        else if (color == "bgr")
            flags = cv::IMREAD_COLOR;
        else if (color == "gray")
            flags = cv::IMREAD_GRAYSCALE;
        else
            throw visionrt::ContractError("Invalid color mode: " + color);

        if (rawPath.empty())
        {
            throw visionrt::ContractError("Image path is empty");
        }

        const auto                         resolved = services_.paths.resolve(rawPath);
        std::vector<std::filesystem::path> files;

        if (mode == "file")
        {
            if (!std::filesystem::exists(resolved) || !std::filesystem::is_regular_file(resolved))
            {
                throw visionrt::ContractError("File does not exist: " + utf8Path(resolved));
            }
            files.push_back(resolved);
        }
        else if (mode == "directory")
        {
            if (!std::filesystem::exists(resolved) || !std::filesystem::is_directory(resolved))
            {
                throw visionrt::ContractError("Directory does not exist: " + utf8Path(resolved));
            }
            const std::vector<std::string>     validExtensions = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"};
            std::vector<std::filesystem::path> candidates;
            for (const auto &entry : std::filesystem::directory_iterator(resolved))
            {
                if (entry.is_regular_file() && hasExtension(entry.path(), validExtensions))
                {
                    candidates.push_back(entry.path());
                }
            }
            std::sort(candidates.begin(), candidates.end(),
                      [](const auto &a, const auto &b) { return a.filename().u8string() < b.filename().u8string(); });
            for (const auto &c : candidates)
            {
                files.push_back(c);
                if (limit > 0 && static_cast<std::int64_t>(files.size()) >= limit)
                    break;
            }
        }
        else
        {
            throw visionrt::ContractError("Invalid mode: " + mode);
        }

        std::vector<visionrt::OwnedImage> images;
        std::vector<std::string>          pathStrings;
        images.reserve(files.size());
        pathStrings.reserve(files.size());

        for (const auto &f : files)
        {
            std::ifstream stream(f, std::ios::binary | std::ios::ate);
            if (!stream.is_open())
            {
                throw visionrt::ContractError("Cannot open image file: " + utf8Path(f));
            }
            const auto fileSize = stream.tellg();
            if (fileSize <= 0)
            {
                throw visionrt::ContractError("Empty image file: " + utf8Path(f));
            }
            std::vector<char> buffer(static_cast<std::size_t>(fileSize));
            stream.seekg(0);
            stream.read(buffer.data(), fileSize);

            cv::Mat decoded;
            try
            {
                const cv::Mat rawData(1, static_cast<int>(buffer.size()), CV_8UC1, buffer.data());
                decoded = cv::imdecode(rawData, flags);
            }
            catch (const cv::Exception &err)
            {
                throw visionrt::ContractError("Cannot decode image: " + utf8Path(f) + ": " + err.what());
            }
            if (decoded.empty())
            {
                throw visionrt::ContractError("Cannot decode image: " + utf8Path(f));
            }

            images.push_back(visionrt::OwnedImage::take(std::move(decoded)));
            pathStrings.push_back(utf8Path(f));
        }

        const auto       count = static_cast<std::int64_t>(pathStrings.size());
        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, visionrt::ImageBatch(std::move(images)));
        outputs.emplace(p::out::PATHS,
                        visionrt::StringArray(1, visionrt::Layout::Dense,
                                              {visionrt::StringBuffer({count}, std::move(pathStrings))}));

        return {{}, std::move(outputs)};
    }
    catch (const std::exception &e)
    {
        visionrt::Values outputs;
        outputs.emplace(p::out::IMAGES, visionrt::Missing{});
        outputs.emplace(p::out::PATHS, visionrt::Missing{});
        return {
            {1, e.what()},
            std::move(outputs)
        };
    }
}

visionrt::ToolStatus ReadImageTool::release()
{
    return {};
}

} // namespace visiontools::preprocess
