#include "CalibrationPlate.h"

#include "errors/MetrologyErrors.h"
#include "common/PlanarCalibration.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <fstream>
#include <limits>
#include <map>
#include <numbers>
#include <set>
#include <sstream>

namespace visiontools::metrology {
namespace {
using Grid = std::pair<int, int>;
constexpr std::array<Grid, 6> kHexNeighbors{{{1,0}, {0,1}, {-1,1}, {-1,0}, {0,-1}, {1,-1}}};
constexpr double kFinderHoleAreaRatio = 0.02;
constexpr double kNeighborTolerance = 0.25;
constexpr double kGridTolerance = 1e-5;
constexpr std::size_t kPrimaryFinderHoles = 6;
constexpr std::size_t kMinimumCalibrationMarks = 12;

struct Blob
{
    cv::Point2d center;
    double radius;
    bool hole;
};

double median(std::vector<double> values)
{
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

std::vector<Blob> calibrationBlobs(const cv::Mat &gray, bool light)
{
    cv::Mat mask;
    cv::threshold(gray, mask, 0, 255, (light ? cv::THRESH_BINARY : cv::THRESH_BINARY_INV) | cv::THRESH_OTSU);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
    std::vector<Blob> blobs;
    for (std::size_t i = 0; i < contours.size(); ++i)
    {
        if (hierarchy[i][3] >= 0 || contours[i].size() < 5) continue;
        const auto bounds = cv::boundingRect(contours[i]);
        if (bounds.x <= 0 || bounds.y <= 0 || bounds.br().x >= gray.cols || bounds.br().y >= gray.rows) continue;
        const auto ellipse = cv::fitEllipse(contours[i]);
        const double area = cv::contourArea(contours[i]);
        const auto small = std::min(ellipse.size.width, ellipse.size.height), large = std::max(ellipse.size.width, ellipse.size.height);
        const double ellipseArea = std::numbers::pi * ellipse.size.area() / 4;
        if (!(small > 0) || small / large < 0.4 || area < ellipseArea * 0.6 || area > ellipseArea * 1.2) continue;
        bool hole = false;
        for (int child = hierarchy[i][2]; child >= 0; child = hierarchy[child][0])
            if (cv::contourArea(contours[child]) > area * kFinderHoleAreaRatio) hole = true;
        blobs.push_back({ellipse.center, (small + large) / 4, hole});
    }
    if (blobs.size() < kPrimaryFinderHoles + 1) fail(MetrologyError::CalibrationFinderMissing);
    std::vector<double> radii;
    for (const auto &blob : blobs) radii.push_back(blob.radius);
    const auto typical = median(std::move(radii));
    std::erase_if(blobs, [&](const auto &blob) { return blob.radius < typical * 0.5 || blob.radius > typical * 2; });
    return blobs;
}

cv::Point2d homographyPoint(const cv::Matx33d &h, cv::Point2d point)
{
    const auto p = h * cv::Vec3d(point.x, point.y, 1);
    return {p[0] / p[2], p[1] / p[2]};
}
}

CalibrationPlate CalibrationPlate::read(const std::filesystem::path &file)
{
    std::ifstream stream(file);
    if (!stream) fail(MetrologyError::CalibrationPlateInvalid, "无法读取描述文件");
    std::string line;
    std::getline(stream, line);
    if (line.find("Plate Description Version 3") == std::string::npos)
        fail(MetrologyError::CalibrationPlateInvalid, "需要 Version 3 六角网格描述");
    CalibrationPlate result;
    int finderCount = 0;
    bool polarity = false;
    const double millimetersPerMeter = geometry::millimetersPerUnit("m");
    while (std::getline(stream, line))
    {
        line = line.substr(0, line.find('#'));
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;
        std::istringstream row(line.substr(first));
        if (std::isalpha(static_cast<unsigned char>(line[first])))
        {
            char key;
            row >> key;
            if (key == 'r') row >> result.rows;
            else if (key == 'c') row >> result.columns;
            else if (key == 'f') row >> finderCount;
            else if (key == 'p')
            {
                std::string value;
                row >> value;
                if (value != "light" && value != "dark") fail(MetrologyError::CalibrationPlateInvalid, "标记极性无效");
                result.lightMarks = value == "light";
                polarity = true;
            }
            else if (key == 'o' || key == 'z') continue;
            else fail(MetrologyError::CalibrationPlateInvalid, "描述文件指令无效");
            if (!row) fail(MetrologyError::CalibrationPlateInvalid, "描述字段不完整");
        }
        else if (static_cast<int>(result.finders.size()) < finderCount)
        {
            cv::Point2i finder;
            if (!(row >> finder.x >> finder.y)) fail(MetrologyError::CalibrationPlateInvalid, "定位标记坐标不完整");
            result.finders.push_back(finder);
        }
        else
        {
            cv::Point2d point;
            double radius;
            if (!(row >> point.x >> point.y >> radius) || !std::isfinite(point.x) || !std::isfinite(point.y) || !(radius > 0))
                fail(MetrologyError::CalibrationPlateInvalid, "标定点坐标或半径无效");
            result.millimeters.push_back(point * millimetersPerMeter);
        }
    }
    if (result.rows < 2 || result.columns < 2 || !polarity || finderCount < 1 || result.finders.size() != static_cast<std::size_t>(finderCount)
        || result.millimeters.size() != static_cast<std::size_t>(result.rows) * result.columns)
        fail(MetrologyError::CalibrationPlateInvalid, "网格尺寸、定位标记或标定点数量不匹配");
    for (const auto &finder : result.finders)
        if (finder.x < 0 || finder.x >= result.columns || finder.y < 0 || finder.y >= result.rows)
            fail(MetrologyError::CalibrationPlateInvalid, "定位标记索引超出网格");
    result.origin = result.millimeters[result.finders.front().y * result.columns + result.finders.front().x];
    result.pitch = cv::norm(result.millimeters[1] - result.millimeters[0]);
    if (!(result.pitch > 0)) fail(MetrologyError::CalibrationPlateInvalid, "点间距无效");
    for (auto &point : result.millimeters)
    {
        point -= result.origin;
        point.y = -point.y;
    }
    result.origin = {};
    return result;
}

std::vector<CalibrationMark> observeCalibrationPlate(const cv::Mat &gray, const CalibrationPlate &plate)
{
    if (gray.empty() || gray.type() != CV_8UC1) fail(CommonToolError::UnsupportedImage, "标定需要八位灰度图或彩色图");
    const auto blobs = calibrationBlobs(gray, plate.lightMarks);
    std::vector<double> nearest;
    for (std::size_t i = 0; i < blobs.size(); ++i)
    {
        double distance = std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < blobs.size(); ++j)
            if (j != i) distance = std::min(distance, cv::norm(blobs[i].center - blobs[j].center));
        nearest.push_back(distance);
    }
    const double pitch = median(std::move(nearest));
    std::optional<std::size_t> primary;
    for (std::size_t i = 0; i < blobs.size(); ++i)
    {
        if (!blobs[i].hole) continue;
        std::size_t holes = 0;
        for (std::size_t j = 0; j < blobs.size(); ++j)
            if (i != j && blobs[j].hole && std::abs(cv::norm(blobs[i].center - blobs[j].center) / pitch - 1) < kNeighborTolerance) ++holes;
        if (holes == kPrimaryFinderHoles - 1)
        {
            if (primary) fail(MetrologyError::CalibrationFinderMissing, "主定位标记不唯一");
            primary = i;
        }
    }
    if (!primary) fail(MetrologyError::CalibrationFinderMissing, "需要完整的六孔主定位标记");
    const auto origin = blobs[*primary].center;
    std::vector<std::size_t> neighbors;
    for (std::size_t i = 0; i < blobs.size(); ++i)
        if (i != *primary && std::abs(cv::norm(blobs[i].center - origin) / pitch - 1) < kNeighborTolerance) neighbors.push_back(i);
    if (neighbors.size() != kHexNeighbors.size()) fail(MetrologyError::CalibrationFinderMissing, "主定位标记邻域不完整");
    const auto plain = std::ranges::find_if(neighbors, [&](auto i) { return !blobs[i].hole; });
    if (plain == neighbors.end()) fail(MetrologyError::CalibrationFinderMissing, "缺少主定位标记方向点");
    const auto xAxis = blobs[*plain].center - origin;
    cv::Point2d diagonal;
    double angularError = std::numeric_limits<double>::infinity();
    for (auto i : neighbors)
    {
        const auto delta = blobs[i].center - origin;
        const auto angle = std::atan2(xAxis.x * delta.y - xAxis.y * delta.x, xAxis.dot(delta));
        if (angle > 0 && std::abs(angle - std::numbers::pi / 3) < angularError)
        {
            diagonal = delta;
            angularError = std::abs(angle - std::numbers::pi / 3);
        }
    }
    const cv::Matx22d basis(xAxis.x, diagonal.x, xAxis.y, diagonal.y);
    if (cv::determinant(basis) == 0) fail(MetrologyError::CalibrationFinderMissing, "定位轴退化");
    const auto inverseBasis = basis.inv();
    std::map<Grid, cv::Point2d> physical;
    const auto rowPitch = plate.pitch * std::sqrt(3.0) / 2;
    for (const auto &point : plate.millimeters)
    {
        // Descriptor rows advance opposite to world positive Y.
        const auto r = -point.y / rowPitch, q = point.x / plate.pitch - r / 2;
        if (std::abs(q - std::round(q)) > kGridTolerance || std::abs(r - std::round(r)) > kGridTolerance)
            fail(MetrologyError::CalibrationPlateInvalid, "需要均匀六角网格");
        physical.emplace(Grid{cvRound(q), cvRound(r)}, point);
    }
    std::map<Grid, std::size_t> matched{{{0,0}, *primary}};
    std::set<std::size_t> used{*primary};
    for (auto i : neighbors)
    {
        const auto local = inverseBasis * cv::Vec2d(blobs[i].center.x - origin.x, blobs[i].center.y - origin.y);
        const Grid key{cvRound(local[0]), cvRound(local[1])};
        if (!physical.contains(key) || !matched.emplace(key, i).second) fail(MetrologyError::CalibrationFinderMissing, "定位邻域与描述不匹配");
        used.insert(i);
    }
    const auto predictLocal = [&](const Grid &key) -> std::optional<cv::Point2d>
    {
        cv::Point2d sum(0.0, 0.0);
        int count = 0;
        // Parallelogram stencils: P(k - d1) + P(k - d2) - P(k - d1 - d2)
        for (std::size_t a = 0; a < kHexNeighbors.size(); ++a)
        {
            const Grid n1{key.first - kHexNeighbors[a].first, key.second - kHexNeighbors[a].second};
            const auto it1 = matched.find(n1);
            if (it1 == matched.end()) continue;
            for (std::size_t b = a + 1; b < kHexNeighbors.size(); ++b)
            {
                const Grid n2{key.first - kHexNeighbors[b].first, key.second - kHexNeighbors[b].second};
                const auto it2 = matched.find(n2);
                if (it2 == matched.end()) continue;
                const Grid n12{n1.first - kHexNeighbors[b].first, n1.second - kHexNeighbors[b].second};
                const auto it12 = matched.find(n12);
                if (it12 == matched.end()) continue;
                sum += blobs[it1->second].center + blobs[it2->second].center - blobs[it12->second].center;
                ++count;
            }
        }
        if (count > 0) return sum * (1.0 / count);
        // Collinear stencils: 2 * P(k - d) - P(k - 2*d)
        for (const auto &[dq, dr] : kHexNeighbors)
        {
            const Grid n1{key.first - dq, key.second - dr};
            const Grid n2{key.first - 2 * dq, key.second - 2 * dr};
            const auto it1 = matched.find(n1);
            const auto it2 = matched.find(n2);
            if (it1 == matched.end() || it2 == matched.end()) continue;
            sum += blobs[it1->second].center * 2.0 - blobs[it2->second].center;
            ++count;
        }
        if (count > 0) return sum * (1.0 / count);
        return std::nullopt;
    };
    const auto fitHomography = [&]
    {
        std::vector<cv::Point2d> object, image;
        for (const auto &[key, index] : matched) { object.push_back(physical.at(key)); image.push_back(blobs[index].center); }
        const auto h = cv::findHomography(object, image);
        if (h.empty()) fail(MetrologyError::CalibrationFailed, "网格单应性不可解");
        return cv::Matx33d(h);
    };
    auto homography = fitHomography();
    for (;;)
    {
        std::set<Grid> frontier;
        for (const auto &[key, index] : matched)
            for (const auto &[q, r] : kHexNeighbors)
            {
                Grid next{key.first + q, key.second + r};
                if (physical.contains(next) && !matched.contains(next)) frontier.insert(next);
            }
        bool added = false;
        for (const auto &key : frontier)
        {
            const auto local = predictLocal(key);
            const auto predicted = local ? *local : homographyPoint(homography, physical.at(key));
            double localPitch = pitch;
            if (local)
            {
                double stepSum = 0.0;
                int stepCount = 0;
                for (const auto &[dq, dr] : kHexNeighbors)
                {
                    const auto it = matched.find({key.first - dq, key.second - dr});
                    if (it != matched.end())
                    {
                        stepSum += cv::norm(*local - blobs[it->second].center);
                        ++stepCount;
                    }
                }
                if (stepCount > 0) localPitch = stepSum / stepCount;
            }
            auto best = blobs.size();
            double distance = localPitch * kNeighborTolerance;
            for (std::size_t i = 0; i < blobs.size(); ++i)
                if (!used.contains(i) && cv::norm(blobs[i].center - predicted) < distance)
                { distance = cv::norm(blobs[i].center - predicted); best = i; }
            if (best != blobs.size()) { matched.emplace(key, best); used.insert(best); added = true; }
        }
        if (!added) break;
    }
    if (matched.size() < kMinimumCalibrationMarks) fail(MetrologyError::InsufficientPoints, "标定点不足");
    std::vector<CalibrationMark> result;
    for (const auto &[key, index] : matched) result.push_back({blobs[index].center, physical.at(key), blobs[index].radius});
    return result;
}
}
