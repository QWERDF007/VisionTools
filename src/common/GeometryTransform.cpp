#include "GeometryTransform.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace visiontools::geometry {
namespace {
constexpr double kPi = 3.14159265358979323846;

double element(const visionrt::PortValue &value, std::size_t index)
{
    const auto &array = std::get<visionrt::Array>(value);
    return array.get<visionrt::NestedVector<double, 3>>().at(index / 9).at((index % 9) / 3).at(index % 3);
}

std::optional<Alignment> decodeMatrix(const std::vector<std::vector<double>> &rows)
{
    if (rows.size() != 3)
        return std::nullopt;
    std::array<double, 9> values{};
    for (std::size_t row = 0; row < 3; ++row)
    {
        if (rows[row].size() != 3)
            return std::nullopt;
        for (std::size_t column = 0; column < 3; ++column)
            values[row * 3 + column] = rows[row][column];
    }
    Alignment alignment;
    alignment.m = values;
    return alignment;
}
} // namespace

Alignment Alignment::identity() noexcept
{
    return Alignment{};
}

std::optional<Alignment> Alignment::fromRows(std::span<const double> rows) noexcept
{
    if (rows.size() != 9)
        return std::nullopt;
    Alignment alignment;
    std::copy(rows.begin(), rows.end(), alignment.m.begin());
    return alignment;
}

bool Alignment::isAffine(double tolerance) const noexcept
{
    const auto finite = std::all_of(m.begin(), m.end(), [](double value) { return std::isfinite(value); });
    return finite && std::abs(m[6]) <= tolerance && std::abs(m[7]) <= tolerance && std::abs(m[8] - 1.0) <= tolerance;
}

bool Alignment::isIdentity(double tolerance) const noexcept
{
    static constexpr std::array<double, 9> identityValues{1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (std::size_t i = 0; i < m.size(); ++i)
        if (std::abs(m[i] - identityValues[i]) > tolerance)
            return false;
    return true;
}

double Alignment::determinant() const noexcept
{
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
         + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

bool Alignment::invertible() const noexcept
{
    const double linear = m[0] * m[4] - m[1] * m[3];
    return std::isfinite(linear) && std::abs(linear) > 1e-12;
}

bool Alignment::isSimilarity(double relativeTolerance) const noexcept
{
    if (!isAffine() || !invertible())
        return false;
    const double a = m[0], b = m[1], c = m[3], d = m[4];
    const double s2      = a * a + b * b;
    const double scaleValue = std::sqrt(s2);
    if (!(scaleValue > 0.0) || !std::isfinite(scaleValue))
        return false;
    const double tolerance = relativeTolerance * std::max(1.0, s2);
    if (std::abs((c * c + d * d) - s2) > tolerance)
        return false;
    if (std::abs(a * c + b * d) > tolerance)
        return false;
    return a * d - b * c > 0.0;
}

double Alignment::scale() const noexcept
{
    return std::sqrt(std::abs(m[0] * m[4] - m[1] * m[3]));
}

std::optional<Alignment> Alignment::inverse() const noexcept
{
    if (!isAffine() || !invertible())
        return std::nullopt;
    const double det    = m[0] * m[4] - m[1] * m[3];
    const double inv00  = m[4] / det;
    const double inv01  = -m[1] / det;
    const double inv10  = -m[3] / det;
    const double inv11  = m[0] / det;
    Alignment    result;
    result.m = {inv00, inv01, -(inv00 * m[2] + inv01 * m[5]), inv10, inv11, -(inv10 * m[2] + inv11 * m[5]), 0, 0, 1};
    return result;
}

Alignment Alignment::operator*(const Alignment &other) const noexcept
{
    Alignment result;
    for (std::size_t row = 0; row < 3; ++row)
        for (std::size_t column = 0; column < 3; ++column)
        {
            double sum = 0;
            for (std::size_t inner = 0; inner < 3; ++inner)
                sum += m[row * 3 + inner] * other.m[inner * 3 + column];
            result.m[row * 3 + column] = sum;
        }
    return result;
}

void Alignment::apply(double x, double y, double &outX, double &outY) const noexcept
{
    outX = m[0] * x + m[1] * y + m[2];
    outY = m[3] * x + m[4] * y + m[5];
}

std::vector<Alignment> alignmentsFromValue(const visionrt::PortValue &value, std::size_t imageCount)
{
    std::vector<Alignment> result;
    if (visionrt::isMissing(value))
    {
        result.assign(std::max<std::size_t>(1, imageCount), Alignment::identity());
        return result;
    }
    const auto &array = std::get<visionrt::Array>(value);
    if (array.type() != visionrt::TypeSpec{visionrt::Type::Array, visionrt::DType::Float64, 3})
        throw visionrt::ContractError("alignment 必须是 Array(float64, rank=3)");
    const auto &matrices = array.get<visionrt::NestedVector<double, 3>>();
    if (matrices.empty())
    {
        result.assign(std::max<std::size_t>(1, imageCount), Alignment::identity());
        return result;
    }
    result.reserve(matrices.size());
    for (const auto &rows : matrices)
    {
        auto alignment = decodeMatrix(rows);
        if (!alignment || !alignment->isAffine())
            throw visionrt::ContractError("alignment 每个元素必须是有限仿射矩阵，最后一行为 [0,0,1]");
        if (!alignment->invertible())
            throw visionrt::ContractError("alignment 线性部分必须可逆");
        result.push_back(*alignment);
    }
    if (result.size() == 1 && imageCount > 1)
        result.resize(imageCount, result.front());
    else if (imageCount != 0 && result.size() != imageCount)
        throw visionrt::ContractError("alignment 数量必须为 1 或与图像数量一致");
    return result;
}

visionrt::Array alignmentArray(const std::vector<Alignment> &alignments)
{
    visionrt::NestedVector<double, 3> rows;
    rows.reserve(alignments.size());
    for (const auto &alignment : alignments)
        rows.push_back({{alignment.m[0], alignment.m[1], alignment.m[2]},
                        {alignment.m[3], alignment.m[4], alignment.m[5]},
                        {alignment.m[6], alignment.m[7], alignment.m[8]}});
    return visionrt::Array(visionrt::DataRef<visionrt::NestedVector<double, 3>>(std::move(rows)));
}

visionrt::Array alignmentArray(const Alignment &alignment)
{
    return alignmentArray(std::vector<Alignment>{alignment});
}

Alignment rotationAround(double centerX, double centerY, double angleDegrees) noexcept
{
    const double radians = angleDegrees * kPi / 180.0;
    const double cos     = std::cos(radians);
    const double sin     = std::sin(radians);
    // Clockwise rotation in an x-right / y-down frame uses the standard matrix
    // with a positive angle, unlike OpenCV's counter-clockwise getRotationMatrix2D.
    Alignment result;
    result.m = {cos, -sin, centerX - cos * centerX + sin * centerY, sin, cos,
                centerY - sin * centerX - cos * centerY, 0, 0, 1};
    return result;
}
} // namespace visiontools::geometry
