#pragma once

#include <visionrt/contracts/Value.h>

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace visiontools::geometry {

// Row-major affine transform: p_current = A * p_reference (column vector).
class Alignment
{
public:
    std::array<double, 9> m{1, 0, 0, 0, 1, 0, 0, 0, 1};

    static Alignment                 identity() noexcept;
    static std::optional<Alignment>  fromRows(std::span<const double> rows) noexcept;

    bool   isAffine(double tolerance = 1e-9) const noexcept;
    bool   isIdentity(double tolerance = 1e-12) const noexcept;
    bool   invertible() const noexcept;
    double determinant() const noexcept;
    // Similarity requires L^T L = s^2 I with det(L) > 0 and s > 0.
    bool   isSimilarity(double relativeTolerance = 1e-6) const noexcept;
    double scale() const noexcept;
    // Returns std::nullopt when the linear part is singular.
    std::optional<Alignment> inverse() const noexcept;
    Alignment               operator*(const Alignment &other) const noexcept;

    void apply(double x, double y, double &outX, double &outY) const noexcept;
};

// Alignment port value: [] (per-image identity), one broadcast matrix, or one per image.
std::vector<Alignment> alignmentsFromValue(const visionrt::PortValue &value, std::size_t imageCount);

// Output value for Array(float64, rank 3) with one matrix per image.
visionrt::Array alignmentArray(const std::vector<Alignment> &alignments);
visionrt::Array alignmentArray(const Alignment &alignment);

// Rotation by clockwise angle (VisionRT convention: x right, y down, positive clockwise).
Alignment rotationAround(double centerX, double centerY, double angleDegrees) noexcept;

} // namespace visiontools::geometry
