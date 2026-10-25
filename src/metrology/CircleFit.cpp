#include "CircleFit.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace visiontools::metrology {
namespace {
constexpr double kPi          = 3.14159265358979323846;
constexpr double kMinScale    = 0.05; // pixel floor for Huber and inlier thresholds
constexpr double kHuberTurn   = 1.345;
constexpr double kInlierSigmas = 2.5;
constexpr double kConvergence  = 1e-6;

double median(std::vector<double> values)
{
    if (values.empty())
        return 0;
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    double result = *middle;
    if (values.size() % 2 == 0)
    {
        const auto lower = std::max_element(values.begin(), middle);
        result           = (*lower + result) / 2.0;
    }
    return result;
}

double mad(const std::vector<double> &residuals)
{
    if (residuals.empty())
        return 0;
    std::vector<double> center(residuals.size());
    const double        mean = [&]
    {
        double sum = 0;
        for (double value : residuals)
            sum += value;
        return sum / static_cast<double>(residuals.size());
    }();
    std::transform(residuals.begin(), residuals.end(), center.begin(),
                   [mean](double value) { return std::abs(value - mean); });
    return median(std::move(center));
}

struct Parameters
{
    double cx = 0, cy = 0, r = 0;
};

// One weighted Gauss-Newton step in normalised coordinates.
bool refine(const std::vector<cv::Point2d> &points, const std::vector<double> &weights, Parameters &parameters,
            double centerX, double centerY, double scale)
{
    cv::Matx33d normal = cv::Matx33d::zeros();
    cv::Vec3d   rhs    = cv::Vec3d(0, 0, 0);
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        const double x = (points[index].x - centerX) / scale;
        const double y = (points[index].y - centerY) / scale;
        const double dx = x - parameters.cx, dy = y - parameters.cy;
        const double distance = std::sqrt(dx * dx + dy * dy);
        if (!(distance > 1e-12))
            return false;
        const cv::Vec3d jacobian(-dx / distance, -dy / distance, -1.0);
        const double    residual = distance - parameters.r;
        const double    weight   = weights[index];
        normal += weight * (jacobian * jacobian.t());
        rhs += weight * residual * jacobian;
    }
    const cv::Vec3d delta = -normal.inv(cv::DECOMP_SVD) * rhs;
    if (!std::isfinite(delta[0]) || !std::isfinite(delta[1]) || !std::isfinite(delta[2]))
        return false;
    parameters.cx += delta[0];
    parameters.cy += delta[1];
    parameters.r += delta[2];
    return std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]) <= kConvergence;
}
} // namespace

std::optional<CircleFitResult> fitCircle(const std::vector<cv::Point2d> &points, const CircleFitOptions &options,
                                         std::string *reason)
{
    const auto fail = [reason](std::string message) -> std::optional<CircleFitResult>
    {
        if (reason)
            *reason = std::move(message);
        return std::nullopt;
    };
    if (points.size() < 3)
        return fail("有效边缘点少于点拟合下限");
    double centerX = 0, centerY = 0;
    for (const auto &point : points)
    {
        centerX += point.x;
        centerY += point.y;
    }
    centerX /= static_cast<double>(points.size());
    centerY /= static_cast<double>(points.size());
    double spread = 0;
    for (const auto &point : points)
        spread = std::max(spread, std::hypot(point.x - centerX, point.y - centerY));
    if (!(spread > 1e-6))
        return fail("边缘点退化，无法拟合圆");
    const double scale = spread;

    // Algebraic initialisation: x^2 + y^2 + D x + E y + F = 0.
    cv::Mat design(static_cast<int>(points.size()), 3, CV_64F);
    cv::Mat values(static_cast<int>(points.size()), 1, CV_64F);
    for (int row = 0; row < design.rows; ++row)
    {
        const double x = (points[static_cast<std::size_t>(row)].x - centerX) / scale;
        const double y = (points[static_cast<std::size_t>(row)].y - centerY) / scale;
        design.at<double>(row, 0) = x;
        design.at<double>(row, 1) = y;
        design.at<double>(row, 2) = 1.0;
        values.at<double>(row, 0) = -(x * x + y * y);
    }
    cv::Mat solution;
    cv::Mat singularValues;
    if (!cv::solve(design, values, solution, cv::DECOMP_SVD))
        return fail("圆拟合初值求解失败");
    cv::SVD decomposition(design, cv::SVD::NO_UV);
    singularValues = decomposition.w;
    const double largest  = singularValues.at<double>(0);
    const double smallest = singularValues.at<double>(2);
    if (!(smallest > 0.0) || !std::isfinite(largest) || largest / smallest > 1e12)
        return fail("圆拟合线性系统病态，边缘点接近共线");
    const double d = solution.at<double>(0), e = solution.at<double>(1), f = solution.at<double>(2);
    const double radiusSquared = d * d / 4.0 + e * e / 4.0 - f;
    if (!(radiusSquared > 0.0))
        return fail("圆拟合得到非正半径");
    Parameters parameters;
    parameters.cx = -d / 2.0;
    parameters.cy = -e / 2.0;
    parameters.r  = std::sqrt(radiusSquared);

    std::vector<double> residuals(points.size(), 0.0);
    std::vector<double> weights(points.size(), 1.0);
    const auto          computeResiduals = [&]
    {
        for (std::size_t index = 0; index < points.size(); ++index)
        {
            const double x = (points[index].x - centerX) / scale;
            const double y = (points[index].y - centerY) / scale;
            residuals[index] = std::hypot(x - parameters.cx, y - parameters.cy) - parameters.r;
        }
    };
    computeResiduals();
    const double initialMad = mad(residuals) * scale;
    bool         converged  = false;
    for (int iteration = 0; iteration < std::max(1, options.maxIterations); ++iteration)
    {
        computeResiduals();
        const double huberScale = std::max(kMinScale / scale, 1.4826 * mad(residuals));
        for (std::size_t index = 0; index < points.size(); ++index)
        {
            const double magnitude = std::abs(residuals[index]);
            weights[index] = magnitude <= kHuberTurn * huberScale ? 1.0 : kHuberTurn * huberScale / magnitude;
        }
        converged = refine(points, weights, parameters, centerX, centerY, scale);
        if (converged)
            break;
    }
    if (!std::isfinite(parameters.cx) || !std::isfinite(parameters.cy) || !std::isfinite(parameters.r)
        || !(parameters.r > 0.0))
        return fail("圆拟合发散或半径为负");
    computeResiduals();
    // Final inliers from the residual distribution, then one refit on inliers.
    const double finalMad   = mad(residuals) * scale;
    const double threshold  = std::max(kMinScale, std::min(3.0 * options.maxRms, kInlierSigmas * 1.4826 * finalMad));
    std::vector<cv::Point2d> inliers;
    std::vector<double>      inlierResiduals;
    for (std::size_t index = 0; index < points.size(); ++index)
        if (std::abs(residuals[index]) * scale <= threshold)
        {
            inliers.push_back(points[index]);
            inlierResiduals.push_back(std::abs(residuals[index]) * scale);
        }
    if (inliers.size() >= 3)
    {
        std::vector<double> unitWeights(inliers.size(), 1.0);
        Parameters          refined = parameters;
        for (int iteration = 0; iteration < std::max(1, options.maxIterations); ++iteration)
            if (refine(inliers, unitWeights, refined, centerX, centerY, scale))
                break;
        if (std::isfinite(refined.cx) && std::isfinite(refined.cy) && refined.r > 0.0)
            parameters = refined;
    }
    CircleFitResult result;
    result.centerX = parameters.cx * scale + centerX;
    result.centerY = parameters.cy * scale + centerY;
    result.radius  = parameters.r * scale;
    result.points  = points.size();
    double squared = 0;
    std::size_t counted = 0;
    for (const auto &point : points)
    {
        const double residual = std::hypot(point.x - result.centerX, point.y - result.centerY) - result.radius;
        if (std::abs(residual) <= threshold)
        {
            squared += residual * residual;
            ++counted;
        }
    }
    result.inliers = counted;
    result.rms     = counted > 0 ? std::sqrt(squared / static_cast<double>(counted)) : 0.0;
    if (!std::isfinite(result.radius) || !(result.radius > 0.0))
        return fail("圆拟合得到无效半径");
    if (!std::isfinite(initialMad))
        return fail("圆拟合残差无效");
    if (result.rms > options.maxRms)
        return fail("圆拟合均方根误差超限");
    return result;
}

double angularCoverage(std::vector<double> anglesDegrees)
{
    if (anglesDegrees.empty())
        return 0.0;
    for (auto &angle : anglesDegrees)
    {
        angle = std::fmod(angle, 360.0);
        if (angle < 0.0)
            angle += 360.0;
    }
    std::sort(anglesDegrees.begin(), anglesDegrees.end());
    double largestGap = 0;
    for (std::size_t index = 0; index < anglesDegrees.size(); ++index)
    {
        const double next = index + 1 == anglesDegrees.size() ? anglesDegrees.front() + 360.0
                                                              : anglesDegrees[index + 1];
        largestGap = std::max(largestGap, next - anglesDegrees[index]);
    }
    return std::clamp(360.0 - largestGap, 0.0, 360.0);
}
} // namespace visiontools::metrology
