#include "CalibrationFit.h"

#include "errors/MetrologyErrors.h"

#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>

namespace visiontools::metrology {
namespace {
enum FitParameter { H00, H01, H02, H10, H11, H12, H20, H21, CenterX, CenterY, Division, ParameterCount };
constexpr int kMaximumFitIterations = 150;
constexpr double kFitPrecision = 1e-12;

class DivisionResidual final : public cv::LMSolver::Callback
{
public:
    DivisionResidual(const std::vector<CalibrationMark> &marks, double scale) : marks_(marks), scale_(scale) {}
    bool compute(cv::InputArray parameters, cv::OutputArray error, cv::OutputArray jacobian) const override
    {
        const auto parameter = parameters.getMat();
        const auto *p = parameter.ptr<double>();
        error.create(static_cast<int>(marks_.size()) * 2, 1, CV_64F);
        auto errors = error.getMat();
        cv::Mat derivatives;
        if (jacobian.needed())
        {
            jacobian.create(errors.rows, ParameterCount, CV_64F);
            derivatives = jacobian.getMat();
            derivatives.setTo(0);
        }
        for (std::size_t i = 0; i < marks_.size(); ++i)
        {
            const auto &mark = marks_[i];
            const auto x = mark.millimeters.x, y = mark.millimeters.y;
            const auto denominator = p[H20] * x + p[H21] * y + 1;
            const auto u = (p[H00] * x + p[H01] * y + p[H02]) / denominator;
            const auto v = (p[H10] * x + p[H11] * y + p[H12]) / denominator;
            const auto dx = mark.image.x / scale_ - p[CenterX], dy = mark.image.y / scale_ - p[CenterY];
            const auto radius2 = dx * dx + dy * dy;
            const auto lens = 1 + p[Division] * radius2, lens2 = lens * lens;
            errors.at<double>(static_cast<int>(2*i)) = (u - p[CenterX] - dx / lens) * scale_;
            errors.at<double>(static_cast<int>(2*i+1)) = (v - p[CenterY] - dy / lens) * scale_;
            if (derivatives.empty()) continue;
            auto *jx = derivatives.ptr<double>(static_cast<int>(2*i));
            auto *jy = derivatives.ptr<double>(static_cast<int>(2*i+1));
            jx[H00] = x / denominator * scale_; jx[H01] = y / denominator * scale_; jx[H02] = scale_ / denominator;
            jy[H10] = x / denominator * scale_; jy[H11] = y / denominator * scale_; jy[H12] = scale_ / denominator;
            jx[H20] = -u * x / denominator * scale_; jx[H21] = -u * y / denominator * scale_;
            jy[H20] = -v * x / denominator * scale_; jy[H21] = -v * y / denominator * scale_;
            jx[CenterX] = -(1 - 1 / lens + 2 * p[Division] * dx * dx / lens2) * scale_;
            jx[CenterY] = -2 * p[Division] * dx * dy / lens2 * scale_;
            jy[CenterX] = jx[CenterY];
            jy[CenterY] = -(1 - 1 / lens + 2 * p[Division] * dy * dy / lens2) * scale_;
            jx[Division] = dx * radius2 / lens2 * scale_; jy[Division] = dy * radius2 / lens2 * scale_;
        }
        return cv::checkRange(errors) && (derivatives.empty() || cv::checkRange(derivatives));
    }
private:
    const std::vector<CalibrationMark> &marks_;
    double scale_;
};
}

CalibrationFit fitCalibration(const cv::Mat &gray, const CalibrationPlate &plate)
{
    CalibrationFit result;
    result.marks = observeCalibrationPlate(gray, plate);
    const double scale = std::max(gray.cols, gray.rows);
    double objectScale = 0.0;
    for (const auto &mark : result.marks)
        objectScale = std::max({objectScale, std::abs(mark.millimeters.x), std::abs(mark.millimeters.y)});
    if (!(objectScale > 0.0)) fail(MetrologyError::CalibrationFailed, "标定点坐标范围无效");

    std::vector<CalibrationMark> normalizedMarks = result.marks;
    std::vector<cv::Point2d> image, object;
    for (auto &mark : normalizedMarks)
    {
        mark.millimeters /= objectScale;
        image.push_back(mark.image / scale);
        object.push_back(mark.millimeters);
    }
    const cv::Point2d initialCenter(gray.cols / (2 * scale), gray.rows / (2 * scale));
    double bestDiv = 0.0;
    double bestError = std::numeric_limits<double>::infinity();
    cv::Matx33d bestH = cv::Matx33d::eye();
    for (const double candidateDiv : {-0.8, -0.6, -0.45, -0.32, -0.22, -0.14, -0.07, -0.02, 0.0, 0.05, 0.15, 0.30})
    {
        std::vector<cv::Point2d> undistorted;
        undistorted.reserve(normalizedMarks.size());
        bool valid = true;
        for (const auto &mark : normalizedMarks)
        {
            const cv::Point2d delta = mark.image / scale - initialCenter;
            const double lens = 1.0 + candidateDiv * delta.dot(delta);
            if (!(lens > 0.15)) { valid = false; break; }
            undistorted.push_back(initialCenter + delta / lens);
        }
        if (!valid) continue;
        const auto candidateHMat = cv::findHomography(object, undistorted);
        if (candidateHMat.empty()) continue;
        const cv::Matx33d candidateH(candidateHMat);
        double sumSq = 0.0;
        for (std::size_t i = 0; i < normalizedMarks.size(); ++i)
        {
            const auto proj = candidateH * cv::Vec3d(object[i].x, object[i].y, 1.0);
            if (std::abs(proj[2]) < 1e-12) { valid = false; break; }
            const cv::Point2d u(proj[0] / proj[2], proj[1] / proj[2]);
            const cv::Point2d diff = (u - undistorted[i]) * scale;
            sumSq += diff.dot(diff);
        }
        if (valid && sumSq < bestError)
        {
            bestError = sumSq;
            bestDiv = candidateDiv;
            bestH = candidateH;
        }
    }
    if (!std::isfinite(bestError)) fail(MetrologyError::CalibrationFailed, "标定点不能确定平面映射");
    cv::Mat parameters(ParameterCount, 1, CV_64F);
    auto *p = parameters.ptr<double>();
    for (int i = H00; i <= H21; ++i) p[i] = bestH.val[i] / bestH(2, 2);
    p[CenterX] = initialCenter.x; p[CenterY] = initialCenter.y; p[Division] = bestDiv;
    const auto callback = cv::makePtr<DivisionResidual>(normalizedMarks, scale);
    cv::LMSolver::create(callback, kMaximumFitIterations, kFitPrecision)->run(parameters);
    const cv::Matx33d fitted(p[H00], p[H01], p[H02], p[H10], p[H11], p[H12], p[H20], p[H21], 1);
    result.calibration.imageSize = gray.size();
    result.calibration.center = {p[CenterX] * scale, p[CenterY] * scale};
    result.calibration.division = p[Division];
    result.calibration.imageToMillimeters = cv::Matx33d(objectScale, 0, 0, 0, objectScale, 0, 0, 0, 1)
        * fitted.inv() * cv::Matx33d(1/scale, 0, 0, 0, 1/scale, 0, 0, 0, 1);
    result.calibration.validate();
    cv::Mat errors;
    if (!callback->compute(parameters, errors, cv::noArray())) fail(MetrologyError::CalibrationFailed, "重投影误差不可计算");
    result.reprojectionError = std::sqrt(errors.dot(errors) / result.marks.size());
    return result;
}
}
