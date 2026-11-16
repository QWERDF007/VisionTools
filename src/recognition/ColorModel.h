#pragma once

#include "TrainingAssets.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace visiontools::recognition {

// One trained colour as a two-dimensional Gaussian over the U/V chroma channels. The covariance is
// regularized once at training time, so classification only evaluates a fixed quadratic form.
struct ColorClass
{
    std::string       label;
    cv::Vec2d         mean;
    cv::Matx22d       inverse;
    double            logDeterminant = 0;
    std::size_t       pixels         = 0;
};

// One BGR frame converted to the U/V working space. Conversion is the expensive part of a frame,
// so it happens once and every detection circle of that frame shares the result.
class ChromaImage
{
public:
    explicit ChromaImage(const cv::Mat &bgr);

    const cv::Mat &planes() const noexcept
    {
        return planes_;
    }

private:
    cv::Mat planes_;
};

// The applied model of one recognition node. It is built from the training asset at initialization
// and never persisted separately: the asset is the single source of what was trained.
class ColorModel
{
public:
    struct Classification
    {
        std::string label = "unknown";
        double      score = 0;
        double      ratio = 0;
    };

    static ColorModel train(const ColorTraining &training);

    bool empty() const noexcept
    {
        return classes_.empty();
    }

    // Classifies every mask pixel. `minScore` rejects pixels that fit no class well, `minRatio`
    // rejects a region whose accepted pixels are too few; both produce "unknown" rather than the
    // nearest colour.
    Classification classify(const ChromaImage &image, const cv::Mat &mask, double minScore,
                            double minRatio) const;

private:
    std::vector<ColorClass> classes_;
};

} // namespace visiontools::recognition
