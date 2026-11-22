#pragma once

#include <opencv2/core.hpp>
#include <visionrt/contracts/ROI.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace visiontools::recognition {

// One trained colour as a two-dimensional Gaussian over the U/V chroma channels. The covariance is
// regularized once at training time, so classification only evaluates a fixed quadratic form.
struct ColorClass
{
    std::string label;
    cv::Vec2d   mean;
    cv::Matx22d inverse;
    double      logDeterminant = 0;
    std::size_t pixels         = 0;
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

// The published model of one recognition node. Training and recognition share this type and its
// byte format: the trainer returns exactly what the business tool decodes, so neither side keeps a
// second definition of a class or its statistics.
class ColorModel
{
public:
    struct Classification
    {
        std::string label = "unknown";
        double      score = 0;
        double      ratio = 0;
    };

    // One labelled frame region. `pixels` is the frame the record names, never a file: the trainer
    // receives the same clone the workbench shows, in the BGR order this model works in.
    struct Sample
    {
        std::string                          label;
        cv::Mat                              pixels;
        std::vector<visionrt::EncodedRegion> regions;
    };

    static ColorModel train(const std::vector<Sample> &samples);

    // Versioned binary form returned by the trainer and loaded by the business tool. The header
    // states the version, the chroma convention and the ridge, so a future change is rejected by
    // version instead of being misread as the current layout.
    std::vector<std::uint8_t> encode() const;
    static ColorModel         decode(std::span<const std::uint8_t> bytes);

    bool empty() const noexcept
    {
        return classes_.empty();
    }

    std::size_t classCount() const noexcept
    {
        return classes_.size();
    }

    // Classifies every mask pixel. `minScore` rejects pixels that fit no class well, `minRatio`
    // rejects a region whose accepted pixels are too few; both produce "unknown" rather than the
    // nearest colour.
    Classification classify(const ChromaImage &image, const cv::Mat &mask, double minScore,
                            double minRatio) const;

private:
    // The body of `decode`: it reports a truncated file by throwing, so `decode` can answer with
    // this model's own error code instead of a generic exception.
    static ColorModel read(std::span<const std::uint8_t> bytes);

    std::vector<ColorClass> classes_;
};

} // namespace visiontools::recognition
