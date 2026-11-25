#include "recognition/AlignByMatchingTool.h"
#include "recognition/CheckPresenceTool.h"
#include "common/GeometryTransform.h"
#include "ToolResultAssertions.h"
#include <visionrt/contracts/TrainingArchive.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

using namespace visionrt;
using namespace visiontools;
namespace align = recognition::ports::align_by_matching;
namespace pres = recognition::ports::check_presence;

namespace {
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

cv::Mat loadImage(const std::filesystem::path &path)
{
    std::ifstream stream(path, std::ios::binary);
    std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    auto image = cv::imdecode(bytes, cv::IMREAD_UNCHANGED);
    require(!image.empty(), "matching test image unavailable");
    return image;
}

std::string u8Path(const std::filesystem::path &p)
{
    const auto s = p.generic_u8string();
    return {reinterpret_cast<const char *>(s.data()), s.size()};
}

cv::Mat renderSyntheticConnector(cv::Point2d shift, double angleCcwDeg, bool bendPin2)
{
    cv::Mat base(360, 480, CV_8UC1, cv::Scalar(40));
    // Connector body with distinctive asymmetric notches for unambiguous rotation.
    cv::rectangle(base, cv::Rect(110, 180, 260, 80), cv::Scalar(210), cv::FILLED);
    cv::rectangle(base, cv::Rect(130, 195, 50, 45), cv::Scalar(70), cv::FILLED);
    cv::rectangle(base, cv::Rect(270, 200, 75, 35), cv::Scalar(90), cv::FILLED);
    cv::circle(base, cv::Point(225, 220), 16, cv::Scalar(55), cv::FILLED);

    // 4 circular pins above the connector.
    const std::vector<cv::Point> pins{{150, 130}, {210, 130}, {270, 130}, {330, 130}};
    for (std::size_t i = 0; i < pins.size(); ++i)
    {
        cv::Point pt = pins[i];
        if (bendPin2 && i == 2)
            pt += cv::Point(9, -6);
        cv::circle(base, pt, 12, cv::Scalar(235), cv::FILLED, cv::LINE_AA);
        cv::circle(base, pt, 5, cv::Scalar(95), cv::FILLED, cv::LINE_AA);
    }

    if (std::abs(angleCcwDeg) < 1e-6 && cv::norm(shift) < 1e-6)
        return base;

    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(240.0f, 220.0f), angleCcwDeg, 1.0);
    M.at<double>(0, 2) += shift.x;
    M.at<double>(1, 2) += shift.y;
    cv::Mat warped;
    cv::warpAffine(base, warped, M, base.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(40));
    return warped;
}

void verifySyntheticPipeline()
{
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto alignModelPath = std::filesystem::temp_directory_path() / ("vt-align-match-" + stamp + ".zip");
    const auto presModelPath = std::filesystem::temp_directory_path() / ("vt-pres-match-" + stamp + ".zip");
    struct Cleanup
    {
        std::filesystem::path a, b;
        ~Cleanup()
        {
            std::error_code ignored;
            std::filesystem::remove(a, ignored);
            std::filesystem::remove(b, ignored);
        }
    } cleanup{alignModelPath, presModelPath};

    const cv::Mat refImg = renderSyntheticConnector({0.0, 0.0}, 0.0, false);
    const cv::Mat okImg = renderSyntheticConnector({14.5, -9.2}, 4.5, false);
    const cv::Mat ngImg = renderSyntheticConnector({-11.0, 8.0}, -3.2, true);

    // 1. Train DetermineAlignmentWithMatching (shape mode).
    TrainingSet alignSet;
    alignSet.images.emplace(1, trainingFrame(1, refImg, ColorOrder::Automatic));
    alignSet.records.emplace(1, TrainingRecord{
        1, 1, {{align::field::TEMPLATE_ROI, Rectangle::fromEncoded(ROI{{makeRectangle(100.0, 170.0, 280.0, 100.0, 0.0)}})}}
    });

    recognition::AlignByMatchingTool alignTool;
    const auto alignTrained = alignTool.train({
        {align::train::MODEL_PATH, u8Path(alignModelPath)},
        {align::train::METHOD, std::string("shape")},
        {align::train::MODEL_CONTRAST, 15.0},
        {align::train::SEARCH_CONTRAST, 8.0},
        {align::train::ANGLE_TOLERANCE, 15.0},
        {align::train::ANGLE_STEP, 1.0},
        {align::train::POLARITY, std::string("use_polarity")},
        {align::train::NUM_FEATURES, std::int64_t{128}},
        {align::train::DATASET, DataRef<TrainingSet>(alignSet)}
    });
    if (!testview::status(alignTrained))
        std::cerr << testview::status(alignTrained).message << ": " << testview::status(alignTrained).detail << '\n';
    require(static_cast<bool>(testview::status(alignTrained)), "synthetic alignment training succeeded");
    require(std::filesystem::is_regular_file(alignModelPath), "alignment zip written");
    require(TrainingArchive::readImages(alignModelPath).size() == 1, "alignment zip keeps training image");

    ToolConfiguration alignCfg;
    alignCfg.models.emplace(align::MODEL_ARTIFACT, alignModelPath);
    require(static_cast<bool>(alignTool.initialize(alignCfg)), "alignment model loaded from zip");

    const auto okAlignRes = alignTool.execute({
        {align::in::IMAGES, DataRef<Image>(Image{okImg})},
        {align::in::SEARCH_ROI, Rectangle{}},
        {align::in::MIN_SCORE, 0.5},
        {align::in::ANGLE_TOLERANCE, 15.0}
    });
    require(static_cast<bool>(testview::status(okAlignRes)), "synthetic rotated alignment succeeded");
    const auto okAlignMat = geometry::alignmentsFromValue(okAlignRes.outputs.at(align::out::ALIGNMENTS), 1).front();
    const double detectedAngleCcw = -std::atan2(okAlignMat.m[3], okAlignMat.m[0]) * 180.0 / std::numbers::pi;
    double tx = 0.0, ty = 0.0;
    okAlignMat.apply(240.0, 220.0, tx, ty);
    require(std::abs(detectedAngleCcw - 4.5) < 0.3, "synthetic alignment rotation accurate within 0.3 deg");
    require(std::abs(tx - 254.5) < 0.5 && std::abs(ty - 210.8) < 0.5, "synthetic alignment center accurate within 0.5 px");

    // 2. Train CheckPresenceWithMatching (dual datasets, automatic reference discovery).
    TrainingSet objSet;
    objSet.images.emplace(1, trainingFrame(1, refImg, ColorOrder::Automatic));
    objSet.records.emplace(1, TrainingRecord{
        1, 1, {{pres::field::OBJECT_ROI, Circle::fromEncoded(ROI{{makeCircle({150.0, 130.0}, 15.0)}})}}
    });
    TrainingSet posSet;
    posSet.images.emplace(1, trainingFrame(1, refImg, ColorOrder::Automatic));
    posSet.records.emplace(1, TrainingRecord{1, 1, {{pres::field::REFERENCE_POINTS, Point{}}}});

    recognition::CheckPresenceTool presTool;
    const auto presTrained = presTool.train({
        {pres::train::MODEL_PATH, u8Path(presModelPath)},
        {pres::train::METHOD, std::string("correlation")},
        {pres::train::MIN_SCORE, 0.75},
        {pres::train::ANGLE_TOLERANCE, 5.0},
        {pres::train::MAX_OVERLAP, 30.0},
        {pres::train::MODEL_CONTRAST, 10.0},
        {pres::train::OBJECT_DATASET, DataRef<TrainingSet>(objSet)},
        {pres::train::POSITION_DATASET, DataRef<TrainingSet>(posSet)}
    });
    if (!testview::status(presTrained))
        std::cerr << testview::status(presTrained).message << ": " << testview::status(presTrained).detail << '\n';
    require(static_cast<bool>(testview::status(presTrained)), "synthetic presence training succeeded");
    require(TrainingArchive::readImages(presModelPath).size() == 2, "presence zip keeps both datasets' images");
    const auto trainedRefPts = std::get<Point>(presTrained.outputs.at(pres::out::REFERENCE_POSITIONS)).get().front();
    require(trainedRefPts.size() == 4, "auto-discovered all 4 synthetic pins");

    ToolConfiguration presCfg;
    presCfg.models.emplace(pres::MODEL_ARTIFACT, presModelPath);
    require(static_cast<bool>(presTool.initialize(presCfg)), "presence model loaded from zip");

    // Execute on rotated OK image -> 4/4 accepted.
    const auto okPres = presTool.execute({
        {pres::in::IMAGES, DataRef<Image>(Image{okImg})},
        {pres::in::ALIGNMENTS, okAlignRes.outputs.at(align::out::ALIGNMENTS)},
        {pres::in::MAX_DISTANCE, 3.0},
        {pres::in::MIN_SCORE, 0.5},
        {pres::in::MAX_OVERLAP, 30.0},
        {pres::in::FAILED_PROCESSING, std::string("ignore")}
    });
    require(static_cast<bool>(testview::status(okPres)), "presence execute on OK image succeeded");
    require(nestedOf<NestedVector<bool, 1>>(std::get<Array>(okPres.outputs.at(pres::out::ALL_ACCEPTED))).front(),
            "all 4 pins accepted on OK image");
    require(nestedOf<NestedVector<double, 2>>(std::get<Array>(okPres.outputs.at(pres::out::ABSOLUTE_DISTANCE))).front().size() == 4,
            "4 distances reported on OK image");

    // Execute on rotated NG image (pin 2 bent) -> ignore mode gives 3 distances; nan mode gives 4 with 1 NaN.
    const auto ngAlignRes = alignTool.execute({
        {align::in::IMAGES, DataRef<Image>(Image{ngImg})},
        {align::in::SEARCH_ROI, Rectangle{}},
        {align::in::MIN_SCORE, 0.5},
        {align::in::ANGLE_TOLERANCE, 15.0}
    });
    require(static_cast<bool>(testview::status(ngAlignRes)), "alignment on NG image succeeded");

    const auto ngPresIgnore = presTool.execute({
        {pres::in::IMAGES, DataRef<Image>(Image{ngImg})},
        {pres::in::ALIGNMENTS, ngAlignRes.outputs.at(align::out::ALIGNMENTS)},
        {pres::in::MAX_DISTANCE, 3.0},
        {pres::in::MIN_SCORE, 0.5},
        {pres::in::MAX_OVERLAP, 30.0},
        {pres::in::FAILED_PROCESSING, std::string("ignore")}
    });
    require(static_cast<bool>(testview::status(ngPresIgnore)), "presence execute on NG image (ignore) succeeded");
    require(!nestedOf<NestedVector<bool, 1>>(std::get<Array>(ngPresIgnore.outputs.at(pres::out::ALL_ACCEPTED))).front(),
            "bent pin fails All_Objects_Accepted");
    const auto ngAcc = nestedOf<NestedVector<bool, 2>>(std::get<Array>(ngPresIgnore.outputs.at(pres::out::OBJECT_ACCEPTED))).front();
    require(ngAcc.size() == 4 && ngAcc[0] && ngAcc[1] && !ngAcc[2] && ngAcc[3], "exact bent pin index rejected");
    require(nestedOf<NestedVector<double, 2>>(std::get<Array>(ngPresIgnore.outputs.at(pres::out::ABSOLUTE_DISTANCE))).front().size() == 3,
            "ignore mode omits failed pin distance");

    const auto ngPresNan = presTool.execute({
        {pres::in::IMAGES, DataRef<Image>(Image{ngImg})},
        {pres::in::ALIGNMENTS, ngAlignRes.outputs.at(align::out::ALIGNMENTS)},
        {pres::in::MAX_DISTANCE, 3.0},
        {pres::in::MIN_SCORE, 0.5},
        {pres::in::MAX_OVERLAP, 30.0},
        {pres::in::FAILED_PROCESSING, std::string("nan")}
    });
    require(static_cast<bool>(testview::status(ngPresNan)), "presence execute on NG image (nan) succeeded");
    const auto nanDists = nestedOf<NestedVector<double, 2>>(std::get<Array>(ngPresNan.outputs.at(pres::out::ABSOLUTE_DISTANCE))).front();
    require(nanDists.size() == 4 && std::isnan(nanDists[2]), "nan mode writes NaN for failed pin");

    // Upstream skip check.
    const auto skipped = presTool.execute({
        {kUpstreamStatusPort, std::int64_t{9}},
        {pres::in::IMAGES, DataRef<Image>(Image{okImg})},
        {pres::in::ALIGNMENTS, okAlignRes.outputs.at(align::out::ALIGNMENTS)},
        {pres::in::MAX_DISTANCE, 3.0},
        {pres::in::MIN_SCORE, 0.5},
        {pres::in::MAX_OVERLAP, 30.0},
        {pres::in::FAILED_PROCESSING, std::string("ignore")}
    });
    require(testview::status(skipped).code == 9 && isMissing(skipped.outputs.at(pres::out::ALL_ACCEPTED)),
            "upstream failure skips presence check");

    alignTool.release();
    presTool.release();
}

void verifyRealBentLeads(const std::filesystem::path &imageDir,
                         const std::filesystem::path &alignModelOut,
                         const std::filesystem::path &presModelOut)
{
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto alignPath = alignModelOut.empty()
        ? (std::filesystem::temp_directory_path() / ("vt-real-align-" + stamp + ".zip"))
        : alignModelOut;
    const auto presPath = presModelOut.empty()
        ? (std::filesystem::temp_directory_path() / ("vt-real-pres-" + stamp + ".zip"))
        : presModelOut;
    if (alignPath.has_parent_path()) std::filesystem::create_directories(alignPath.parent_path());
    if (presPath.has_parent_path()) std::filesystem::create_directories(presPath.parent_path());

    const cv::Mat img01 = loadImage(imageDir / "bent_leads_01.png");

    // MERLIC Training_ROI for DetermineAlignmentWithMatching:
    // center (col=595.5667, row=576.1923), width=904.1255, height=181.8353, angle=0.
    const double alignW = 904.1254901960784;
    const double alignH = 181.835294117647;
    const double alignLeft = 595.5666666666666 - alignW * 0.5;
    const double alignTop = 576.1923076923076 - alignH * 0.5;

    TrainingSet alignSet;
    alignSet.images.emplace(1, trainingFrame(1, img01, ColorOrder::Automatic));
    alignSet.records.emplace(1, TrainingRecord{
        1, 1, {{align::field::TEMPLATE_ROI, Rectangle::fromEncoded(ROI{{makeRectangle(alignLeft, alignTop, alignW, alignH, 0.0)}})}}
    });

    recognition::AlignByMatchingTool alignTool;
    const auto aTrained = alignTool.train({
        {align::train::MODEL_PATH, u8Path(alignPath)},
        {align::train::METHOD, std::string("shape")},
        {align::train::MODEL_CONTRAST, 10.0},
        {align::train::SEARCH_CONTRAST, 5.0},
        {align::train::ANGLE_TOLERANCE, 20.0},
        {align::train::ANGLE_STEP, 1.0},
        {align::train::POLARITY, std::string("use_polarity")},
        {align::train::NUM_FEATURES, std::int64_t{128}},
        {align::train::DATASET, DataRef<TrainingSet>(alignSet)}
    });
    if (!testview::status(aTrained))
        std::cerr << testview::status(aTrained).message << ": " << testview::status(aTrained).detail << '\n';
    require(static_cast<bool>(testview::status(aTrained)), "real bent_leads alignment training succeeded");

    ToolConfiguration aCfg;
    aCfg.models.emplace(align::MODEL_ARTIFACT, alignPath);
    require(static_cast<bool>(alignTool.initialize(aCfg)), "real bent_leads alignment model loaded");

    // MERLIC CheckPresenceWithMatching:
    // Object circle: (col=567.7863, row=437.2278, r=18.2116).
    // 25 reference positions from MERLIC check_bent_leads.
    const std::vector<cv::Point2d> merlicRefs = {
        {346.6207105496316, 396.2635577198920}, {366.1701300856635, 437.3642787345493},
        {386.0691032409668, 395.9750061035156}, {407.0751953125000, 438.2521057128906},
        {426.2909240722656, 396.0199584960938}, {448.0273742675781, 435.4027099609375},
        {465.5419921875000, 395.5499877929688}, {488.2680053710938, 437.7449951171875},
        {505.7309875488281, 395.6310119628906}, {529.1250000000000, 436.5969848632812},
        {546.0809936523438, 398.1770019531250}, {567.5000000000000, 437.5000000000000},
        {589.7000122070312, 396.1050109863281}, {608.5620117187500, 437.1210021972656},
        {628.6040039062500, 396.7720031738281}, {648.5250244140625, 433.7019958496094},
        {672.5410156250000, 398.1570129394531}, {690.4680175781250, 435.6189880371094},
        {710.4840087890625, 398.6820068359375}, {729.5520019531250, 435.7189941406250},
        {753.0130004882812, 396.1270141601562}, {770.1199951171875, 437.5759887695312},
        {790.5300292968750, 395.2090148925781}, {810.7440185546875, 432.6040039062500},
        {831.4619750976562, 397.0520019531250}
    };
    std::vector<EncodedRegion> refPointRecords;
    refPointRecords.reserve(merlicRefs.size());
    for (const auto &pt : merlicRefs)
        refPointRecords.push_back(makePoint(pt.x, pt.y));

    TrainingSet objSet;
    objSet.images.emplace(1, trainingFrame(1, img01, ColorOrder::Automatic));
    objSet.records.emplace(1, TrainingRecord{
        1, 1, {{pres::field::OBJECT_ROI, Circle::fromEncoded(ROI{{makeCircle({567.7862745098038, 437.2278106508874}, 18.21157036725636)}})}}
    });
    TrainingSet posSet;
    posSet.images.emplace(1, trainingFrame(1, img01, ColorOrder::Automatic));
    posSet.records.emplace(1, TrainingRecord{
        1, 1, {{pres::field::REFERENCE_POINTS, Point::fromEncoded(ROI{refPointRecords})}}
    });

    recognition::CheckPresenceTool presTool;
    const auto pTrained = presTool.train({
        {pres::train::MODEL_PATH, u8Path(presPath)},
        {pres::train::METHOD, std::string("correlation")},
        {pres::train::MIN_SCORE, 0.75},
        {pres::train::ANGLE_TOLERANCE, 5.0},
        {pres::train::MAX_OVERLAP, 30.0},
        {pres::train::MODEL_CONTRAST, 10.0},
        {pres::train::OBJECT_DATASET, DataRef<TrainingSet>(objSet)},
        {pres::train::POSITION_DATASET, DataRef<TrainingSet>(posSet)}
    });
    if (!testview::status(pTrained))
        std::cerr << testview::status(pTrained).message << ": " << testview::status(pTrained).detail << '\n';
    require(static_cast<bool>(testview::status(pTrained)), "real bent_leads presence training succeeded");

    ToolConfiguration pCfg;
    pCfg.models.emplace(pres::MODEL_ARTIFACT, presPath);
    require(static_cast<bool>(presTool.initialize(pCfg)), "real bent_leads presence model loaded");

    // Run across all 9 bent_leads images.
    for (int i = 1; i <= 9; ++i)
    {
        const std::string name = "bent_leads_0" + std::to_string(i) + ".png";
        const cv::Mat frame = loadImage(imageDir / name);
        const auto aRes = alignTool.execute({
            {align::in::IMAGES, DataRef<Image>(Image{frame})},
            {align::in::SEARCH_ROI, Rectangle{}},
            {align::in::MIN_SCORE, 0.5},
            {align::in::ANGLE_TOLERANCE, 20.0}
        });
        require(static_cast<bool>(testview::status(aRes)), "alignment succeeded on real bent_leads frame");
        const double score = nestedOf<NestedVector<double, 1>>(std::get<Array>(aRes.outputs.at(align::out::SCORES))).front();
        const auto mat = geometry::alignmentsFromValue(aRes.outputs.at(align::out::ALIGNMENTS), 1).front();
        const double angleCcw = -std::atan2(mat.m[3], mat.m[0]) * 180.0 / std::numbers::pi;

        const auto pRes = presTool.execute({
            {pres::in::IMAGES, DataRef<Image>(Image{frame})},
            {pres::in::ALIGNMENTS, aRes.outputs.at(align::out::ALIGNMENTS)},
            {pres::in::MAX_DISTANCE, 3.0},
            {pres::in::MIN_SCORE, 0.5},
            {pres::in::MAX_OVERLAP, 30.0},
            {pres::in::FAILED_PROCESSING, std::string("ignore")}
        });
        require(static_cast<bool>(testview::status(pRes)), "presence check succeeded on real bent_leads frame");
        const bool allOk = nestedOf<NestedVector<bool, 1>>(std::get<Array>(pRes.outputs.at(pres::out::ALL_ACCEPTED))).front();
        const auto acc = nestedOf<NestedVector<bool, 2>>(std::get<Array>(pRes.outputs.at(pres::out::OBJECT_ACCEPTED))).front();
        const auto dists = nestedOf<NestedVector<double, 2>>(std::get<Array>(pRes.outputs.at(pres::out::ABSOLUTE_DISTANCE))).front();
        const auto okCount = std::count(acc.begin(), acc.end(), true);
        std::cout << name << ": alignScore=" << score << " angleCcw=" << angleCcw
                  << " deg accepted=" << okCount << "/25 allOk=" << (allOk ? "true" : "false")
                  << " distCount=" << dists.size() << "\n";
        require(score >= 0.75, "real bent_leads alignment score >= 0.75");
        if (i == 1)
        {
            require(allOk && okCount == 25 && dists.size() == 25, "bent_leads_01 must pass 25/25");
        }
        if (i == 3)
        {
            require(std::abs(angleCcw - 2.417) < 0.3, "bent_leads_03 rotation near 2.417 deg");
            require(!allOk && okCount == 24 && !acc[15] && dists.size() == 24,
                    "bent_leads_03 must pass 24/25 with pin 15 rejected");
        }
    }

    if (alignModelOut.empty())
    {
        std::error_code ignored;
        std::filesystem::remove(alignPath, ignored);
    }
    if (presModelOut.empty())
    {
        std::error_code ignored;
        std::filesystem::remove(presPath, ignored);
    }
    alignTool.release();
    presTool.release();
}
} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t **argv)
#else
int main(int argc, char **argv)
#endif
try
{
    verifySyntheticPipeline();
    if (argc >= 2)
    {
        const std::filesystem::path alignOut = argc >= 3 ? std::filesystem::path(argv[2]) : std::filesystem::path{};
        const std::filesystem::path presOut = argc >= 4 ? std::filesystem::path(argv[3]) : std::filesystem::path{};
        verifyRealBentLeads(std::filesystem::path(argv[1]), alignOut, presOut);
    }
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
