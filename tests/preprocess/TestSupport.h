#pragma once

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <visionrt/contracts/Module.h>
#include <visionrt/contracts/Tool.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <cstdlib>

#define REQUIRE(cond, msg)                                                                                      \
    do                                                                                                          \
    {                                                                                                           \
        if (!(cond))                                                                                            \
        {                                                                                                       \
            std::cerr << "Assertion failed: " << #cond << " (" << msg << ") at line " << __LINE__ << std::endl; \
            std::exit(1);                                                                                       \
        }                                                                                                       \
    }                                                                                                           \
    while (0)

namespace visiontools::tests {

inline void requireSuccess(const visionrt::ToolResult &result, const char *context)
{
    REQUIRE(result.status.code == 0, context);
    REQUIRE(result.status.message == "运行成功", "Successful execute must use the shared success message");
}

inline void requireFailure(const visionrt::ToolResult &result, std::string_view expected, const char *context)
{
    REQUIRE(result.status.code != 0, context);
    REQUIRE(result.status.message.find(expected) != std::string::npos,
            std::string(context) + ": " + result.status.message);
}

inline cv::Mat gradient(int rows, int cols)
{
    cv::Mat image(rows, cols, CV_8UC1);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            image.at<unsigned char>(y, x) = static_cast<unsigned char>((x * 3 + y * 2) % 256);
    return image;
}

}
