#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace visiontools::common {

// The executable directory cannot be determined. The system reason is the message, so a tool can
// report it instead of guessing or falling back to the working directory.
class ProductDirectoryError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// Directory of the running product's main executable: VisionRT.exe in the desktop product, the
// consumer's own executable in a console tool or test.
std::filesystem::path productDirectory();

// Absolute and UNC paths are used as given; every other non-empty string is joined to
// productDirectory(). Only the tools that actually open a file call this.
std::filesystem::path resolveResourcePath(std::string_view utf8);

} // namespace visiontools::common
