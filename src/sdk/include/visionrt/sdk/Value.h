#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace visionrt {

enum class Type { Int, Float, Bool, String, ImageBatch, NumericArray, StringArray, RoiBatch };
enum class DType { UInt8, Int32, Int64, Float32, Float64, Bool };
enum class Layout { Dense, BatchRagged };

struct TypeSpec {
    Type type = Type::Int;
    std::optional<DType> dtype;
    int rank = -1;
    Layout layout = Layout::Dense;
    bool operator==(const TypeSpec&) const = default;
};

class ContractError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

void validateTypeSpec(const TypeSpec& type);
bool compatible(const TypeSpec& source, const TypeSpec& target);
std::size_t dtypeSize(DType dtype);
std::size_t checkedElementCount(std::span<const std::int64_t> shape);
std::size_t checkedByteCount(std::span<const std::int64_t> shape, std::size_t elementSize);
std::int64_t parseInt64(std::string_view decimal);
std::string formatInt64(std::int64_t value);
void validateUtf8(std::string_view value);

struct Missing { bool operator==(const Missing&) const = default; };

class Utf8 {
public:
    Utf8();
    Utf8(std::string value);
    Utf8(const char* value);
    const std::string& str() const noexcept { return *value_; }
    std::shared_ptr<const std::string> owner() const noexcept { return value_; }
    bool operator==(const Utf8& other) const noexcept { return str() == other.str(); }
private:
    std::shared_ptr<const std::string> value_;
};

// Owns only C++ storage; no borrowed pointer, Python object, or mutable accessor.
class BufferOwner {
public:
    static std::shared_ptr<const BufferOwner> copy(std::span<const std::byte> bytes);
    const std::byte* data() const noexcept { return bytes_.get(); }
    std::size_t size() const noexcept { return size_; }
private:
    explicit BufferOwner(std::span<const std::byte> bytes);
    std::unique_ptr<std::byte[]> bytes_;
    std::size_t size_;
};

class NumericBuffer {
public:
    static NumericBuffer copy(DType dtype, std::vector<std::int64_t> shape,
                              std::span<const std::byte> bytes);
    DType dtype() const noexcept { return dtype_; }
    std::span<const std::int64_t> shape() const noexcept { return *shape_; }
    const std::byte* data() const noexcept { return owner_->data(); }
    std::size_t byteSize() const noexcept { return owner_->size(); }
    std::shared_ptr<const BufferOwner> owner() const noexcept { return owner_; }
    template<class T> std::span<const T> values() const {
        constexpr DType requested = [] {
            if constexpr (std::is_same_v<T, std::uint8_t>) return DType::UInt8;
            else if constexpr (std::is_same_v<T, std::int32_t>) return DType::Int32;
            else if constexpr (std::is_same_v<T, std::int64_t>) return DType::Int64;
            else if constexpr (std::is_same_v<T, float>) return DType::Float32;
            else if constexpr (std::is_same_v<T, double>) return DType::Float64;
            else if constexpr (std::is_same_v<T, bool>) return DType::Bool;
            else static_assert(!sizeof(T), "Unsupported numeric element type");
        }();
        if (dtype_ != requested) throw ContractError("Numeric buffer dtype mismatch");
        return {reinterpret_cast<const T*>(data()), byteSize() / sizeof(T)};
    }
private:
    NumericBuffer(DType dtype, std::vector<std::int64_t> shape,
                  std::shared_ptr<const BufferOwner> owner);
    DType dtype_;
    std::shared_ptr<const std::vector<std::int64_t>> shape_;
    std::shared_ptr<const BufferOwner> owner_;
};

class NumericArray {
public:
    NumericArray(DType dtype, int rank, Layout layout, std::vector<NumericBuffer> buffers);
    DType dtype() const noexcept { return dtype_; }
    int rank() const noexcept { return rank_; }
    Layout layout() const noexcept { return layout_; }
    std::span<const NumericBuffer> buffers() const noexcept { return *buffers_; }
    std::shared_ptr<const std::vector<NumericBuffer>> owner() const noexcept { return buffers_; }
private:
    DType dtype_;
    int rank_;
    Layout layout_;
    std::shared_ptr<const std::vector<NumericBuffer>> buffers_;
};

class StringBuffer {
public:
    StringBuffer(std::vector<std::int64_t> shape, std::vector<std::string> values);
    std::span<const std::int64_t> shape() const noexcept { return *shape_; }
    std::span<const std::string> values() const noexcept { return *values_; }
    std::shared_ptr<const std::vector<std::string>> owner() const noexcept { return values_; }
private:
    std::shared_ptr<const std::vector<std::int64_t>> shape_;
    std::shared_ptr<const std::vector<std::string>> values_;
};

class StringArray {
public:
    StringArray(int rank, Layout layout, std::vector<StringBuffer> buffers);
    int rank() const noexcept { return rank_; }
    Layout layout() const noexcept { return layout_; }
    std::span<const StringBuffer> buffers() const noexcept { return *buffers_; }
    std::shared_ptr<const std::vector<StringBuffer>> owner() const noexcept { return buffers_; }
private:
    int rank_;
    Layout layout_;
    std::shared_ptr<const std::vector<StringBuffer>> buffers_;
};

class OwnedImage {
public:
    OwnedImage() = default;
    static OwnedImage copyOf(const cv::Mat& image);
    // Transfers unique OpenCV storage. Aliased/external storage is copied once.
    static OwnedImage take(cv::Mat&& image);
    OwnedImage region(const cv::Rect& rectangle) const;
    bool empty() const noexcept;
    int rows() const noexcept;
    int cols() const noexcept;
    int type() const noexcept;
    int channels() const noexcept;
    std::size_t rowStride() const noexcept;
    const unsigned char* data() const noexcept;
    std::shared_ptr<const void> owner() const noexcept { return image_; }
    // OpenCV input-only use: never shallow-copy this header to mutate its pixels.
    // Writable consumers must call clone(); shared outputs copy OwnedImage itself.
    const cv::Mat& readOnlyMat() const noexcept;
    cv::Mat clone() const;
private:
    explicit OwnedImage(cv::Mat image);
    std::shared_ptr<const cv::Mat> image_;
};

class ImageBatch {
public:
    ImageBatch();
    explicit ImageBatch(std::vector<OwnedImage> images);
    std::span<const OwnedImage> images() const noexcept { return *images_; }
    std::shared_ptr<const std::vector<OwnedImage>> owner() const noexcept { return images_; }
private:
    std::shared_ptr<const std::vector<OwnedImage>> images_;
};

enum class RoiKind { Segment = 1, Rectangle = 2, Circle = 3, Polygon = 4, PointSet = 5 };

class Roi {
public:
    static Roi decode(std::span<const float> encoded);
    static Roi segment(cv::Point2f start, cv::Point2f end);
    static Roi rectangle(float x, float y, float width, float height, float angle);
    static Roi circle(float x, float y, float radius);
    static Roi polygon(std::span<const cv::Point2f> points);
    static Roi points(std::span<const cv::Point2f> points);
    RoiKind kind() const noexcept { return static_cast<RoiKind>(static_cast<int>((*data_)[0])); }
    std::span<const float> encoded() const noexcept { return *data_; }
    std::shared_ptr<const std::vector<float>> owner() const noexcept { return data_; }
    // Corner order is top-left, top-right, bottom-right, bottom-left before rotation.
    std::array<cv::Point2d, 4> rectangleCorners() const;
private:
    explicit Roi(std::vector<float> encoded);
    std::shared_ptr<const std::vector<float>> data_;
};

class RoiBatch {
public:
    RoiBatch();
    explicit RoiBatch(std::vector<std::vector<Roi>> images);
    std::span<const std::vector<Roi>> images() const noexcept { return *images_; }
    std::shared_ptr<const std::vector<std::vector<Roi>>> owner() const noexcept { return images_; }
private:
    std::shared_ptr<const std::vector<std::vector<Roi>>> images_;
};

using PortValue = std::variant<Missing, std::int64_t, double, bool, Utf8,
                               ImageBatch, NumericArray, StringArray, RoiBatch>;
bool isMissing(const PortValue& value) noexcept;
// Configuration equality compares array/ROI contents, not shared owner addresses.
// Nonempty images compare the backing pixels and view identity, without pixel scans.
bool equalValues(const PortValue& left, const PortValue& right);
// Missing has no TypeSpec; it is accepted by every valid port declaration.
std::optional<TypeSpec> typeOf(const PortValue& value);
void validateValue(const TypeSpec& expected, const PortValue& value);

} // namespace visionrt
