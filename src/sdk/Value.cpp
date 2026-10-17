#include <visionrt/sdk/Value.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <utility>

namespace visionrt {
namespace {

void validateArrayShape(int rank, Layout layout, std::size_t bufferCount) {
    if (layout != Layout::Dense && layout != Layout::BatchRagged)
        throw ContractError("Unknown array layout");
    if (rank < 0 || (layout == Layout::BatchRagged && rank < 2))
        throw ContractError("Array rank must be nonnegative; batch_ragged requires rank >= 2");
    if (layout == Layout::Dense && bufferCount != 1)
        throw ContractError("Dense array requires exactly one buffer");
}

void validateImage(const cv::Mat& image) {
    if (image.dims != 2 || image.empty())
        throw ContractError("Image must be a nonempty two-dimensional OpenCV matrix");
    const auto rowBytes = checkedByteCount(
        std::array<std::int64_t, 1>{image.cols}, image.elemSize());
    if (image.step[1] != image.elemSize() || image.step[0] < rowBytes)
        throw ContractError("Image pixel/row strides exceed its buffer layout");
    const auto rows = static_cast<std::size_t>(image.rows - 1);
    if (rows != 0 && image.step[0] > (std::numeric_limits<std::size_t>::max() - rowBytes) / rows)
        throw ContractError("Image stride byte count overflows size_t");
    const auto bytes = rows * image.step[0] + rowBytes;
    if (!image.dataend || image.dataend < image.data ||
        bytes > static_cast<std::size_t>(image.dataend - image.data))
        throw ContractError("Image strides exceed backing buffer range");
}

void validateRoi(std::span<const float> encoded) {
    if (encoded.empty()) throw ContractError("ROI is missing its type code");
    for (const float value : encoded)
        if (!std::isfinite(value)) throw ContractError("ROI coordinates and type code must be finite");
    const float code = encoded[0];
    if (code == 1) {
        if (encoded.size() != 5) throw ContractError("Segment ROI requires 5 float32 elements");
    } else if (code == 2) {
        if (encoded.size() != 6) throw ContractError("Rectangle ROI requires 6 float32 elements");
        if (encoded[3] <= 0 || encoded[4] <= 0)
            throw ContractError("Rectangle ROI width and height must be positive");
    } else if (code == 3) {
        if (encoded.size() != 4) throw ContractError("Circle ROI requires 4 float32 elements");
        if (encoded[3] <= 0) throw ContractError("Circle ROI radius must be positive");
    } else if (code == 4) {
        if (encoded.size() < 7 || encoded.size() % 2 != 1)
            throw ContractError("Polygon ROI requires at least three coordinate pairs");
    } else if (code == 5) {
        if (encoded.size() < 3 || encoded.size() % 2 != 1)
            throw ContractError("Point-set ROI requires at least one coordinate pair");
    } else {
        throw ContractError("ROI type code must be an integer in [1,5]");
    }
}

std::vector<float> encodePoints(RoiKind kind, std::span<const cv::Point2f> points) {
    if (points.size() > (std::numeric_limits<std::size_t>::max() - 1) / 2)
        throw ContractError("ROI coordinate count overflows size_t");
    std::vector<float> encoded;
    encoded.reserve(1 + points.size() * 2);
    encoded.push_back(static_cast<float>(kind));
    for (const auto& point : points) {
        encoded.push_back(point.x);
        encoded.push_back(point.y);
    }
    return encoded;
}

} // namespace

std::size_t dtypeSize(DType dtype) {
    switch (dtype) {
    case DType::UInt8: return sizeof(std::uint8_t);
    case DType::Int32: return sizeof(std::int32_t);
    case DType::Int64: return sizeof(std::int64_t);
    case DType::Float32: return sizeof(float);
    case DType::Float64: return sizeof(double);
    case DType::Bool: return sizeof(bool);
    }
    throw ContractError("Unknown numeric dtype");
}

void validateTypeSpec(const TypeSpec& type) {
    if (type.type == Type::NumericArray || type.type == Type::StringArray) {
        validateArrayShape(type.rank, type.layout, 1);
        if (type.type == Type::NumericArray) {
            if (!type.dtype) throw ContractError("NumericArray requires an explicit dtype");
            (void)dtypeSize(*type.dtype);
        } else if (type.dtype) {
            throw ContractError("StringArray must not declare a numeric dtype");
        }
        return;
    }
    switch (type.type) {
    case Type::Int: case Type::Float: case Type::Bool: case Type::String:
    case Type::ImageBatch: case Type::RoiBatch: break;
    default: throw ContractError("Unknown port type");
    }
    if (type.dtype || type.rank != -1 || type.layout != Layout::Dense)
        throw ContractError("Only arrays may declare dtype, rank, or layout");
}

bool compatible(const TypeSpec& source, const TypeSpec& target) {
    validateTypeSpec(source);
    validateTypeSpec(target);
    return source == target;
}

std::size_t checkedElementCount(std::span<const std::int64_t> shape) {
    bool zero = false;
    for (const auto dimension : shape) {
        if (dimension < 0) throw ContractError("Array shape contains a negative dimension");
        if (static_cast<std::uint64_t>(dimension) > std::numeric_limits<std::size_t>::max())
            throw ContractError("Array dimension exceeds size_t");
        zero = zero || dimension == 0;
    }
    if (zero) return 0;
    std::size_t result = 1;
    for (const auto dimension : shape) {
        const auto size = static_cast<std::size_t>(dimension);
        if (result > std::numeric_limits<std::size_t>::max() / size)
            throw ContractError("Array shape element count overflows size_t");
        result *= size;
    }
    return result;
}

std::size_t checkedByteCount(std::span<const std::int64_t> shape, std::size_t elementSize) {
    if (elementSize == 0) throw ContractError("Array element size must be positive");
    const auto count = checkedElementCount(shape);
    if (count > std::numeric_limits<std::size_t>::max() / elementSize)
        throw ContractError("Array shape byte count overflows size_t");
    return count * elementSize;
}

std::int64_t parseInt64(std::string_view decimal) {
    if (decimal.empty()) throw ContractError("Expected an int64 decimal string");
    std::int64_t value = 0;
    const auto result = std::from_chars(decimal.data(), decimal.data() + decimal.size(), value, 10);
    if (result.ec == std::errc::result_out_of_range)
        throw ContractError("Decimal integer is outside the int64 range");
    if (result.ec != std::errc{} || result.ptr != decimal.data() + decimal.size())
        throw ContractError("Expected only an optional minus and int64 decimal digits");
    return value;
}

std::string formatInt64(std::int64_t value) {
    std::array<char, 20> text{};
    const auto result = std::to_chars(text.data(), text.data() + text.size(), value);
    return std::string(text.data(), result.ptr);
}

void validateUtf8(std::string_view value) {
    for (std::size_t i = 0; i < value.size();) {
        const auto first = static_cast<unsigned char>(value[i++]);
        if (first < 0x80) continue;
        std::uint32_t code = 0;
        std::uint32_t minimum = 0;
        std::size_t remaining = 0;
        if (first >= 0xc2 && first <= 0xdf) {
            code = first & 0x1f; minimum = 0x80; remaining = 1;
        } else if (first >= 0xe0 && first <= 0xef) {
            code = first & 0x0f; minimum = 0x800; remaining = 2;
        } else if (first >= 0xf0 && first <= 0xf4) {
            code = first & 0x07; minimum = 0x10000; remaining = 3;
        } else {
            throw ContractError("String contains invalid UTF-8");
        }
        if (remaining > value.size() - i) throw ContractError("String contains truncated UTF-8");
        while (remaining-- != 0) {
            const auto next = static_cast<unsigned char>(value[i++]);
            if ((next & 0xc0) != 0x80) throw ContractError("String contains invalid UTF-8 continuation");
            code = (code << 6) | (next & 0x3f);
        }
        if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff))
            throw ContractError("String contains an invalid UTF-8 code point");
    }
}

Utf8::Utf8() : Utf8(std::string{}) {}
Utf8::Utf8(std::string value) {
    validateUtf8(value);
    value_ = std::make_shared<const std::string>(std::move(value));
}
Utf8::Utf8(const char* value) : Utf8(std::string(value ? value : "")) {
    if (!value) throw ContractError("UTF-8 string cannot be a null pointer");
}

BufferOwner::BufferOwner(std::span<const std::byte> bytes)
    : bytes_(bytes.empty() ? nullptr : new std::byte[bytes.size()]), size_(bytes.size()) {
    if (!bytes.empty()) std::memcpy(bytes_.get(), bytes.data(), bytes.size());
}
std::shared_ptr<const BufferOwner> BufferOwner::copy(std::span<const std::byte> bytes) {
    return std::shared_ptr<const BufferOwner>(new BufferOwner(bytes));
}

NumericBuffer::NumericBuffer(DType dtype, std::vector<std::int64_t> shape,
                             std::shared_ptr<const BufferOwner> owner)
    : dtype_(dtype), shape_(std::make_shared<const std::vector<std::int64_t>>(std::move(shape))),
      owner_(std::move(owner)) {}

NumericBuffer NumericBuffer::copy(DType dtype, std::vector<std::int64_t> shape,
                                   std::span<const std::byte> bytes) {
    if (checkedByteCount(shape, dtypeSize(dtype)) != bytes.size())
        throw ContractError("Numeric array shape byte count does not match its buffer range");
    if (dtype == DType::Bool) {
        static_assert(sizeof(bool) == 1);
        for (const auto byte : bytes)
            if (byte != std::byte{0} && byte != std::byte{1})
                throw ContractError("Bool array elements must be represented by 0 or 1");
    }
    return NumericBuffer(dtype, std::move(shape), BufferOwner::copy(bytes));
}

NumericArray::NumericArray(DType dtype, int rank, Layout layout, std::vector<NumericBuffer> buffers)
    : dtype_(dtype), rank_(rank), layout_(layout) {
    validateTypeSpec({Type::NumericArray, dtype, rank, layout});
    validateArrayShape(rank, layout, buffers.size());
    const auto leafRank = static_cast<std::size_t>(rank - (layout == Layout::BatchRagged ? 1 : 0));
    for (const auto& buffer : buffers) {
        if (buffer.dtype() != dtype) throw ContractError("Numeric array leaf dtype mismatch");
        if (buffer.shape().size() != leafRank) throw ContractError("Numeric array leaf rank mismatch");
    }
    buffers_ = std::make_shared<const std::vector<NumericBuffer>>(std::move(buffers));
}

StringBuffer::StringBuffer(std::vector<std::int64_t> shape, std::vector<std::string> values) {
    if (checkedElementCount(shape) != values.size())
        throw ContractError("String array shape does not match its element count");
    for (const auto& value : values) validateUtf8(value);
    shape_ = std::make_shared<const std::vector<std::int64_t>>(std::move(shape));
    values_ = std::make_shared<const std::vector<std::string>>(std::move(values));
}

StringArray::StringArray(int rank, Layout layout, std::vector<StringBuffer> buffers)
    : rank_(rank), layout_(layout) {
    validateTypeSpec({Type::StringArray, {}, rank, layout});
    validateArrayShape(rank, layout, buffers.size());
    const auto leafRank = static_cast<std::size_t>(rank - (layout == Layout::BatchRagged ? 1 : 0));
    for (const auto& buffer : buffers)
        if (buffer.shape().size() != leafRank) throw ContractError("String array leaf rank mismatch");
    buffers_ = std::make_shared<const std::vector<StringBuffer>>(std::move(buffers));
}

OwnedImage::OwnedImage(cv::Mat image)
    : image_(std::make_shared<const cv::Mat>(std::move(image))) {}

OwnedImage OwnedImage::copyOf(const cv::Mat& image) {
    validateImage(image);
    return OwnedImage(image.clone());
}

OwnedImage OwnedImage::take(cv::Mat&& image) {
    validateImage(image);
    if (!image.u || image.u->refcount != 1) return OwnedImage(image.clone());
    return OwnedImage(std::move(image));
}

OwnedImage OwnedImage::region(const cv::Rect& rectangle) const {
    if (empty() || rectangle.x < 0 || rectangle.y < 0 || rectangle.width <= 0 || rectangle.height <= 0 ||
        rectangle.x > cols() || rectangle.y > rows() ||
        rectangle.width > cols() - rectangle.x || rectangle.height > rows() - rectangle.y)
        throw ContractError("Image region is outside the image bounds or empty");
    return OwnedImage((*image_)(rectangle));
}

bool OwnedImage::empty() const noexcept { return !image_ || image_->empty(); }
int OwnedImage::rows() const noexcept { return image_ ? image_->rows : 0; }
int OwnedImage::cols() const noexcept { return image_ ? image_->cols : 0; }
int OwnedImage::type() const noexcept { return image_ ? image_->type() : -1; }
int OwnedImage::channels() const noexcept { return image_ ? image_->channels() : 0; }
std::size_t OwnedImage::rowStride() const noexcept { return image_ ? image_->step[0] : 0; }
const unsigned char* OwnedImage::data() const noexcept { return image_ ? image_->data : nullptr; }
const cv::Mat& OwnedImage::readOnlyMat() const noexcept {
    static const cv::Mat emptyImage;
    return image_ ? *image_ : emptyImage;
}
cv::Mat OwnedImage::clone() const { return readOnlyMat().clone(); }

ImageBatch::ImageBatch() : ImageBatch(std::vector<OwnedImage>{}) {}
ImageBatch::ImageBatch(std::vector<OwnedImage> images) {
    for (const auto& image : images)
        if (image.empty()) throw ContractError("ImageBatch cannot contain an empty image; use an empty batch");
    images_ = std::make_shared<const std::vector<OwnedImage>>(std::move(images));
}

Roi::Roi(std::vector<float> encoded) {
    validateRoi(encoded);
    data_ = std::make_shared<const std::vector<float>>(std::move(encoded));
}
Roi Roi::decode(std::span<const float> encoded) {
    return Roi(std::vector<float>(encoded.begin(), encoded.end()));
}
Roi Roi::segment(cv::Point2f start, cv::Point2f end) { return Roi({1, start.x, start.y, end.x, end.y}); }
Roi Roi::rectangle(float x, float y, float width, float height, float angle) {
    return Roi({2, x, y, width, height, angle});
}
Roi Roi::circle(float x, float y, float radius) { return Roi({3, x, y, radius}); }
Roi Roi::polygon(std::span<const cv::Point2f> points) { return Roi(encodePoints(RoiKind::Polygon, points)); }
Roi Roi::points(std::span<const cv::Point2f> points) { return Roi(encodePoints(RoiKind::PointSet, points)); }

std::array<cv::Point2d, 4> Roi::rectangleCorners() const {
    if (kind() != RoiKind::Rectangle) throw ContractError("Rectangle corners require a rectangle ROI");
    const auto values = encoded();
    const double halfWidth = static_cast<double>(values[3]) / 2;
    const double halfHeight = static_cast<double>(values[4]) / 2;
    const double centerX = static_cast<double>(values[1]) + halfWidth;
    const double centerY = static_cast<double>(values[2]) + halfHeight;
    const double radians = std::remainder(static_cast<double>(values[5]), 360.0) * std::numbers::pi / 180;
    const double cosine = std::cos(radians);
    const double sine = std::sin(radians);
    std::array<cv::Point2d, 4> corners{{{-halfWidth, -halfHeight}, {halfWidth, -halfHeight},
                                     {halfWidth, halfHeight}, {-halfWidth, halfHeight}}};
    for (auto& point : corners) {
        const double x = point.x;
        point.x = centerX + cosine * x - sine * point.y;
        point.y = centerY + sine * x + cosine * point.y;
    }
    return corners;
}

RoiBatch::RoiBatch() : RoiBatch(std::vector<std::vector<Roi>>{}) {}
RoiBatch::RoiBatch(std::vector<std::vector<Roi>> images)
    : images_(std::make_shared<const std::vector<std::vector<Roi>>>(std::move(images))) {}

bool isMissing(const PortValue& value) noexcept { return std::holds_alternative<Missing>(value); }

bool equalValues(const PortValue& left, const PortValue& right) {
    if (left.index() != right.index()) return false;
    return std::visit([&](const auto& a) {
        using T = std::decay_t<decltype(a)>;
        const auto& b = std::get<T>(right);
        if constexpr (std::is_same_v<T, NumericArray> || std::is_same_v<T, StringArray>) {
            if (a.rank() != b.rank() || a.layout() != b.layout() ||
                a.buffers().size() != b.buffers().size()) return false;
            if constexpr (std::is_same_v<T, NumericArray>)
                if (a.dtype() != b.dtype()) return false;
            for (std::size_t i = 0; i < a.buffers().size(); ++i) {
                const auto& first = a.buffers()[i];
                const auto& second = b.buffers()[i];
                if (!std::ranges::equal(first.shape(), second.shape())) return false;
                if constexpr (std::is_same_v<T, NumericArray>) {
                    if (first.byteSize() != second.byteSize()) return false;
                    if (first.data() != second.data() && first.byteSize() != 0 &&
                        std::memcmp(first.data(), second.data(), first.byteSize()) != 0) return false;
                } else {
                    if (!std::ranges::equal(first.values(), second.values())) return false;
                }
            }
            return true;
        } else if constexpr (std::is_same_v<T, RoiBatch>) {
            if (a.images().size() != b.images().size()) return false;
            for (std::size_t i = 0; i < a.images().size(); ++i) {
                if (a.images()[i].size() != b.images()[i].size()) return false;
                for (std::size_t j = 0; j < a.images()[i].size(); ++j)
                    if (!std::ranges::equal(a.images()[i][j].encoded(), b.images()[i][j].encoded()))
                        return false;
            }
            return true;
        } else if constexpr (std::is_same_v<T, ImageBatch>) {
            if (a.images().size() != b.images().size()) return false;
            for (std::size_t i = 0; i < a.images().size(); ++i) {
                const auto& first = a.images()[i];
                const auto& second = b.images()[i];
                if (first.data() != second.data() || first.rows() != second.rows() ||
                    first.cols() != second.cols() || first.type() != second.type() ||
                    first.rowStride() != second.rowStride()) return false;
            }
            return true;
        } else {
            return a == b;
        }
    }, left);
}

std::optional<TypeSpec> typeOf(const PortValue& value) {
    return std::visit([](const auto& item) -> std::optional<TypeSpec> {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, Missing>) return std::nullopt;
        else if constexpr (std::is_same_v<T, std::int64_t>) return TypeSpec{Type::Int};
        else if constexpr (std::is_same_v<T, double>) return TypeSpec{Type::Float};
        else if constexpr (std::is_same_v<T, bool>) return TypeSpec{Type::Bool};
        else if constexpr (std::is_same_v<T, Utf8>) return TypeSpec{Type::String};
        else if constexpr (std::is_same_v<T, ImageBatch>) return TypeSpec{Type::ImageBatch};
        else if constexpr (std::is_same_v<T, NumericArray>)
            return TypeSpec{Type::NumericArray, item.dtype(), item.rank(), item.layout()};
        else if constexpr (std::is_same_v<T, StringArray>)
            return TypeSpec{Type::StringArray, {}, item.rank(), item.layout()};
        else return TypeSpec{Type::RoiBatch};
    }, value);
}

void validateValue(const TypeSpec& expected, const PortValue& value) {
    validateTypeSpec(expected);
    const auto actual = typeOf(value);
    if (actual && *actual != expected)
        throw ContractError("Port value type/dtype/rank/layout does not match its declaration");
}

} // namespace visionrt
