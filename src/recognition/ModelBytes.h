#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

// The byte plumbing every published model shares: little-endian integers, IEEE doubles and a reader
// that refuses to walk past the end of the buffer. A model keeps its own magic, version and failure
// code; this header only decides how a number is spelled.
namespace visiontools::recognition::modelbytes {

// Thrown when a model file ends early or holds a length that cannot be honoured. The message names
// the field, so the tool can report it with its own "model is damaged" code.
struct Corrupt
{
    std::string detail;
};

inline void putU16(std::vector<std::uint8_t> &bytes, std::uint16_t value)
{
    for (int shift = 0; shift < 16; shift += 8) bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

inline void putU32(std::vector<std::uint8_t> &bytes, std::uint32_t value)
{
    for (int shift = 0; shift < 32; shift += 8) bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

inline void putU64(std::vector<std::uint8_t> &bytes, std::uint64_t value)
{
    for (int shift = 0; shift < 64; shift += 8) bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

inline void putDouble(std::vector<std::uint8_t> &bytes, double value)
{
    putU64(bytes, std::bit_cast<std::uint64_t>(value));
}

inline void putText(std::vector<std::uint8_t> &bytes, const std::string &text, std::size_t limit)
{
    if (text.size() > limit)
        throw Corrupt{"文本字段过长"};
    putU16(bytes, static_cast<std::uint16_t>(text.size()));
    bytes.insert(bytes.end(), text.begin(), text.end());
}

class Reader
{
public:
    explicit Reader(std::span<const std::uint8_t> bytes)
        : bytes_(bytes)
    {
    }

    const std::uint8_t *take(std::size_t count)
    {
        if (count > bytes_.size() - offset_)
            throw Corrupt{"长度不足"};
        const auto *start = bytes_.data() + offset_;
        offset_ += count;
        return start;
    }

    std::uint16_t u16()
    {
        const auto *data = take(2);
        return static_cast<std::uint16_t>(data[0] | (data[1] << 8));
    }

    std::uint32_t u32()
    {
        const auto   *data  = take(4);
        std::uint32_t value = 0;
        for (int index = 3; index >= 0; --index) value = (value << 8) | data[index];
        return value;
    }

    std::uint64_t u64()
    {
        const auto   *data  = take(8);
        std::uint64_t value = 0;
        for (int index = 7; index >= 0; --index) value = (value << 8) | data[index];
        return value;
    }

    double f64()
    {
        return std::bit_cast<double>(u64());
    }

    std::string text(std::size_t limit)
    {
        const auto length = u16();
        if (length > limit)
            throw Corrupt{"文本字段过长"};
        const auto *data = take(length);
        return {reinterpret_cast<const char *>(data), length};
    }

    std::size_t remaining() const noexcept
    {
        return bytes_.size() - offset_;
    }

private:
    std::span<const std::uint8_t> bytes_;
    std::size_t                   offset_ = 0;
};

} // namespace visiontools::recognition::modelbytes
