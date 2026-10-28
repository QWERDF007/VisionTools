#pragma once

#include "common/errors/ToolError.h"

#include <visionrt/contracts/Parameters.h>
#include <visionrt/contracts/Tool.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace visiontools {
namespace detail {

// A failed read still has to return something of the requested type. The value is never read: the
// tool returns as soon as it sees the issue. DataRef has no empty state, so it gets an empty payload.
template<class T>
struct Placeholder
{
    static T make()
    {
        return T{};
    }
};

template<class T>
struct Placeholder<visionrt::DataRef<T>>
{
    static visionrt::DataRef<T> make()
    {
        return visionrt::DataRef<T>(T{});
    }
};

} // namespace detail

// Typed reads of one execution's inputs. A failed read is remembered once and the tool asks for it
// right after the read block, before any algorithm runs; the accessors stay terse so a tool keeps
// its own structure instead of repeating an error branch per field.
//
// The returned value of a failed read is never a usable value: it is the type's default and the
// tool must have returned already. Nothing here throws.
class Readings
{
public:
    explicit Readings(const visionrt::Values &values) noexcept
        : reader_(values)
    {
    }

    explicit Readings(const visionrt::InputReader &reader) noexcept
        : reader_(reader)
    {
    }

    const visionrt::Values &values() const noexcept
    {
        return reader_.values();
    }

    bool present(std::string_view id) const noexcept
    {
        return reader_.present(id);
    }

    const visionrt::PortValue *find(std::string_view id) const noexcept
    {
        return reader_.find(id);
    }

    std::int64_t integer(std::string_view id)
    {
        return take(reader_.integer(id));
    }

    double real(std::string_view id)
    {
        return take(reader_.real(id));
    }

    bool boolean(std::string_view id)
    {
        return take(reader_.boolean(id));
    }

    std::string text(std::string_view id)
    {
        return take(reader_.text(id));
    }

    visionrt::DataRef<visionrt::Image> image(std::string_view id)
    {
        return take(reader_.image(id));
    }

    visionrt::DataRef<visionrt::ROI> roi(std::string_view id)
    {
        return take(reader_.roi(id));
    }

    std::optional<std::int64_t> optionalInteger(std::string_view id)
    {
        return take(reader_.optionalInteger(id));
    }

    std::optional<double> optionalReal(std::string_view id)
    {
        return take(reader_.optionalReal(id));
    }

    std::optional<bool> optionalBoolean(std::string_view id)
    {
        return take(reader_.optionalBoolean(id));
    }

    std::optional<std::string> optionalText(std::string_view id)
    {
        return take(reader_.optionalText(id));
    }

    std::optional<visionrt::DataRef<visionrt::Image>> nullableImage(std::string_view id)
    {
        return take(reader_.nullableImage(id));
    }

    std::optional<visionrt::DataRef<visionrt::ROI>> nullableRoi(std::string_view id)
    {
        return take(reader_.nullableRoi(id));
    }

    template<class T>
    visionrt::DataRef<T> array(std::string_view id)
    {
        return take(reader_.template array<T>(id));
    }

    template<class T>
    std::optional<visionrt::DataRef<T>> nullableArray(std::string_view id)
    {
        return take(reader_.template nullableArray<T>(id));
    }

    // The first failed read, or nothing when every read succeeded.
    bool failed() const noexcept
    {
        return issue_.has_value();
    }

    const visionrt::PortIssue *issue() const noexcept
    {
        return issue_ ? &*issue_ : nullptr;
    }

    // The status a tool reports for its first failed read, in that tool's own codes.
    template<class MissingCode, class TypeCode>
        requires std::is_enum_v<MissingCode> && std::is_enum_v<TypeCode>
    visionrt::ToolStatus status(MissingCode missingCode, TypeCode typeCode) const
    {
        return issue_ ? inputIssueStatus(*issue_, static_cast<std::int64_t>(missingCode),
                                         static_cast<std::int64_t>(typeCode))
                      : visionrt::ToolStatus{};
    }

private:
    template<class T>
    T take(visionrt::ReadResult<T> &&result)
    {
        if (!result)
        {
            if (!issue_)
                issue_ = result.issue();
            return detail::Placeholder<T>::make();
        }
        return std::move(result.take());
    }

    visionrt::InputReader       reader_;
    std::optional<visionrt::PortIssue> issue_;
};

} // namespace visiontools
