#pragma once

#include "common/errors/ToolError.h"

#include <visionrt/contracts/Parameters.h>
#include <visionrt/contracts/Declaration.h>
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
    explicit Readings(const visionrt::Values &values, std::string_view tool, visionrt::PortStage stage = visionrt::PortStage::Inputs) noexcept
        : reader_(values), tool_(tool), stage_(stage)
    {
    }

    explicit Readings(const visionrt::InputReader &reader, std::string_view tool, visionrt::PortStage stage = visionrt::PortStage::Inputs) noexcept
        : reader_(reader), tool_(tool), stage_(stage)
    {
    }

    const visionrt::Values &values() const noexcept
    {
        return reader_.values();
    }

    bool present(visionrt::PortId id) const noexcept
    {
        return reader_.present(id);
    }

    const visionrt::PortValue *find(visionrt::PortId id) const noexcept
    {
        return reader_.find(id);
    }

    std::int64_t integer(visionrt::PortId id)
    {
        return take(reader_.integer(id), id);
    }

    double real(visionrt::PortId id)
    {
        return take(reader_.real(id), id);
    }

    bool boolean(visionrt::PortId id)
    {
        return take(reader_.boolean(id), id);
    }

    std::string text(visionrt::PortId id)
    {
        return take(reader_.text(id), id);
    }

    visionrt::DataRef<visionrt::Image> image(visionrt::PortId id)
    {
        return take(reader_.image(id), id);
    }

    template<class Shape> requires visionrt::isShapeValue<Shape>
    Shape shape(visionrt::PortId id) { return take(reader_.shape<Shape>(id), id); }

#define VRT_SHAPE(name, kind, view, code, wire, label) \
    visionrt::name wire(visionrt::PortId id) { return shape<visionrt::name>(id); } \
    std::optional<visionrt::name> nullable##name(visionrt::PortId id) { return take(reader_.nullable##name(id), id); }
#include <visionrt/contracts/ShapeKinds.def>
#undef VRT_SHAPE

    visionrt::DataRef<visionrt::TrainingSet> trainingSet(visionrt::PortId id)
    {
        return take(reader_.trainingSet(id), id);
    }

    std::optional<visionrt::DataRef<visionrt::Image>> nullableImage(visionrt::PortId id)
    {
        return take(reader_.nullableImage(id), id);
    }

    visionrt::Array array(visionrt::PortId id)
    {
        return take(reader_.array(id), id);
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
    T take(visionrt::ReadResult<T> &&result, visionrt::PortId id)
    {
        if (!result)
        {
            if (!issue_)
            {
                issue_ = result.issue();
                issue_->field = parameterName(tool_, visionrt::stageName(stage_), id);
            }
            return detail::Placeholder<T>::make();
        }
        return std::move(result.take());
    }

    visionrt::InputReader       reader_;
    std::string_view tool_;
    visionrt::PortStage stage_;
    std::optional<visionrt::PortIssue> issue_;
};

} // namespace visiontools
