set(_error_source "${PROJECT_SOURCE_DIR}/python/author_sdk/visiontools_sdk/errors.json")
file(READ "${_error_source}" _errors)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_error_source}")
set(_header "#pragma once
#include <string>
#include <string_view>
#include <cstdint>
#include <visionrt/contracts/Tool.h>
namespace visiontools {
enum class CommonToolError : std::int64_t {
")
string(JSON _count LENGTH "${_errors}")
math(EXPR _last "${_count} - 1")
foreach(_i RANGE 0 ${_last})
 string(JSON _name GET "${_errors}" ${_i} name)
 string(JSON _code GET "${_errors}" ${_i} code)
 string(APPEND _header "${_name} = ${_code},
")
endforeach()
string(APPEND _header "};
inline std::string_view commonErrorMessage(std::int64_t code) noexcept { switch(code) {
")
foreach(_i RANGE 0 ${_last})
 string(JSON _code GET "${_errors}" ${_i} code)
 string(JSON _message GET "${_errors}" ${_i} message)
 string(APPEND _header "case ${_code}: return \"${_message}\";
")
endforeach()
string(APPEND _header "} return \"工具执行失败\"; }
")
file(GLOB_RECURSE _definitions CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/assets/tools/*.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_definitions})
string(APPEND _header "inline std::string parameterName(std::string_view tool, std::string_view stage, visionrt::PortId id) {
")
foreach(_file IN LISTS _definitions)
 file(READ "${_file}" _definition)
 string(JSON _tool GET "${_definition}" name_en)
 string(APPEND _header "if (tool == \"${_tool}\") {
")
 foreach(_stage init_params inputs outputs)
  string(JSON _length ERROR_VARIABLE _error LENGTH "${_definition}" ${_stage})
  if(NOT _error AND _length GREATER 0)
   math(EXPR _end "${_length} - 1")
   string(APPEND _header "if (stage == \"${_stage}\") { switch(id) {
")
   foreach(_j RANGE 0 ${_end})
    string(JSON _id GET "${_definition}" ${_stage} ${_j} id)
    string(JSON _name GET "${_definition}" ${_stage} ${_j} name_zh)
    string(APPEND _header "case ${_id}: return \"${_name}\";
")
   endforeach()
   string(APPEND _header "} }
")
  endif()
 endforeach()
 string(APPEND _header "}
")
endforeach()
string(APPEND _header "return \"未声明的参数（编号 \" + std::to_string(id) + \"）\"; }
}
")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/generated/common/errors")
file(CONFIGURE OUTPUT "${PROJECT_BINARY_DIR}/generated/common/errors/ToolMetadata.h" CONTENT "${_header}" @ONLY)
