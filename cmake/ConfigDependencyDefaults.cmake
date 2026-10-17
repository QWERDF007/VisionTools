include_guard(GLOBAL)

# 统一依赖路径解析单一真相源 (SSOT)
# 优先级：CMake 显式变量 > 环境变量 > CMAKE_PREFIX_PATH 候选匹配 > 默认路径
function(visiontools_resolve_path output_var)
    cmake_parse_arguments(ARG "" "DEFAULT;REQUIRED_FILE" "VARIABLES;ENVIRONMENT_VARIABLES;PATHS" ${ARGN})

    set(_candidates)

    # 1. CMake 变量（包括命令行 -D 传入）
    foreach(_var IN LISTS ARG_VARIABLES)
        if(DEFINED ${_var} AND NOT "${${_var}}" STREQUAL "")
            file(TO_CMAKE_PATH "${${_var}}" _var_norm)
            list(APPEND _candidates "${_var_norm}")
        endif()
    endforeach()

    # 2. 操作系统环境变量 (将 Windows 反斜杠规范化为 CMake 正斜杠)
    foreach(_env IN LISTS ARG_ENVIRONMENT_VARIABLES)
        if(DEFINED ENV{${_env}} AND NOT "$ENV{${_env}}" STREQUAL "")
            file(TO_CMAKE_PATH "$ENV{${_env}}" _env_norm)
            list(APPEND _candidates "${_env_norm}")
        endif()
    endforeach()

    # 3. 候选路径匹配
    foreach(_p IN LISTS ARG_PATHS)
        if(_p)
            file(TO_CMAKE_PATH "${_p}" _p_norm)
            list(APPEND _candidates "${_p_norm}")
        endif()
    endforeach()

    # 4. 平台默认基准路径
    if(ARG_DEFAULT)
        file(TO_CMAKE_PATH "${ARG_DEFAULT}" _def_norm)
        list(APPEND _candidates "${_def_norm}")
    endif()

    foreach(_cand IN LISTS _candidates)
        if(EXISTS "${_cand}")
            if(ARG_REQUIRED_FILE)
                if(EXISTS "${_cand}/${ARG_REQUIRED_FILE}")
                    set(${output_var} "${_cand}" PARENT_SCOPE)
                    return()
                endif()
            else()
                set(${output_var} "${_cand}" PARENT_SCOPE)
                return()
            endif()
        endif()
    endforeach()

    set(${output_var} "" PARENT_SCOPE)
endfunction()
