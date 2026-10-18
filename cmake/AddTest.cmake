include_guard(GLOBAL)

# 测试所需的 DLL 搜索路径（C++ 测试与 Python 测试共用，全部来自各 Config*.cmake 的解析结果，无机器相关硬编码）
function(visiontools_test_path out_var)
    set(_paths
        "${CMAKE_BINARY_DIR}/src/preprocess"
        "${CMAKE_BINARY_DIR}/src/infer"
        "${VISIONTOOLS_OPENCV_BIN}"
    )
    if(DEFINED INFERRT_ROOT AND EXISTS "${INFERRT_ROOT}/bin")
        list(APPEND _paths "${INFERRT_ROOT}/bin")
    endif()
    if(DEFINED CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/bin")
        list(APPEND _paths "${CUDAToolkit_ROOT}/bin")
    endif()
    if(DEFINED TRT_ROOT AND EXISTS "${TRT_ROOT}/bin")
        list(APPEND _paths "${TRT_ROOT}/bin")
    endif()
    if(DEFINED VISIONTOOLS_PYTHON_HOME AND EXISTS "${VISIONTOOLS_PYTHON_HOME}/Library/bin")
        list(APPEND _paths "${VISIONTOOLS_PYTHON_HOME}/Library/bin")
    endif()
    string(REPLACE ";" "\\;" _escaped "${_paths};$ENV{PATH}")
    set(${out_var} "${_escaped}" PARENT_SCOPE)
endfunction()

function(visiontools_add_test test_name)
    set(options "")
    set(oneValueArgs WORKING_DIRECTORY)
    set(multiValueArgs SOURCES LIBRARIES DEFINITIONS COMMAND_ARGS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_executable(${test_name} ${ARG_SOURCES})
    target_link_libraries(${test_name} PRIVATE ${ARG_LIBRARIES})
    if(ARG_DEFINITIONS)
        target_compile_definitions(${test_name} PRIVATE ${ARG_DEFINITIONS})
    endif()

    if(ARG_WORKING_DIRECTORY)
        add_test(NAME ${test_name} COMMAND ${test_name} ${ARG_COMMAND_ARGS} WORKING_DIRECTORY ${ARG_WORKING_DIRECTORY})
    else()
        add_test(NAME ${test_name} COMMAND ${test_name} ${ARG_COMMAND_ARGS})
    endif()

    # 在 Windows 上设置运行测试所需的 DLL PATH 环境 (VS-06 / VS-08)
    if(WIN32)
        visiontools_test_path(_test_paths)
        set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "PATH=${_test_paths}")
    endif()
endfunction()
