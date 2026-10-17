include_guard(GLOBAL)

function(visiontools_add_test test_name)
    set(options "")
    set(oneValueArgs WORKING_DIRECTORY)
    set(multiValueArgs SOURCES LIBRARIES DEFINITIONS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_executable(${test_name} ${ARG_SOURCES})
    target_link_libraries(${test_name} PRIVATE ${ARG_LIBRARIES})
    if(ARG_DEFINITIONS)
        target_compile_definitions(${test_name} PRIVATE ${ARG_DEFINITIONS})
    endif()

    if(ARG_WORKING_DIRECTORY)
        add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${ARG_WORKING_DIRECTORY})
    else()
        add_test(NAME ${test_name} COMMAND ${test_name})
    endif()

    # 在 Windows 上设置运行测试所需的 DLL PATH 环境 (VS-06 / VS-08)
    if(WIN32)
        set(_test_paths
            "${CMAKE_BINARY_DIR}/src/preprocess"
            "${CMAKE_BINARY_DIR}/src/infer"
            "${VISIONTOOLS_OPENCV_BIN}"
        )
        if(DEFINED INFERRT_ROOT AND EXISTS "${INFERRT_ROOT}/bin")
            list(APPEND _test_paths "${INFERRT_ROOT}/bin")
        endif()
        if(DEFINED CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/bin")
            list(APPEND _test_paths "${CUDAToolkit_ROOT}/bin")
        endif()
        if(DEFINED TRT_ROOT AND EXISTS "${TRT_ROOT}/bin")
            list(APPEND _test_paths "${TRT_ROOT}/bin")
        endif()

        # 转换为 Windows 路径分隔符分号列表
        string(REPLACE ";" "\\;" _escaped_paths "${_test_paths}")
        set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "PATH=${_escaped_paths};$ENV{PATH}")
    endif()
endfunction()
