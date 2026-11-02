include_directories(
    "${CMAKE_SOURCE_DIR}"
)

visiontools_add_test(test_writable_tools_contract
    SOURCES test_writable_tools_contract.cpp
    LIBRARIES visiontools_preprocess VisionRT::Contracts ${OpenCV_LIBS}
)
if(MSVC)
    target_compile_options(test_writable_tools_contract PRIVATE /UNDEBUG)
else()
    target_compile_options(test_writable_tools_contract PRIVATE -UNDEBUG)
endif()

visiontools_add_test(test_upstream_status_contract
    SOURCES test_upstream_status_contract.cpp
    LIBRARIES visiontools_preprocess VisionRT::Contracts ${OpenCV_LIBS}
)
if(MSVC)
    target_compile_options(test_upstream_status_contract PRIVATE /UNDEBUG)
else()
    target_compile_options(test_upstream_status_contract PRIVATE -UNDEBUG)
endif()

visiontools_add_test(test_preprocess_contract
    SOURCES
        test_preprocess_contract.cpp
        preprocess/adaptive_binary.cpp
        preprocess/roi_projection.cpp
        preprocess/image_crop.cpp
        preprocess/detection_filter.cpp
        preprocess/image_preprocess.cpp
        preprocess/read_image.cpp
        "${CMAKE_SOURCE_DIR}/src/preprocess/ReadImageTool.cpp"
        "${CMAKE_SOURCE_DIR}/src/preprocess/RoiProjectionTool.cpp"
        "${CMAKE_SOURCE_DIR}/src/preprocess/ImageCropTool.cpp"
        "${CMAKE_SOURCE_DIR}/src/preprocess/ImagePreprocessTool.cpp"
        "${CMAKE_SOURCE_DIR}/src/preprocess/DetectionFilterTool.cpp"
        "${CMAKE_SOURCE_DIR}/src/preprocess/AdaptiveBinaryTool.cpp"
    # The tool sources are compiled here as well, so the shared error vocabulary they call into
    # has to be linked explicitly: the module links it privately.
    LIBRARIES visiontools_preprocess visiontools_common VisionRT::Contracts ${OpenCV_LIBS}
    DEFINITIONS VISIONTOOLS_TESTING
)

# VT-02: cache correctness runs on three small frames; the residency measurement needs a stated
# 1600x900 batch and a long loop, so it is a second entry over the same binary. It compiles
# ReadImageTool.cpp itself so the decode counter (VISIONTOOLS_TESTING) is the one the test drives.
visiontools_add_test(image_read_cache_test
    SOURCES
        image_read_cache_test.cpp
        "${CMAKE_SOURCE_DIR}/src/preprocess/ReadImageTool.cpp"
    LIBRARIES visiontools_preprocess visiontools_common VisionRT::Contracts ${OpenCV_LIBS}
    DEFINITIONS VISIONTOOLS_TESTING
)
add_test(NAME image_read_cache_benchmark
    COMMAND image_read_cache_test --perf --evidence "${CMAKE_BINARY_DIR}/evidence")
visiontools_test_path(_vt_cache_benchmark_paths)
set_tests_properties(image_read_cache_benchmark PROPERTIES
    ENVIRONMENT "PATH=${_vt_cache_benchmark_paths}" LABELS "perf")

# VT-03: proof that rotate/brightness allocate their own pixels before dropping the input owner.
visiontools_add_test(test_output_ownership_contract
    SOURCES test_output_ownership_contract.cpp
    LIBRARIES visiontools_preprocess VisionRT::Contracts ${OpenCV_LIBS}
)

visiontools_add_test(test_geometry_contract
    SOURCES test_geometry_contract.cpp
    LIBRARIES visiontools_metrology visiontools_preprocess visiontools_common VisionRT::Contracts ${OpenCV_LIBS}
)

visiontools_add_test(test_module_contract
    SOURCES test_module_contract.cpp
    LIBRARIES visiontools_preprocess VisionRT::Contracts
)

visiontools_add_test(test_installation_consumer
    SOURCES test_installation_consumer.cpp
    LIBRARIES VisionRT::Contracts ${OpenCV_LIBS}
    COMMAND_ARGS "$<TARGET_FILE:visiontools_preprocess>"
)

# 纯后处理回归始终构建：只编译 YoloPostprocess.cpp，不依赖 InferRT/CUDA/推理模块。
visiontools_add_test(test_yolo_postprocess
    SOURCES test_yolo_postprocess.cpp "${CMAKE_SOURCE_DIR}/src/infer/YoloPostprocess.cpp"
    LIBRARIES VisionRT::Contracts
)

if(VISIONTOOLS_BUILD_INFER AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_infer_contract.cpp")
    visiontools_add_test(test_infer_contract
        SOURCES test_infer_contract.cpp "${CMAKE_SOURCE_DIR}/src/infer/YoloPostprocess.cpp"
        LIBRARIES visiontools_infer VisionRT::Contracts ${OpenCV_LIBS}
    )
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/infer_trace_test.cpp")
        # Trace capabilities are separate entries: geometry and the uninitialized path always run,
        # while the initialized trace needs a GPU/model fixture and may skip on its own.
        visiontools_add_test(infer_trace_test
            SOURCES infer_trace_test.cpp "${CMAKE_SOURCE_DIR}/src/infer/YoloPostprocess.cpp"
            LIBRARIES visiontools_infer VisionRT::Contracts ${OpenCV_LIBS} InferRT::model CUDA::cudart
            NO_TEST
        )
        visiontools_test_path(_vt_infer_trace_paths)
        foreach(_case IN ITEMS geometry uninitialized)
            add_test(NAME infer_trace_${_case} COMMAND infer_trace_test --case ${_case})
            set_tests_properties(infer_trace_${_case} PROPERTIES
                ENVIRONMENT "PATH=${_vt_infer_trace_paths}" LABELS "smoke")
        endforeach()
        add_test(NAME infer_trace_initialized COMMAND infer_trace_test --case initialized-trace)
        set_tests_properties(infer_trace_initialized PROPERTIES
            ENVIRONMENT "PATH=${_vt_infer_trace_paths}" LABELS "gpu" SKIP_RETURN_CODE 77)
    endif()
endif()

if(VISIONTOOLS_BUILD_PYTHON)
    foreach(_test IN ITEMS package package_payload package_profile python_tools distribution logic_engine)
        add_test(NAME test_${_test}
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tests/test_${_test}.py"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endforeach()
    visiontools_test_path(_vt_test_paths)
    set_tests_properties(test_package test_package_payload test_package_profile test_distribution test_python_tools test_logic_engine PROPERTIES
        ENVIRONMENT "PATH=${_vt_test_paths};VISIONTOOLS_VALIDATOR=${VISIONTOOLS_VALIDATOR};PYTHONDONTWRITEBYTECODE=1")
endif()
