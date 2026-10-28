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
    LIBRARIES visiontools_preprocess VisionRT::Contracts ${OpenCV_LIBS}
    DEFINITIONS VISIONTOOLS_TESTING
)

# VT-02: measured read-cache residency. It compiles ReadImageTool.cpp itself so the decode counter
# (VISIONTOOLS_TESTING) is the one the test drives.
visiontools_add_test(test_read_cache_contract
    SOURCES
        test_read_cache_contract.cpp
        "${CMAKE_SOURCE_DIR}/src/preprocess/ReadImageTool.cpp"
    LIBRARIES visiontools_preprocess VisionRT::Contracts ${OpenCV_LIBS}
    DEFINITIONS VISIONTOOLS_TESTING
    COMMAND_ARGS "${CMAKE_BINARY_DIR}/evidence"
)

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
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_infer_whitebox_trace.cpp")
        visiontools_add_test(test_infer_whitebox_trace
            SOURCES test_infer_whitebox_trace.cpp "${CMAKE_SOURCE_DIR}/src/infer/YoloPostprocess.cpp"
            LIBRARIES visiontools_infer VisionRT::Contracts ${OpenCV_LIBS} InferRT::model CUDA::cudart
        )
        set_tests_properties(test_infer_whitebox_trace PROPERTIES SKIP_RETURN_CODE 77)
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
