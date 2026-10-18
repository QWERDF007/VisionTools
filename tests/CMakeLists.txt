include_directories(
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/generated"
)

visiontools_add_test(test_preprocess_contract
    SOURCES
        test_preprocess_contract.cpp
        "${CMAKE_SOURCE_DIR}/src/preprocess/ReadImageTool.cpp"
        "${CMAKE_SOURCE_DIR}/src/preprocess/RoiProjectionTool.cpp"
        "${CMAKE_SOURCE_DIR}/src/preprocess/ImageCropTool.cpp"
        "${CMAKE_SOURCE_DIR}/src/preprocess/ImagePreprocessTool.cpp"
        "${CMAKE_SOURCE_DIR}/src/preprocess/DetectionFilterTool.cpp"
        "${CMAKE_SOURCE_DIR}/src/preprocess/AdaptiveBinaryTool.cpp"
    LIBRARIES visiontools_preprocess VisionRT::ToolSDK ${OpenCV_LIBS}
)

visiontools_add_test(test_module_contract
    SOURCES test_module_contract.cpp
    LIBRARIES visiontools_preprocess VisionRT::ToolSDK
)

visiontools_add_test(test_installation_consumer
    SOURCES test_installation_consumer.cpp
    LIBRARIES VisionRT::ToolSDK ${OpenCV_LIBS} yaml-cpp::yaml-cpp
    COMMAND_ARGS "$<TARGET_FILE:visiontools_preprocess>"
)

if(VISIONTOOLS_BUILD_INFER AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_infer_contract.cpp")
    visiontools_add_test(test_infer_contract
        SOURCES test_infer_contract.cpp "${CMAKE_SOURCE_DIR}/src/infer/YoloPostprocess.cpp"
        LIBRARIES visiontools_infer VisionRT::ToolSDK ${OpenCV_LIBS}
    )
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_infer_whitebox_trace.cpp")
        visiontools_add_test(test_infer_whitebox_trace
            SOURCES test_infer_whitebox_trace.cpp "${CMAKE_SOURCE_DIR}/src/infer/YoloPostprocess.cpp"
            LIBRARIES visiontools_infer VisionRT::ToolSDK ${OpenCV_LIBS} InferRT::model CUDA::cudart
        )
        set_tests_properties(test_infer_whitebox_trace PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

# 注册 Python 测试到 CTest (VS-02 / VS-06 / VS-07 / VS-08)
add_test(
    NAME test_generator
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tests/test_generator.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_test(
    NAME test_package
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tests/test_package.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_test(
    NAME test_python_tools
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tests/test_python_tools.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

