include_directories(
    "${CMAKE_SOURCE_DIR}"
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
    LIBRARIES visiontools_preprocess VisionRT::Contracts ${OpenCV_LIBS}
    DEFINITIONS VISIONTOOLS_TESTING
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
    foreach(_test IN ITEMS package python_tools distribution)
        add_test(NAME test_${_test}
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tests/test_${_test}.py"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endforeach()
    visiontools_test_path(_vt_test_paths)
    set_tests_properties(test_package test_distribution test_python_tools PROPERTIES
        ENVIRONMENT "PATH=${_vt_test_paths};VISIONTOOLS_VALIDATOR=${VISIONTOOLS_VALIDATOR}")
endif()
