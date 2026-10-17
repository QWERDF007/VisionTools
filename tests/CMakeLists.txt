include_directories("${CMAKE_SOURCE_DIR}")

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

if(VISIONTOOLS_BUILD_INFER AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_infer_contract.cpp")
    visiontools_add_test(test_infer_contract
        SOURCES test_infer_contract.cpp "${CMAKE_SOURCE_DIR}/src/infer/YoloPostprocess.cpp"
        LIBRARIES visiontools_infer VisionRT::ToolSDK ${OpenCV_LIBS}
    )
endif()
