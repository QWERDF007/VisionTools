add_library(visiontools_preprocess SHARED
    ReadImageTool.cpp
    ImageSequence.cpp
    RotateImageTool.cpp
    BrightnessTool.cpp
    RoiProjectionTool.cpp
    ImageCropTool.cpp
    ImagePreprocessTool.cpp
    DetectionFilterTool.cpp
    AdaptiveBinaryTool.cpp
    PreprocessModule.cpp
)

target_compile_definitions(visiontools_preprocess PRIVATE VISIONTOOLS_EXPORTS)
target_include_directories(visiontools_preprocess PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/src"
)

target_link_libraries(visiontools_preprocess PRIVATE
    visiontools_common
    VisionRT::Contracts
    ${OpenCV_LIBS}
)

set_target_properties(visiontools_preprocess PROPERTIES
    PREFIX ""
    OUTPUT_NAME vst_image
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
