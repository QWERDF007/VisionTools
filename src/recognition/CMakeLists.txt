add_library(visiontools_recognition SHARED
    TrainingAssets.cpp
    StraightEdgeLocator.cpp
    ColorModel.cpp
    StraightEdgeProbe.cpp
    AlignStraightBordersTool.cpp
    RecognizeColorTool.cpp
    PickStraightEdgeTool.cpp
    RecognitionModule.cpp
)

target_compile_definitions(visiontools_recognition PRIVATE VISIONTOOLS_EXPORTS)
target_include_directories(visiontools_recognition PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
# Header-only JSON: training assets are plain files the host writes, so the tool reads the same
# document the training UI edits instead of receiving samples across the ABI.
target_include_directories(visiontools_recognition PRIVATE
    "${CMAKE_SOURCE_DIR}/3rdparty"
)

target_link_libraries(visiontools_recognition PRIVATE
    visiontools_common
    VisionRT::Contracts
    ${OpenCV_LIBS}
)

set_target_properties(visiontools_recognition PROPERTIES
    PREFIX ""
    OUTPUT_NAME vst_recognition
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
