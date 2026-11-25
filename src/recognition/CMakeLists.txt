add_library(visiontools_recognition SHARED
    StraightEdgeLocator.cpp
    StraightEdgeModel.cpp
    ColorModel.cpp
    TrainColorModel.cpp
    TrainStraightBorders.cpp
    StraightEdgeProbe.cpp
    AlignStraightBordersTool.cpp
    RecognizeColorTool.cpp
    LocateStraightEdgeTool.cpp
    TemplateMatcher.cpp
    AlignByMatchingTool.cpp
    CheckPresenceTool.cpp
    RecognitionModule.cpp
)

target_compile_definitions(visiontools_recognition PRIVATE VISIONTOOLS_EXPORTS)
target_include_directories(visiontools_recognition PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
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
