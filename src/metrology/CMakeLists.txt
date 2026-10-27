add_library(visiontools_metrology SHARED
    EdgeProfile.cpp
    CircleFit.cpp
    CircleMeasureKernel.cpp
    CircleTool.cpp
    RowSpacingKernel.cpp
    RowSpacingTool.cpp
    AssistSupport.cpp
    AssistTools.cpp
    MetrologyModule.cpp
)

target_compile_definitions(visiontools_metrology PRIVATE VISIONTOOLS_EXPORTS)
target_include_directories(visiontools_metrology PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(visiontools_metrology PRIVATE
    visiontools_common
    VisionRT::Contracts
    ${OpenCV_LIBS}
)

set_target_properties(visiontools_metrology PROPERTIES
    PREFIX ""
    OUTPUT_NAME vst_metrology
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
