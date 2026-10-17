add_library(visiontools_infer SHARED
    YoloPostprocess.cpp
    YoloModel.cpp
    InferModule.cpp
)

target_compile_definitions(visiontools_infer PRIVATE VISIONTOOLS_EXPORTS)
if(DEFINED InferRT_VERSION)
    target_compile_definitions(visiontools_infer PRIVATE VISIONTOOLS_INFERRT_VERSION="${InferRT_VERSION}")
endif()

target_include_directories(visiontools_infer PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../generated"
)

target_link_libraries(visiontools_infer PRIVATE
    VisionRT::ToolSDK
    InferRT::model
    CUDA::cudart
    ${OpenCV_LIBS}
)

set_target_properties(visiontools_infer PROPERTIES
    PREFIX ""
    OUTPUT_NAME visiontools_infer
)
