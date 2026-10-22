include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/ConfigDependencyDefaults.cmake")

visiontools_resolve_path(_vt_cuda_root
    VARIABLES CUDAToolkit_ROOT CUDA_PATH CUDA_TOOLKIT_ROOT_DIR
    ENVIRONMENT_VARIABLES CUDA_PATH CUDAToolkit_ROOT CUDA_TOOLKIT_ROOT_DIR
    PATHS ${CMAKE_PREFIX_PATH}
)

if(_vt_cuda_root)
    set(CUDAToolkit_ROOT "${_vt_cuda_root}" CACHE PATH "CUDA Toolkit root directory" FORCE)
    set(CUDAToolkit_ROOT "${_vt_cuda_root}")
    list(PREPEND CMAKE_PREFIX_PATH "${_vt_cuda_root}")
endif()

find_package(CUDAToolkit REQUIRED)

if(EXISTS "${CUDAToolkit_TARGET_DIR}/bin")
    set(VISIONTOOLS_CUDA_BIN "${CUDAToolkit_TARGET_DIR}/bin")
elseif(EXISTS "${CUDAToolkit_ROOT}/bin")
    set(VISIONTOOLS_CUDA_BIN "${CUDAToolkit_ROOT}/bin")
endif()

file(GLOB CUDA_DLLS
    "${VISIONTOOLS_CUDA_BIN}/cudart64_*.dll"
    "${VISIONTOOLS_CUDA_BIN}/cublas64_*.dll"
    "${VISIONTOOLS_CUDA_BIN}/cublasLt64_*.dll"
    "${VISIONTOOLS_CUDA_BIN}/cufft64_*.dll"
    "${VISIONTOOLS_CUDA_BIN}/cudnn*.dll"
)
