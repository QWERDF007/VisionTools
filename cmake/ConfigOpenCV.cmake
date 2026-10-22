include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/ConfigDependencyDefaults.cmake")

visiontools_resolve_path(_vt_opencv_home
    VARIABLES OpenCV_DIR OpenCV_HOME OpenCV_ROOT
    ENVIRONMENT_VARIABLES OpenCV_DIR OpenCV_HOME OpenCV_ROOT
    PATHS ${CMAKE_PREFIX_PATH}
    REQUIRED_FILES "OpenCVConfig.cmake" "lib/OpenCVConfig.cmake" "lib/cmake/opencv4/OpenCVConfig.cmake" "x64/vc16/lib/OpenCVConfig.cmake"
)

if(_vt_opencv_home)
    foreach(_candidate IN ITEMS
            "${_vt_opencv_home}/lib"
            "${_vt_opencv_home}/lib/cmake/opencv4"
            "${_vt_opencv_home}")
        if(EXISTS "${_candidate}/OpenCVConfig.cmake")
            set(OpenCV_DIR "${_candidate}" CACHE PATH "OpenCV CMake package directory" FORCE)
            set(OpenCV_DIR "${_candidate}")
            break()
        endif()
    endforeach()
    list(PREPEND CMAKE_PREFIX_PATH "${_vt_opencv_home}")
endif()

find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs)

# 解析 OpenCV 运行时 DLL 目录
if(EXISTS "${OpenCV_DIR}/../bin")
    file(REAL_PATH "${OpenCV_DIR}/../bin" VISIONTOOLS_OPENCV_BIN)
elseif(EXISTS "${OpenCV_DIR}/bin")
    file(REAL_PATH "${OpenCV_DIR}/bin" VISIONTOOLS_OPENCV_BIN)
elseif(DEFINED _vt_opencv_home AND EXISTS "${_vt_opencv_home}/bin")
    file(REAL_PATH "${_vt_opencv_home}/bin" VISIONTOOLS_OPENCV_BIN)
endif()

set(VISIONTOOLS_OPENCV_RELEASE_DLL "")
set(VISIONTOOLS_OPENCV_DEBUG_DLL "")

if(WIN32)
    if(EXISTS "${VISIONTOOLS_OPENCV_BIN}")
        file(GLOB _opencv_world_rel "${VISIONTOOLS_OPENCV_BIN}/opencv_world[0-9]*.dll")
        file(GLOB _opencv_world_dbg "${VISIONTOOLS_OPENCV_BIN}/opencv_world[0-9]*d.dll")

        # Exclude debug from release list
        foreach(_cand IN LISTS _opencv_world_rel)
            if(NOT _cand MATCHES "d\\.dll$")
                set(VISIONTOOLS_OPENCV_RELEASE_DLL "${_cand}")
                break()
            endif()
        endforeach()

        if(_opencv_world_dbg)
            list(GET _opencv_world_dbg 0 VISIONTOOLS_OPENCV_DEBUG_DLL)
        endif()

        # If opencv_world not present, fall back to core component
        if(NOT VISIONTOOLS_OPENCV_RELEASE_DLL)
            file(GLOB _opencv_core_rel "${VISIONTOOLS_OPENCV_BIN}/opencv_core[0-9]*.dll")
            foreach(_cand IN LISTS _opencv_core_rel)
                if(NOT _cand MATCHES "d\\.dll$")
                    set(VISIONTOOLS_OPENCV_RELEASE_DLL "${_cand}")
                    break()
                endif()
            endforeach()
            file(GLOB _opencv_core_dbg "${VISIONTOOLS_OPENCV_BIN}/opencv_core[0-9]*d.dll")
            if(_opencv_core_dbg)
                list(GET _opencv_core_dbg 0 VISIONTOOLS_OPENCV_DEBUG_DLL)
            endif()
        endif()
    endif()

    if(NOT VISIONTOOLS_OPENCV_RELEASE_DLL)
        message(FATAL_ERROR "Required OpenCV runtime DLL not found in ${VISIONTOOLS_OPENCV_BIN}")
    endif()
endif()
