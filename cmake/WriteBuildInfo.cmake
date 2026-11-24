# 生成组件构建元数据 build_info.json。
#
# 由构建完成事件（POST_BUILD / 构建结束目标）调用：配置时刻早于真正的编译链接，不能冒充构建
# 时刻。时刻来源按优先级：
#   SOURCE_FILE    指定产物：使用该文件的最后修改时间，未重新链接的组件因此保留上一次真实构建时刻；
#   SCAN_DIR       扫描目录内全部 .dll/.exe/.pyd，取最新产物的修改时间（工具包多模块场景）；
#   都未给出       使用当前时刻（刚刚完成构建的目标）。
#
# 参数：
#   COMPONENT      组件名（写入 JSON）
#   VERSION        版本号
#   OUTPUT         目标 build_info.json
#   SOURCE_FILE    可选，单个构建产物
#   SCAN_DIR       可选，产物所在目录
#   REPO_DIR       可选，仓库目录；构建完成时实际查询当前 git commit 与 branch，防止配置期缓存滞后
#   RELEASE_BUILD  可选，发布模式（ON 时遇未知状态或脏工作区报错中断）
if(NOT DEFINED COMPONENT OR NOT DEFINED OUTPUT)
    message(FATAL_ERROR "WriteBuildInfo.cmake 需要 COMPONENT 与 OUTPUT")
endif()

# Strip any accidental surrounding quotes from paths (VERBATIM escaping)
string(REGEX REPLACE "^\"|\"$" "" OUTPUT "${OUTPUT}")
if(DEFINED REPO_DIR)
    string(REGEX REPLACE "^\"|\"$" "" REPO_DIR "${REPO_DIR}")
endif()
if(DEFINED SOURCE_FILE)
    string(REGEX REPLACE "^\"|\"$" "" SOURCE_FILE "${SOURCE_FILE}")
endif()
if(DEFINED SCAN_DIR)
    string(REGEX REPLACE "^\"|\"$" "" SCAN_DIR "${SCAN_DIR}")
endif()

if(NOT GIT_EXECUTABLE)
    find_program(GIT_EXECUTABLE NAMES git git.exe
        PATHS
            "D:/Software/Git/cmd"
            "C:/Program Files/Git/cmd"
            "C:/Program Files (x86)/Git/cmd"
    )
endif()
if(NOT GIT_EXECUTABLE)
    set(GIT_EXECUTABLE git)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../3rdparty/VisionRTContracts/cmake/GitWorkspace.cmake")
visionrt_git_workspace("${REPO_DIR}" "${GIT_EXECUTABLE}" _workspace)
string(JSON COMMIT GET "${_workspace}" commit)
string(JSON BRANCH GET "${_workspace}" branch)
string(JSON _dirty GET "${_workspace}" dirty)
if(_dirty STREQUAL "unknown")
    set(IS_DIRTY_VALUE "\"unknown\"")
elseif(_dirty)
    set(IS_DIRTY_VALUE true)
else()
    set(IS_DIRTY_VALUE false)
endif()
if(RELEASE_BUILD AND (_dirty STREQUAL "unknown" OR COMMIT STREQUAL "unknown" OR BRANCH STREQUAL "unknown"))
    message(FATAL_ERROR "WriteBuildInfo: 发布模式禁止未知候选（${REPO_DIR}）")
elseif(RELEASE_BUILD AND _dirty)
    message(FATAL_ERROR "WriteBuildInfo: 发布模式禁止脏工作区候选（${REPO_DIR}）")
endif()
# Git branch names can contain JSON quote characters.
string(REPLACE "\\" "\\\\" BRANCH "${BRANCH}")
string(REPLACE "\"" "\\\"" BRANCH "${BRANCH}")
set(BUILT_AT_EPOCH "")
if(DEFINED SOURCE_FILE AND EXISTS "${SOURCE_FILE}")
    file(TIMESTAMP "${SOURCE_FILE}" BUILT_AT_EPOCH "%s" UTC)
elseif(DEFINED SCAN_DIR AND IS_DIRECTORY "${SCAN_DIR}")
    file(GLOB_RECURSE _build_info_artifacts "${SCAN_DIR}/*.dll" "${SCAN_DIR}/*.exe" "${SCAN_DIR}/*.pyd")
    foreach(_build_info_artifact IN LISTS _build_info_artifacts)
        if(_build_info_artifact MATCHES "[/\\\\]CMakeFiles[/\\\\]")
            continue()
        endif()
        file(TIMESTAMP "${_build_info_artifact}" _build_info_stamp "%s" UTC)
        if(NOT BUILT_AT_EPOCH OR _build_info_stamp GREATER BUILT_AT_EPOCH)
            set(BUILT_AT_EPOCH "${_build_info_stamp}")
        endif()
    endforeach()
endif()
if(NOT BUILT_AT_EPOCH)
    string(TIMESTAMP BUILT_AT_EPOCH "%s" UTC)
endif()

get_filename_component(_build_info_dir "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${_build_info_dir}")

file(WRITE "${OUTPUT}"
"{
  \"schema_version\": 1,
  \"component\": \"${COMPONENT}\",
  \"version\": \"${VERSION}\",
  \"commit\": \"${COMMIT}\",
  \"branch\": \"${BRANCH}\",
  \"dirty\": ${IS_DIRTY_VALUE},
  \"built_at_epoch\": ${BUILT_AT_EPOCH}
}
")
