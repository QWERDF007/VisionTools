# 生成组件构建元数据 build_info.json。
#
# 由构建完成事件（POST_BUILD / 构建结束目标）调用：配置时刻早于真正的编译链接，不能冒充构建
# 时刻。时刻来源按优先级：
#   SOURCE_FILE  指定产物：使用该文件的最后修改时间，未重新链接的组件因此保留上一次真实构建时刻；
#   SCAN_DIR     扫描目录内全部 .dll/.exe/.pyd，取最新产物的修改时间（工具包多模块场景）；
#   都未给出     使用当前时刻（刚刚完成构建的目标）。
#
# 参数：
#   COMPONENT    组件名（写入 JSON）
#   VERSION      版本号
#   COMMIT       组件自身仓库的 40 位提交
#   BRANCH       组件自身仓库的分支
#   OUTPUT       目标 build_info.json
#   SOURCE_FILE  可选，单个构建产物
#   SCAN_DIR     可选，产物所在目录
#   REPO_DIR     可选，仓库目录；构建完成时实际查询当前 git commit 与 branch，防止配置期缓存滞后
if(NOT DEFINED COMPONENT OR NOT DEFINED OUTPUT)
    message(FATAL_ERROR "WriteBuildInfo.cmake 需要 COMPONENT 与 OUTPUT")
endif()

if(DEFINED REPO_DIR AND IS_DIRECTORY "${REPO_DIR}")
    execute_process(
        COMMAND git rev-parse HEAD
        WORKING_DIRECTORY "${REPO_DIR}"
        OUTPUT_VARIABLE _live_commit
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_live_commit)
        set(COMMIT "${_live_commit}")
    endif()
    execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY "${REPO_DIR}"
        OUTPUT_VARIABLE _live_branch
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_live_branch)
        set(BRANCH "${_live_branch}")
    endif()
endif()

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
  \"built_at_epoch\": ${BUILT_AT_EPOCH}
}
")
