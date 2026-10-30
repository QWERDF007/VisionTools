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
#   COMMIT         组件自身仓库的 40 位提交（若未提供 REPO_DIR 时的备选）
#   BRANCH         组件自身仓库的分支
#   OUTPUT         目标 build_info.json
#   SOURCE_FILE    可选，单个构建产物
#   SCAN_DIR       可选，产物所在目录
#   REPO_DIR       可选，仓库目录；构建完成时实际查询当前 git commit 与 branch，防止配置期缓存滞后
#   RELEASE_BUILD  可选，发布模式（ON 时遇未知状态或脏工作区报错中断）
if(NOT DEFINED COMPONENT OR NOT DEFINED OUTPUT)
    message(FATAL_ERROR "WriteBuildInfo.cmake 需要 COMPONENT 与 OUTPUT")
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

set(GIT_QUERY_OK FALSE)
set(IS_DIRTY_VALUE "false")

if(DEFINED REPO_DIR AND IS_DIRECTORY "${REPO_DIR}")
    # 1. 查询当前 HEAD commit
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${REPO_DIR}"
        OUTPUT_VARIABLE _live_commit
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _git_commit_res
        ERROR_QUIET
    )

    # 2. 查询当前 HEAD branch
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY "${REPO_DIR}"
        OUTPUT_VARIABLE _live_branch
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _git_branch_res
        ERROR_QUIET
    )

    if(_git_commit_res EQUAL 0 AND _live_commit AND _git_branch_res EQUAL 0 AND _live_branch)
        set(GIT_QUERY_OK TRUE)
        set(COMMIT "${_live_commit}")
        set(BRANCH "${_live_branch}")

        # 3. 检查工作区状态（跟踪变更、未跟踪生产文件、子模块）：git status --porcelain -uall
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" status --porcelain -uall
            WORKING_DIRECTORY "${REPO_DIR}"
            OUTPUT_VARIABLE _git_status_out
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _git_status_res
            ERROR_QUIET
        )

        set(IS_DIRTY FALSE)
        if(_git_status_res EQUAL 0)
            if(_git_status_out)
                string(REPLACE "\r\n" "\n" _status_norm "${_git_status_out}")
                string(REPLACE "\n" ";" _status_lines "${_status_norm}")
                foreach(_line IN LISTS _status_lines)
                    if(_line STREQUAL "")
                        continue()
                    endif()
                    string(SUBSTRING "${_line}" 3 -1 _raw_path)
                    string(STRIP "${_raw_path}" _raw_path)
                    if(_raw_path MATCHES ".* -> (.*)")
                        set(_raw_path "${CMAKE_MATCH_1}")
                    endif()
                    string(REPLACE "\\" "/" _norm_path "${_raw_path}")

                    # 仅显式排除本地维护的 WORKLOG.md 与约定的本地审查报告/资料（如 docs/R*.md, docs/VisionRT_*.zip）
                    # 其余所有修改或未跟踪生产文件均记为 dirty
                    if(_norm_path STREQUAL "WORKLOG.md" OR _norm_path MATCHES ".*/WORKLOG\\.md$")
                        continue()
                    elseif(_norm_path MATCHES "^docs/R[0-9].*\\.md$")
                        continue()
                    elseif(_norm_path MATCHES "^docs/VisionRT_.*\\.zip$")
                        continue()
                    else()
                        set(IS_DIRTY TRUE)
                        break()
                    endif()
                endforeach()
            endif()
        else()
            set(IS_DIRTY TRUE)
        endif()

        if(IS_DIRTY)
            set(IS_DIRTY_VALUE "true")
            if(RELEASE_BUILD)
                message(FATAL_ERROR "WriteBuildInfo: 发布模式禁止脏工作区候选（${REPO_DIR} 存在未提交变更或未跟踪生产输入）")
            endif()
        else()
            set(IS_DIRTY_VALUE "false")
        endif()
    else()
        # Git 查询失败（非 Git 仓库、Git 不存在或查询失败）：
        # 禁止回退到旧配置期 COMMIT 并声称 dirty=false
        set(COMMIT "unknown")
        set(BRANCH "unknown")
        set(IS_DIRTY_VALUE "\"unknown\"")

        if(RELEASE_BUILD)
            message(FATAL_ERROR "WriteBuildInfo: 发布模式禁止未知候选。无法在 ${REPO_DIR} 查询 Git 提交信息（退出码：${_git_commit_res}）")
        endif()
    endif()
else()
    if(NOT COMMIT OR COMMIT STREQUAL "")
        set(COMMIT "unknown")
    endif()
    if(NOT BRANCH OR BRANCH STREQUAL "")
        set(BRANCH "unknown")
    endif()
    set(IS_DIRTY_VALUE "\"unknown\"")

    if(RELEASE_BUILD)
        message(FATAL_ERROR "WriteBuildInfo: 发布模式需要有效的 REPO_DIR 以验证组件候选身份")
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
  \"dirty\": ${IS_DIRTY_VALUE},
  \"built_at_epoch\": ${BUILT_AT_EPOCH}
}
")
