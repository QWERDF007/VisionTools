include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/ConfigDependencyDefaults.cmake")

visiontools_resolve_path(_vt_python_home
    VARIABLES PYTHON_HOME Python_ROOT Python3_ROOT
    ENVIRONMENT_VARIABLES PYTHON_HOME Python_ROOT Python3_ROOT
    REQUIRED_FILE "python.exe"
    DEPENDENCY python
)

if(_vt_python_home)
    set(PYTHON_HOME "${_vt_python_home}" CACHE PATH "Python home directory" FORCE)
    set(PYTHON_HOME "${_vt_python_home}")
    set(Python3_ROOT_DIR "${_vt_python_home}")
    list(PREPEND CMAKE_PREFIX_PATH "${_vt_python_home}")
endif()

set(VISIONTOOLS_PYTHON_HOME "${_vt_python_home}")
