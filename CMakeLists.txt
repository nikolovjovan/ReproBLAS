cmake_minimum_required (VERSION 3.18.4)

project (reproBLAS LANGUAGES C CXX)
enable_testing ()

include (CheckCSourceCompiles)
include (CheckCXXSourceCompiles)
include (CheckFunctionExists)
include (GenerateExportHeader)
include (GNUInstallDirs)

# The version number.
#
set (REPROBLAS_VERSION_MAJOR 2)
set (REPROBLAS_VERSION_MINOR 1)
set (REPROBLAS_API_VERSION 2)

option (BUILD_SHARED_LIBS "Build shared library" ON)
option (TEST_EXAMPLES "Execute examples with default parameters during CTest" OFF)

if (BUILD_SHARED_LIBS)
	set (CMAKE_POSITION_INDEPENDENT_CODE ON)
endif ()

# Config header file.
#
set (REPROBLAS_C_CONFIG_HEADER ${PROJECT_SOURCE_DIR}/config.h)

# Add library install target.
#
set (REPROBLAS_LIB ${PROJECT_BINARY_DIR}/lib)

# Add headers install target.
#
set (REPROBLAS_INCLUDE ${PROJECT_BINARY_DIR}/include)

# Traverse sources.
#
add_subdirectory (src)