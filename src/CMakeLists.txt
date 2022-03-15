# Set include directories.
#
include_directories (${PROJECT_SOURCE_DIR}/include)
include_directories (${PROJECT_BINARY_DIR}/include)

# Set default C compiler flags.
#
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -march=native -masm=intel -fabi-version=0 -O1 -Wall -Wconversion -Wsign-conversion -Wsign-compare -fdiagnostics-color")

# Set default C++ compiler flags.
#
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native -masm=intel -fabi-version=0 -O1 -Wall -Wconversion -Wsign-conversion -Wsign-compare -fdiagnostics-color")

# Traverse library and examples.
#
add_subdirectory(binned)
add_subdirectory(binnedBLAS)
add_subdirectory(reproBLAS)
add_subdirectory(examples)
