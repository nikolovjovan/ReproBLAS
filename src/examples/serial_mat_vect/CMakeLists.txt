# Link math library.
#
set (EXTRA_LIBS ${EXTRA_LIBS} m)

# Set target name.
#
set (EXAMPLE_TARGET_NAME sum_mat_vect)

# Set install target directory.
#
set (EXAMPLE_TARGET_DIR ${PROJECT_BINARY_DIR}/examples/${EXAMPLE_TARGET_NAME})

# Grab the .c and .cpp files.
#
file (GLOB_RECURSE EXAMPLE_C_CPP_SOURCE "*.c" "*.cpp")

# Grab the C/C++ headers.
#
file (GLOB_RECURSE EXAMPLE_C_CPP_HEADERS "*.h")

set (EXAMPLE_FILES "${EXAMPLE_C_CPP_SOURCE};${EXAMPLE_C_CPP_HEADERS}")

# Add target example executable.
#
add_executable (${EXAMPLE_TARGET_NAME} ${EXAMPLE_FILES})

# Link libraries.
#
target_link_libraries (${EXAMPLE_TARGET_NAME} $<TARGET_OBJECTS:binned> $<TARGET_OBJECTS:binnedBLAS> $<TARGET_OBJECTS:reproBLAS> ${EXTRA_LIBS})

# Add install targets.
#
install (TARGETS ${EXAMPLE_TARGET_NAME} DESTINATION ${EXAMPLE_TARGET_DIR})

# Add CMake tests to execute examples to verify default execution does not crash if option is specified.
#
if (TEST_EXAMPLES)
    add_test (NAME ${EXAMPLE_TARGET_NAME} COMMAND ${EXAMPLE_TARGET_DIR}/${EXAMPLE_TARGET_NAME})
endif ()
