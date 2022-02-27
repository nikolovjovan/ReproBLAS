# Grab the .c files.
#
file (GLOB_RECURSE BINNED_C_SOURCE "*.c")

# Set the header files.
#
set (BINNED_C_HEADERS ${PROJECT_SOURCE_DIR}/include/binned.h)

# Set all binned files.
#
set (BINNED_FILES "${BINNED_C_SOURCE};${BINNED_C_HEADERS};${REPROBLAS_C_CONFIG_HEADER}")

# Add as an object library (not linked but used elsewhere)
#
add_library (binned STATIC)

# Add sources.
#
target_sources (binned PUBLIC ${BINNED_FILES})

# Add headers install target.
#
install (FILES ${BINNED_C_HEADERS} DESTINATION ${REPROBLAS_INCLUDE})