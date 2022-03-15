# Grab the .c files.
#
file (GLOB_RECURSE BINNEDBLAS_C_SOURCE "*.c")

# Set the header files.
#
set (BINNEDBLAS_C_HEADERS ${PROJECT_SOURCE_DIR}/include/binnedBLAS.h)

# Set all binned files.
#
set (BINNEDBLAS_FILES "${BINNEDBLAS_C_SOURCE};${BINNEDBLAS_C_HEADERS};${REPROBLAS_C_CONFIG_HEADER}")

# Add as an object library (not linked but used elsewhere)
#
add_library (binnedBLAS OBJECT ${BINNEDBLAS_FILES})

# Link cmake objects into library.
#
target_link_libraries (binnedBLAS PUBLIC $<TARGET_OBJECTS:binned>)

# Add headers install target.
#
install (FILES ${BINNEDBLAS_C_HEADERS} DESTINATION ${REPROBLAS_INCLUDE})