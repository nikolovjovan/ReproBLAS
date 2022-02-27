# Grab the .c files.
#
file (GLOB_RECURSE REPROBLAS_C_SOURCE "*.c")

# Set the header files.
#
set (REPROBLAS_C_HEADERS ${PROJECT_SOURCE_DIR}/include/reproBLAS.h)

# Set all binned files.
#
set (REPROBLAS_FILES "${REPROBLAS_C_SOURCE};${REPROBLAS_C_HEADERS};${REPROBLAS_C_CONFIG_HEADER}")

# Add as an object library (not linked but used elsewhere)
#
add_library (reproBLAS STATIC)

# Add sources.
#
target_sources (reproBLAS PUBLIC ${REPROBLAS_FILES})

# Link cmake objects into library.
#
target_link_libraries (reproBLAS PUBLIC binned binnedBLAS)

# Set visibility.
#
set_target_properties (reproBLAS PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE)

# Set version.
#
set_target_properties (reproBLAS PROPERTIES
    VERSION "${REPROBLAS_VERSION_MAJOR}.${REPROBLAS_VERSION_MINOR}.0"
    SOVERSION ${REPROBLAS_API_VERSION}
    INTERFACE_INCLUDE_DIRECTORIES $<INSTALL_INTERFACE:include>)

# Add library install target.
#
install (TARGETS reproBLAS DESTINATION ${REPROBLAS_LIB})

# Add headers install target.
#
install (FILES ${REPROBLAS_C_HEADERS} DESTINATION ${REPROBLAS_INCLUDE})