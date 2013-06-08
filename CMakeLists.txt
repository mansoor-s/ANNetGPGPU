# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
cmake_minimum_required (VERSION 2.8)
project (ANNetGPGPU)

FIND_PACKAGE(Doxygen)

add_subdirectory (src) 			# ANNet library
add_subdirectory (examples/designer) 	# BP network designer
add_subdirectory (examples/misc) 	# several examples

include_directories (${ANNetGPGPU_SOURCE_DIR}/include)
link_directories (${ANNetGPU_SOURCE_DIR}/ANNet) 

# Build docu
if(DOXYGEN_FOUND)
  configure_file(Doxyfile Doxyfile @ONLY)
  add_custom_target(doc #ALL
    ${DOXYGEN_EXECUTABLE} Doxyfile
    WORKING_DIRECTORY ${ANNetGPU_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
  )
endif(DOXYGEN_FOUND)
