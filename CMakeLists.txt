# The name of our project is "ANNetGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
cmake_minimum_required (VERSION 2.8)
project (ANNetGPU)

FIND_PACKAGE(Qt4)
FIND_PACKAGE(Doxygen)
FIND_PACKAGE(CUDA)

# Recurse into the "Hello" and "Demo" subdirectories. This does not actually
# cause another cmake executable to run. The same process will walk through
# the project's entire directory structure.
add_subdirectory (ANNet)

# Make sure the compiler can find include files from our ANNet library.
include_directories (${ANNetGPU_SOURCE_DIR}/include)

# Make sure the linker can find the Hello library once it is built.
link_directories (${ANNetGPU_SOURCE_DIR}/ANNet) 

if (QT4_FOUND)
  INCLUDE(${QT_USE_FILE})
  ADD_DEFINITIONS(${QT_DEFINITIONS})
endif(QT4_FOUND)

if(DOXYGEN_FOUND)
  configure_file(Doxyfile.in Doxyfile @ONLY)
  add_custom_target(doc ALL
    ${DOXYGEN_EXECUTABLE} Doxyfile
    WORKING_DIRECTORY ${ANNetGPU_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
  )
endif(DOXYGEN_FOUND)

# Add executable called "ANNetGPU" that is built from the source file
# "main.cpp". The extensions are automatically found.
if (CUDA_FOUND)
  cuda_add_executable (SOMNet examples/SOMNet.cpp)
  target_link_libraries (SOMNet ANNet) 
endif(CUDA_FOUND)

cuda_add_executable (BPNet examples/BPNet.cpp)
target_link_libraries (BPNet ANNet) 

cuda_add_executable (HFNet examples/HFNet.cpp)
target_link_libraries (HFNet ANNet) 

if (QT4_FOUND)
  add_executable (ANNetDesigner ANNetDesigner.cpp)
  target_link_libraries (ANNetDesigner ANNet) 
endif(QT4_FOUND)