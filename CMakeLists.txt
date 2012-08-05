# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
cmake_minimum_required (VERSION 2.8)
project (ANNetGPGPU)

FIND_PACKAGE(Qt4)
FIND_PACKAGE(Doxygen)
FIND_PACKAGE(CUDA)

add_subdirectory (ANNet)

include_directories (${ANNetGPGPU_SOURCE_DIR}/include)
link_directories (${ANNetGPU_SOURCE_DIR}/ANNet) 

if (QT4_FOUND)
  INCLUDE(${QT_USE_FILE})
  ADD_DEFINITIONS(${QT_DEFINITIONS})
endif(QT4_FOUND)

if(DOXYGEN_FOUND)
  configure_file(Doxyfile.in Doxyfile @ONLY)
  add_custom_target(doc #ALL
    ${DOXYGEN_EXECUTABLE} Doxyfile
    WORKING_DIRECTORY ${ANNetGPU_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
  )
endif(DOXYGEN_FOUND)

if (CUDA_FOUND)
  cuda_add_executable (SOMNet examples/SOMNet.cpp)
  target_link_libraries (SOMNet ANNet ANNetGUI) 

  cuda_add_executable (BPNetGPU examples/BPNetGPU.cpp)
  target_link_libraries (BPNetGPU ANNet) 
endif(CUDA_FOUND)

add_executable (LoadBPNet examples/LoadBPNet.cpp)
target_link_libraries (LoadBPNet ANNet) 

add_executable (BPNet examples/BPNet.cpp)
target_link_libraries (BPNet ANNet) 

add_executable (HFNet examples/HFNet.cpp)
target_link_libraries (HFNet ANNet) 

if (QT4_FOUND)
  if (WIN32)
    add_executable (ANNetDesigner WIN32 ANNetDesigner.cpp)
  elseif(NOT WIN32)
    add_executable (ANNetDesigner ANNetDesigner.cpp)
  endif(WIN32)
  target_link_libraries (ANNetDesigner ANNetGUI) 
endif(QT4_FOUND)