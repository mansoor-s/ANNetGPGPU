# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
cmake_minimum_required (VERSION 2.8)
PROJECT (ANNetMisc)

FIND_PACKAGE(Qt4)
FIND_PACKAGE(CUDA)

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include)

SET( ANGUIHeaderFiles 
  QSOMReader.h
)

SET( ANGUISourceFiles
  QSOMReader.cpp
)

if (QT4_FOUND)
  QT4_WRAP_CPP(ANGUIHeaderFiles_MOC ${ANGUIHeaderFiles} ${3rdPartyHeaderFiles})
  QT4_WRAP_UI(ANGUIHeaderFiles_MOC ${ANGUIFormularFiles})
  INCLUDE(${QT_USE_FILE})
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
  ADD_DEFINITIONS(${QT_DEFINITIONS})

  # Build Qt library
  add_library (SOMReader STATIC ${ANGUIHeaderFiles_MOC} ${ANGUISourceFiles})
  target_link_libraries (SOMReader ${QT_LIBRARIES})
endif (QT4_FOUND)

# Build misc examples
if (CUDA_FOUND)
  if (QT4_FOUND)
    cuda_add_executable (SOMNetGPU SOMNetGPU.cpp)
    target_link_libraries (SOMNetGPU ANNet SOMReader) 
  endif (QT4_FOUND)
  
  cuda_add_executable (BPNetGPU BPNetGPU.cpp)
  target_link_libraries (BPNetGPU ANNet) 
endif(CUDA_FOUND)

add_executable (LoadBPNet LoadBPNet.cpp)
target_link_libraries (LoadBPNet ANNet) 

add_executable (BPNet BPNet.cpp)
target_link_libraries (BPNet ANNet) 

add_executable (HFNet HFNet.cpp)
target_link_libraries (HFNet ANNet) 

if (QT4_FOUND)
  add_executable (SOMNetCPU SOMNetCPU.cpp)
  target_link_libraries (SOMNetCPU ANNet SOMReader) 
    
  add_executable (SOMNetClusteringApproach SOMNetClusteringApproach.cpp)
  target_link_libraries (SOMNetClusteringApproach ANNet SOMReader)  
endif(QT4_FOUND)

