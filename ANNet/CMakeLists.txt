# The name of our project is "ANNetGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
cmake_minimum_required (VERSION 2.8)
project (ANNetGPU)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})

FIND_PACKAGE(BZip2 REQUIRED)
FIND_PACKAGE(Qt4)
FIND_PACKAGE(CUDA)
FIND_PACKAGE(CUDAThrust)

# Make sure the compiler can find include files from our ANNet library.
#include_directories (${ANNetGPU_SOURCE_DIR})
include_directories (${ANNetGPU_SOURCE_DIR}/include)
include_directories (${ANNetGPU_SOURCE_DIR}/include/basic)
include_directories (${ANNetGPU_SOURCE_DIR}/include/math)
include_directories (${ANNetGPU_SOURCE_DIR}/include/gui)
include_directories (${ANNetGPU_SOURCE_DIR}/include/containers)
include_directories (${ANNetGPU_SOURCE_DIR}/include/gpgpu)

set( ANSourceFiles 
  AN2DArray.cpp
  AN3DArray.cpp
  ANAbsLayer.cpp
  ANAbsNet.cpp
  ANAbsNeuron.cpp
  ANBPLayer.cpp
  ANBPNet.cpp
  ANBPNeuron.cpp
  ANEdge.cpp
  ANFunctions.cpp
  ANHFLayer.cpp
  ANHFNet.cpp
  ANHFNeuron.cpp
  ANSOMLayer.cpp
  ANSOMNet.cpp
  ANSOMNeuron.cpp
  ANTrainingSet.cpp
)

set( ANGUISourceFiles
  QSOMReader.cpp 
)

set( ANCUDASourceFiles
  ANSOMNetGPU.cpp
  ANBPKernel.cu
  ANSOMKernel.cu
  ANHFKernel.cu
  ANMatrix.cu
)

set( QtHeaderFiles 
  include/gui/QSOMReader.h
)

if (QT4_FOUND)
  QT4_WRAP_CPP(QtHeaderFiles_MOC ${QtHeaderFiles})
  INCLUDE(${QT_USE_FILE})
  ADD_DEFINITIONS(${QT_DEFINITIONS})
endif(QT4_FOUND)

if (CUDA_FOUND)

  INCLUDE(FindCUDA)
  set(CUDA_NVCC_FLAGS "-arch=sm_11")
  include_directories (${CUDA_SDK_ROOT_DIR}/C/common/inc/)
#  LINK_LIBRARIES("-L/${CUDA_SDK_ROOT_DIR}/C/lib -lcutil")

  if (CUDATHRUST_FOUND)
    include_directories (${CUDATHRUST_INCLUDE_DIR})
  endif (CUDATHRUST_FOUND)
  
endif (CUDA_FOUND)

# Create a library called "ANNet" which includes the source files listed in "ANSourceFiles".
# The extension is already found. Any number of sources could be listed here.
if (CUDA_FOUND AND QT4_FOUND)
  cuda_add_library (ANNet SHARED ${ANSourceFiles} ${QtHeaderFiles_MOC} ${ANGUISourceFiles} ${ANCUDASourceFiles}) 
  target_link_libraries (ANNet ${QT_LIBRARIES})
elseif (QT4_FOUND AND NOT CUDA_FOUND)
  add_library (ANNet SHARED ${ANSourceFiles} ${QtHeaderFiles_MOC} ${ANGUISourceFiles})
  target_link_libraries (ANNet ${QT_LIBRARIES})
elseif (CUDA_FOUND AND NOT QT4_FOUND)
  cuda_add_library (ANNet SHARED ${ANSourceFiles} ${ANCUDASourceFiles}) 
endif ()

if (BZIP2_FOUND)
  include_directories(${BZIP_INCLUDE_DIRS})
  target_link_libraries (ANNet ${BZIP2_LIBRARIES})
endif (BZIP2_FOUND)
