# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
cmake_minimum_required (VERSION 2.8)
PROJECT (ANNetGPGPU)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

FIND_PACKAGE(BZip2 REQUIRED)
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(CUDA)
FIND_PACKAGE(CUDAThrust)
FIND_PACKAGE(SWIG)
FIND_PACKAGE(PythonLibs)

if(SWIG_FOUND)
  INCLUDE(${SWIG_USE_FILE})
  SET(CMAKE_SWIG_FLAGS "")
endif()
if(PYTHONLIBS_FOUND)
  INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
endif()

include(FindOpenMP)
if(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/base)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/math)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/containers)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/gpgpu)

set( ANPythonCPUInterfaceFiles
  include/PyNetCPU.i
)

set( ANPythonGPUInterfaceFiles
  include/PyNetGPU.i
)

set( ANSourceFiles 
  2DArray.cpp
  3DArray.cpp
  AbsLayer.cpp
  AbsNet.cpp
  AbsNeuron.cpp
  BPLayer.cpp
  BPNet.cpp
  BPNeuron.cpp
  Edge.cpp
  Functions.cpp
  HFLayer.cpp
  HFNet.cpp
  HFNeuron.cpp
  SOMLayer.cpp
  SOMNet.cpp
  SOMNeuron.cpp
  TrainingSet.cpp
)

set( ANCUDASourceFiles
  BPNetGPU.cpp
  SOMNetGPU.cpp
  BPKernel.cu
  SOMKernel.cu
  HFKernel.cu
  2DArray.cu
)

if (CUDA_FOUND)
  INCLUDE(FindCUDA)
  set(CUDA_NVCC_FLAGS "-arch=sm_13")
  include_directories (${CUDA_SDK_ROOT_DIR}/C/common/inc/)

  if (CUDATHRUST_FOUND)
    include_directories (${CUDATHRUST_INCLUDE_DIR})
  endif (CUDATHRUST_FOUND)
  
  ADD_DEFINITIONS("-DCUDA") # needed for conditional compilation of some files
endif (CUDA_FOUND)

# Create a library called "ANNet" which includes the source files listed in "ANSourceFiles".
# The extension is already found. Any number of sources could be listed here.
if (BZIP2_FOUND)
 
  # Build ANNet C library
  if (CUDA_FOUND)
    cuda_add_library (ANNet STATIC ${ANSourceFiles} ${ANCUDASourceFiles} ${BZIP_INCLUDE_DIRS}) 
  elseif (NOT CUDA_FOUND)
    add_library (ANNet STATIC ${ANSourceFiles} ${BZIP_INCLUDE_DIRS})
  endif(CUDA_FOUND)

  # Linke OpenMP to ANNet library
  if(OPENMP_FOUND)
    target_link_libraries (ANNet ${BZIP2_LIBRARIES} -fopenmp)
  elseif(NOT OPENMP_FOUND)
    target_link_libraries (ANNet ${BZIP2_LIBRARIES})
  endif(OPENMP_FOUND)

  # Build python interface
  if (SWIG_FOUND AND PYTHONLIBS_FOUND)
    # Create sub directory for the interface intermediate
    FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)

    SET_SOURCE_FILES_PROPERTIES(${ANPythonCPUInterfaceFiles} PROPERTIES CPLUSPLUS ON)
    SET_SOURCE_FILES_PROPERTIES(${ANPythonCPUInterfaceFiles} PROPERTIES CMAKE_SWIG_FLAGS "-includeall")
    SWIG_ADD_MODULE(ANPyNetCPU python ${ANPythonCPUInterfaceFiles})
    SWIG_LINK_LIBRARIES(ANPyNetCPU ${PYTHON_LIBRARIES} ANNet)

    if (CUDA_FOUND)
      SET_SOURCE_FILES_PROPERTIES(${ANPythonGPUInterfaceFiles} PROPERTIES CPLUSPLUS ON)
      SET_SOURCE_FILES_PROPERTIES(${ANPythonGPUInterfaceFiles} PROPERTIES CMAKE_SWIG_FLAGS "-includeall")
      SWIG_ADD_MODULE(ANPyNetGPU python ${ANPythonGPUInterfaceFiles})
      SWIG_LINK_LIBRARIES(ANPyNetGPU ${PYTHON_LIBRARIES} ANNet)
    endif(CUDA_FOUND)
  endif(SWIG_FOUND AND PYTHONLIBS_FOUND)

endif (BZIP2_FOUND)
