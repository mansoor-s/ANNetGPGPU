# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
CMAKE_MINIMUM_REQUIRED (VERSION 2.8)
PROJECT (ANNetGPGPU)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

FIND_PACKAGE(BZip2 REQUIRED)
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(CUDA)
FIND_PACKAGE(CUDAThrust)
FIND_PACKAGE(SWIG)
FIND_PACKAGE(PythonLibs)

IF(SWIG_FOUND)
  INCLUDE(${SWIG_USE_FILE})
  SET(CMAKE_SWIG_FLAGS "")
ENDIF()
IF(PYTHONLIBS_FOUND)
  INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
ENDIF()

INCLUDE(FindOpenMP)
IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF()

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/base)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/math)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/containers)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/gpgpu)

SET( ANPythonCPUInterfaceFiles
  include/PyNetCPU.i
)

SET( ANPythonGPUInterfaceFiles
  include/PyNetGPU.i
)

SET( ANSourceFiles 
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

SET( ANCUDASourceFiles
  BPNetGPU.cpp
  SOMNetGPU.cpp
  BPKernel.cu
  SOMKernel.cu
  HFKernel.cu
  2DArray.cu
)

IF (CUDA_FOUND)
  INCLUDE(FindCUDA)
  SET(CUDA_NVCC_FLAGS "-O3 -arch=sm_13 --pre-include gcc_cuda_compat.h")
  INCLUDE_DIRECTORIES (${CUDA_SDK_ROOT_DIR}/C/common/inc/)

  IF (CUDATHRUST_FOUND)
    INCLUDE_DIRECTORIES (${CUDATHRUST_INCLUDE_DIR})
  ENDIF (CUDATHRUST_FOUND)
  
  ADD_DEFINITIONS("-DCUDA") # needed for conditional compilation of some files
ENDIF (CUDA_FOUND)

# Create a library called "ANNet" which includes the source files listed in "ANSourceFiles".
IF (BZIP2_FOUND)
  # Build ANNet C library
  IF (CUDA_FOUND)
    CUDA_ADD_LIBRARY (ANNet STATIC ${ANCUDASourceFiles} ${ANSourceFiles} ${BZIP_INCLUDE_DIRS})
  ELSEIF (NOT CUDA_FOUND)
    ADD_LIBRARY (ANNet STATIC ${ANSourceFiles} ${BZIP_INCLUDE_DIRS})
  ENDIF(CUDA_FOUND)

  # Linke OpenMP to ANNet library
  IF(OPENMP_FOUND)
    TARGET_LINK_LIBRARIES (ANNet ${BZIP2_LIBRARIES} -fopenmp)
  ELSEIF(NOT OPENMP_FOUND)
    TARGET_LINK_LIBRARIES (ANNet ${BZIP2_LIBRARIES})
  ENDIF(OPENMP_FOUND)

  # Build python interface
  IF (SWIG_FOUND AND PYTHONLIBS_FOUND)
    # Create sub directory for the interface intermediate
    FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)

    SET_SOURCE_FILES_PROPERTIES(${ANPythonCPUInterfaceFiles} PROPERTIES CPLUSPLUS ON)
    SET_SOURCE_FILES_PROPERTIES(${ANPythonCPUInterfaceFiles} PROPERTIES CMAKE_SWIG_FLAGS "-includeall")
    SWIG_ADD_MODULE(ANPyNetCPU python ${ANPythonCPUInterfaceFiles})
    SWIG_LINK_LIBRARIES(ANPyNetCPU ${PYTHON_LIBRARIES} ANNet)

    IF (CUDA_FOUND)
      SET_SOURCE_FILES_PROPERTIES(${ANPythonGPUInterfaceFiles} PROPERTIES CPLUSPLUS ON)
      SET_SOURCE_FILES_PROPERTIES(${ANPythonGPUInterfaceFiles} PROPERTIES CMAKE_SWIG_FLAGS "-includeall")
      SWIG_ADD_MODULE(ANPyNetGPU python ${ANPythonGPUInterfaceFiles})
      SWIG_LINK_LIBRARIES(ANPyNetGPU ${PYTHON_LIBRARIES} ANNet)
    ENDIF(CUDA_FOUND)
  ENDIF(SWIG_FOUND AND PYTHONLIBS_FOUND)
ENDIF (BZIP2_FOUND)
