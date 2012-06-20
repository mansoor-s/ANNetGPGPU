# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
cmake_minimum_required (VERSION 2.8)
project (ANNetGPGPU)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})

FIND_PACKAGE(BZip2 REQUIRED)
FIND_PACKAGE(Qt4)
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(CUDA)
FIND_PACKAGE(CUDAThrust)

include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

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

set( ANCUDASourceFiles
  ANSOMNetGPU.cpp
  ANBPKernel.cu
  ANSOMKernel.cu
  ANHFKernel.cu
  ANMatrix.cu
)

set( ANGUISourceFiles
  QSOMReader.cpp
  include/gui/QCustomPlot/qcustomplot.cpp
  include/gui/utils/fancymainwindow.cpp
  include/gui/utils/manhattanstyle.cpp
  include/gui/utils/minisplitter.cpp
  include/gui/utils/styleanimator.cpp
  include/gui/utils/styledbar.cpp
  include/gui/utils/stylehelper.cpp
  include/gui/fancytabwidget.cpp
  QInputdialog.cpp
  QMainWindow.cpp
  QEdge.cpp
  QNode.cpp
  QLayer.cpp
  QLabel.cpp
  QScene.cpp
  QViewer.cpp
)

set( ANGUIHeaderFiles 
  include/gui/QSOMReader.h
  include/gui/QCustomPlot/qcustomplot.h
  include/gui/utils/fancymainwindow.h
  include/gui/utils/manhattanstyle.h
  include/gui/utils/minisplitter.h
  include/gui/utils/styleanimator.h
  include/gui/utils/styledbar.h
  include/gui/utils/stylehelper.h
  include/gui/fancytabwidget.h
  include/gui/QInputdialog.h
  include/gui/QMainWindow.h
  include/gui/QEdge.h
  include/gui/QNode.h
  include/gui/QLayer.h
  include/gui/QLabel.h
  include/gui/QScene.h
  include/gui/QViewer.h
)

if (QT4_FOUND)
  QT4_WRAP_CPP(ANGUIHeaderFiles_MOC ${ANGUIHeaderFiles})
  INCLUDE(${QT_USE_FILE})
  ADD_DEFINITIONS(${QT_DEFINITIONS})
endif(QT4_FOUND)

if (CUDA_FOUND)
  INCLUDE(FindCUDA)
  set(CUDA_NVCC_FLAGS "-arch=sm_11")
  include_directories (${CUDA_SDK_ROOT_DIR}/C/common/inc/)

  if (CUDATHRUST_FOUND)
    include_directories (${CUDATHRUST_INCLUDE_DIR})
  endif (CUDATHRUST_FOUND)
endif (CUDA_FOUND)

# Create a library called "ANNet" which includes the source files listed in "ANSourceFiles".
# The extension is already found. Any number of sources could be listed here.
if (CUDA_FOUND)
  cuda_add_library (ANNet SHARED ${ANSourceFiles} ${ANCUDASourceFiles}) 
elseif (NOT CUDA_FOUND)
  add_library (ANNet SHARED ${ANSourceFiles} )
endif(CUDA_FOUND)

if (QT4_FOUND)
  add_library (ANNetGUI SHARED ${ANGUIHeaderFiles_MOC} ${ANGUISourceFiles})
  target_link_libraries (ANNetGUI ${QT_LIBRARIES})
endif (QT4_FOUND)

if (BZIP2_FOUND)
  include_directories(${BZIP_INCLUDE_DIRS})
  target_link_libraries (ANNet ${BZIP2_LIBRARIES})
endif (BZIP2_FOUND)
