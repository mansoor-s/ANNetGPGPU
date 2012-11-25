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

#include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/basic)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/math)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/gui)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/containers)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/gpgpu)

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
  ANBPNetGPU.cpp
  ANSOMNetGPU.cpp
  ANBPKernel.cu
  ANSOMKernel.cu
  ANHFKernel.cu
  ANMatrix.cu
)

SET(ANGUIFormularFiles
  QTrainingForm.ui
)

set( 3rdPartySourceFiles
  include/gui/3rdparty/qcustomplot.cpp
  include/gui/3rdparty/utils/fancymainwindow.cpp
  include/gui/3rdparty/utils/manhattanstyle.cpp
  include/gui/3rdparty/utils/minisplitter.cpp
  include/gui/3rdparty/utils/styleanimator.cpp
  include/gui/3rdparty/utils/styledbar.cpp
  include/gui/3rdparty/utils/stylehelper.cpp
  include/gui/3rdparty/utils/stringutils.cpp
  include/gui/3rdparty/fancytabwidget.cpp
  include/gui/3rdparty/fancyactionbar.cpp
  include/gui/3rdparty/delegate/spinboxdelegate.cpp
)

set( 3rdPartyHeaderFiles
  include/gui/3rdparty/qcustomplot.h
  include/gui/3rdparty/utils/fancymainwindow.h
  include/gui/3rdparty/utils/manhattanstyle.h
  include/gui/3rdparty/utils/minisplitter.h
  include/gui/3rdparty/utils/styleanimator.h
  include/gui/3rdparty/utils/styledbar.h
  include/gui/3rdparty/utils/stylehelper.h
  include/gui/3rdparty/utils/stringutils.h
  include/gui/3rdparty/coreplugin/coreconstants.h
  include/gui/3rdparty/fancytabwidget.h
  include/gui/3rdparty/fancyactionbar.h
  include/gui/3rdparty/delegate/spinboxdelegate.h
)

set( ANGUISourceFiles
  QSOMReader.cpp
  QMainWindow.cpp
  QEdge.cpp
  QNode.cpp
  QLayer.cpp
  QLabel.cpp
  QZLabel.cpp
  QScene.cpp
  QTrainingForm.cpp
  QIOForm.cpp
  QViewer.cpp
  QOutput.cpp
  QGraphTab.cpp
  QTableWidget.cpp
  QTrainingThread.cpp
)

set( ANGUIHeaderFiles 
  include/gui/QSOMReader.h
  include/gui/QMainWindow.h
  include/gui/QEdge.h
  include/gui/QNode.h
  include/gui/QLayer.h
  include/gui/QLabel.h
  include/gui/QZLabel.h
  include/gui/QScene.h
  include/gui/QTrainingForm.h
  include/gui/QIOForm.h
  include/gui/QViewer.h
  include/gui/QOutput.h
  include/gui/QGraphTab.h
  include/gui/QTableWidget.h
  include/gui/QTrainingThread.h
)

if (QT4_FOUND)
  QT4_WRAP_CPP(ANGUIHeaderFiles_MOC ${ANGUIHeaderFiles} ${3rdPartyHeaderFiles})
  QT4_WRAP_UI(ANGUIHeaderFiles_MOC ${ANGUIFormularFiles})
  INCLUDE(${QT_USE_FILE})

  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

  ADD_DEFINITIONS(${QT_DEFINITIONS})
endif(QT4_FOUND)

if (CUDA_FOUND)
  INCLUDE(FindCUDA)
  set(CUDA_NVCC_FLAGS "-arch=sm_20")
  include_directories (${CUDA_SDK_ROOT_DIR}/C/common/inc/)

  if (CUDATHRUST_FOUND)
    include_directories (${CUDATHRUST_INCLUDE_DIR})
  endif (CUDATHRUST_FOUND)
  
  ADD_DEFINITIONS("-DCUDA") # needed for conditional compilation of some files
endif (CUDA_FOUND)

# Create a library called "ANNet" which includes the source files listed in "ANSourceFiles".
# The extension is already found. Any number of sources could be listed here.
if (BZIP2_FOUND)
  if (CUDA_FOUND)
    cuda_add_library (ANNet SHARED ${ANSourceFiles} ${ANCUDASourceFiles} ${BZIP_INCLUDE_DIRS}) 
  elseif (NOT CUDA_FOUND)
    add_library (ANNet SHARED ${ANSourceFiles} ${BZIP_INCLUDE_DIRS})
  endif(CUDA_FOUND)

  # -fopenmp necessary for mingw NOT gcc
  if(OPENMP_FOUND)
    target_link_libraries (ANNet ${BZIP2_LIBRARIES} -fopenmp)
  elseif(NOT OPENMP_FOUND)
    target_link_libraries (ANNet ${BZIP2_LIBRARIES})
  endif(OPENMP_FOUND)

  if (QT4_FOUND)
    add_library (ANNetGUI SHARED ${ANGUIHeaderFiles_MOC} ${ANGUISourceFiles} ${3rdPartySourceFiles})
    target_link_libraries (ANNetGUI ANNet ${QT_LIBRARIES})
  endif (QT4_FOUND)
endif (BZIP2_FOUND)
