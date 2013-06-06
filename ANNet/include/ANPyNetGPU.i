%define DOCSTRING
"ANNet is a small library to create neural networks. 
At the moment there are implementations of several neural network models with usage of OpenMP and/or Thrust.
Author: Daniel Frenzel"
%enddef
%module(docstring=DOCSTRING) ANPyNetGPU

%include ANIgnore.i

%include basic/ANEdge.i
%include basic/ANAbsNeuron.i
%include basic/ANAbsLayer.i
%include basic/ANAbsNet.i

%include containers/ANCentroid.i
%include containers/AN2DArray.i
%include containers/AN3DArray.i
%include containers/ANTrainingSet.i
%include containers/ANConTable.i

%include ANHFNeuron.i
%include ANHFLayer.i
%include ANHFNet.i

%include ANSOMNeuron.i
%include ANSOMLayer.i
%include ANSOMNet.i

%include ANBPNeuron.i
%include ANBPLayer.i
%include ANBPNet.i

%include math/ANFunctions.i
//%include math/ANRandom.i

//%include gpgpu/ANKernels.i
//%include gpgpu/ANMatrix.i
%include gpgpu/ANBPNetGPU.i
%include gpgpu/ANSOMNetGPU.i

%include ANStdOut.i