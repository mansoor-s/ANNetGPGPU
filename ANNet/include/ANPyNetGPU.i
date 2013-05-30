%define DOCSTRING
"ANNet is a small library to create neural networks. 
At the moment there are implementations of several neural network models with usage of OpenMP and/or Thrust.
Author: Daniel Frenzel"
%enddef
%module(docstring=DOCSTRING) ANPyNetGPU


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


%include <std_vector.i>
%extend std::vector<int> {
	char *__str__() {
		std::ostringstream ostrs;
		char *c_str;
		
		ostrs << "[";
		for(unsigned int i = 0; i < $self->size(); i++) {
			int fVal = $self->at(i);
			ostrs << fVal;
			if(i < $self->size()-1) {
				ostrs << ", ";
			}
		}
		ostrs << "]\n";

		c_str = new char[ostrs.str().length()+1];
		strcpy(c_str, ostrs.str().c_str());
		return c_str;
	}
}
%extend std::vector<float> {
	char *__str__() {
		std::ostringstream ostrs;
		char *c_str;
		
		ostrs << "[";
		for(unsigned int i = 0; i < $self->size(); i++) {
			float fVal = $self->at(i);
			ostrs << fVal;
			if(i < $self->size()-1) {
				ostrs << ", ";
			}
		}
		ostrs << "]";

		c_str = new char[ostrs.str().length()+1];
		strcpy(c_str, ostrs.str().c_str());
		return c_str;
	}
}
%extend std::vector<ANN::Centroid> {
	char *__str__() {
		std::ostringstream ostrs;
		char *c_str;
		
		ostrs << "[";
		for(unsigned int i = 0; i < $self->size(); i++) {
			ostrs << "[";
			for(unsigned int j = 0; j < $self->at(i).m_vCentroid.size(); j++) {
				float fVal = $self->at(i).m_vCentroid.at(j); 
				ostrs << fVal;
				if(j < $self->at(i).m_vCentroid.size()-1) {
					ostrs << ", ";
				}
			}
			ostrs << "]";
			if(i < $self->size()-1) {
				ostrs << ", ";
			}
		}
		ostrs << "]";

		c_str = new char[ostrs.str().length()+1];
		strcpy(c_str, ostrs.str().c_str());
		return c_str;
	}
}
%template(vectori) std::vector<int>;
%template(vectorf) std::vector<float>;
%template(vectorc) std::vector<ANN::Centroid>;