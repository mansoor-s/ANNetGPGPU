%{
#include <AN3DArray.h>
%}

%ignore ANN::F3DArray::operator [];

%include <AN3DArray.h>   

%extend ANN::F3DArray {
	ANN::F2DArray __getitem__(int z) {
		return self->GetSubArrayXY(z);
	}
};