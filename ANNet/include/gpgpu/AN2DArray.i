%{
#include <AN2DArray.h>
%}

%include <AN2DArray.h>  

namespace ANN {
	%extend AN2DArray {
		char *__str__() {
			std::ostringstream ostrs;
			char *c_str;
			
			for(unsigned int y = 0; y < $self->GetHeight(); y++) {
				for(unsigned int x = 0; x < $self->GetWidth(); x++) {
					float fVal = $self[y*$self->GetWidth()+x];
					ostrs << fVal << std::endl;
				}
			}

			c_str = new char[ostrs.str().length()+1];
			strcpy(c_str, ostrs.str().c_str());
			return c_str;
		}
	}
}