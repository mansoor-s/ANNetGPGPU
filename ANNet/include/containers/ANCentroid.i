%{
#include <ANCentroid.h>
%}

%include <ANCentroid.h> 
%include <std_sstream.i>

namespace ANN {
	%extend Centroid {
		char *__str__() {
			std::ostringstream ostrs;
			char *c_str;
			
			for(unsigned int i = 0; i < $self->m_vCentroid.size(); i++) {
				float fVal = $self->m_vCentroid[i]->GetValue();
				ostrs << fVal << std::endl;
			}

			c_str = new char[ostrs.str().length()+1];
			strcpy(c_str, ostrs.str().c_str());
			return c_str;
		}
	}
}