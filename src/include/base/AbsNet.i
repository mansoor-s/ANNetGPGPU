%{
#include "base/AbsNet.h"
%}

%ignore ANN::AbsNet::SetTrainingSet(TrainingSet *);

%include "AbsNet.h"  
