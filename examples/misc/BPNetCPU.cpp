/*
 * main.cpp
 *
 *  Created on: 12.04.2010
 *      Author: dgrat
 */

#include <ANNet>
#include <ANContainers>
#include <ANMath>

#include "Samples.h"

#include <ctime>
#include <iostream>


int main(int argc, char *argv[]) {
	ANN::BPNet cpu_one;
	ANN::BPLayer layer1(3, ANN::ANLayerInput);
	//layer1.AddFlag(ANN::ANBiasNeuron);
	ANN::BPLayer layer2(64, ANN::ANLayerHidden);
	//layer2.AddFlag(ANN::ANBiasNeuron);
	ANN::BPLayer layer3(64, ANN::ANLayerHidden);
	//layer3.AddFlag(ANN::ANBiasNeuron);
	ANN::BPLayer layer4(64, ANN::ANLayerHidden);
	//layer4.AddFlag(ANN::ANBiasNeuron);
	ANN::BPLayer layer5(6, ANN::ANLayerOutput);

	layer1.ConnectLayer(&layer2);
	layer2.ConnectLayer(&layer3);
	layer3.ConnectLayer(&layer4);
	layer4.ConnectLayer(&layer5);

	cpu_one.AddLayer(&layer1);
	cpu_one.AddLayer(&layer2);
	cpu_one.AddLayer(&layer3);
	cpu_one.AddLayer(&layer4);
	cpu_one.AddLayer(&layer5);

	ANN::TrainingSet input;
	input.AddInput(fInp1, 3);
	input.AddOutput(fOut1, 6);
	input.AddInput(fInp2, 3);
	input.AddOutput(fOut2, 6);
	input.AddInput(fInp3, 3);
	input.AddOutput(fOut3, 6);
	input.AddInput(fInp4, 3);
	input.AddOutput(fOut4, 6);

	std::vector<float> errors;
	cpu_one.SetLearningRate(0.075);
	cpu_one.SetMomentum(0);
	cpu_one.SetWeightDecay(0);
	cpu_one.SetTrainingSet(input);

	bool b = false;
	float f;
	errors = cpu_one.TrainFromData(1000, 0.001, b, f);
	std::cout<< cpu_one <<std::endl;

	cpu_one.ExpToFS("foo.bar");
	cpu_one.ImpFromFS("foo.bar");

	cpu_one.SetTrainingSet(input);
	std::cout<< cpu_one <<std::endl;

	std::cout<<"CREATE OTHER INSTANCE"<<std::endl;

	ANN::BPNet cpu_two;
	cpu_two.ImpFromFS("foo.bar");
	cpu_two.SetTrainingSet(input);
	std::cout<< cpu_two <<std::endl;

	return 0;
}
