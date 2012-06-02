/*
 * main.cpp
 *
 *  Created on: 12.04.2010
 *      Author: dgrat
 */

#include <ANNet>
#include <ANContainers>
#include <ANMath>
#include <ANGPGPU>
#include <ANGUI>
#include <math/ANRandom.h>

#include <gpgpu/ANKernels.h>

#include <ctime>
#include <iostream>

//#include <gpgpu/ANSOMKernel.cu>

//using namespace ANN;

int main(int argc, char *argv[]) {
/*
QApplication a(argc, argv);

	//create a data set
std::vector<float> red, green, blue, yellow, orange, purple, dk_green, dk_blue, black, white;

white.push_back(1);
white.push_back(1);
white.push_back(1);

black.push_back(0);
black.push_back(0);
black.push_back(0);

red.push_back(1);
red.push_back(0);
red.push_back(0);

green.push_back(0);
green.push_back(1);
green.push_back(0);

dk_green.push_back(0);
dk_green.push_back(0.5);
dk_green.push_back(0.25);

blue.push_back(0);
blue.push_back(0);
blue.push_back(1);

dk_blue.push_back(0);
dk_blue.push_back(0);
dk_blue.push_back(0.5);

yellow.push_back(1);
yellow.push_back(1);
yellow.push_back(0.2);

orange.push_back(1);
orange.push_back(0.4);
orange.push_back(0.25);

purple.push_back(1);
purple.push_back(0);
purple.push_back(1);

ANN::TrainingSet input;
input.AddInput(red);
input.AddInput(green);
input.AddInput(dk_green);
input.AddInput(blue);
input.AddInput(dk_blue);
input.AddInput(yellow);
input.AddInput(orange);
input.AddInput(purple);
input.AddInput(black);
input.AddInput(white);

std::vector<float> vCol(3);
int w1 = 40;
int w2 = 4;
SOMReader w(w1, w1, w2);

ANN::SOMNet SOMap;
SOMap.SetTrainingSet(input);
SOMap.CreateSOM(3, 1, w1,w1);

ANN::SOMNetGPU gpu(&SOMap);
//	ANN::SOMNetGPU gpu;
//	gpu.SetTrainingSet(input);
//	gpu.CreateSOM(3, 1, w1,w1);

SOMap.Training(9);

//for(int x = 0; x < w1*w1; x++) {
//	ANN::SOMNeuron *pNeur = (ANN::SOMNeuron*)((ANN::SOMLayer*)SOMap.GetOPLayer())->GetNeuron(x);
//	vCol[0] = pNeur->GetConI(0)->GetValue();
//	vCol[1] = pNeur->GetConI(1)->GetValue();
//	vCol[2] = pNeur->GetConI(2)->GetValue();

//	w.SetField(QPoint(pNeur->GetPosition()[0], pNeur->GetPosition()[1]), vCol );
//}
//w.Save("CPU.png");

// GPU
gpu.Training(1000);

for(int x = 0; x < w1*w1; x++) {
	ANN::SOMNeuron *pNeur = (ANN::SOMNeuron*)((ANN::SOMLayer*)gpu.GetOPLayer())->GetNeuron(x);
	vCol[0] = pNeur->GetConI(0)->GetValue();
	vCol[1] = pNeur->GetConI(1)->GetValue();
	vCol[2] = pNeur->GetConI(2)->GetValue();

	w.SetField(QPoint(pNeur->GetPosition()[0], pNeur->GetPosition()[1]), vCol );
}

w.Save("GPU_1.png");

//gpu.ExpToFS("foo1.bar");
//gpu.ImpFromFS("foo1.bar");

gpu.ExpToFS("foo2.bar");
gpu.ImpFromFS("foo2.bar");

for(int x = 0; x < w1*w1; x++) {
	ANN::SOMNeuron *pNeur = (ANN::SOMNeuron*)((ANN::SOMLayer*)gpu.GetOPLayer())->GetNeuron(x);
	vCol[0] = pNeur->GetConI(0)->GetValue();
	vCol[1] = pNeur->GetConI(1)->GetValue();
	vCol[2] = pNeur->GetConI(2)->GetValue();

	w.SetField(QPoint(pNeur->GetPosition()[0], pNeur->GetPosition()[1]), vCol );
}

w.Save("GPU_2.png");
*/

/*
float fInp1[3];
fInp1[0] = 0;
fInp1[1] = 0;
fInp1[2] = 0;

float fInp2[3];
fInp2[0] = 0;
fInp2[1] = 1;
fInp2[2] = 0;

float fInp3[3];
fInp3[0] = 0;
fInp3[1] = 0;
fInp3[2] = 1;

float fInp4[3];
fInp4[0] = 1;
fInp4[1] = 0;
fInp4[2] = 1;

float fOut1[6];
fOut1[0] = 0.1;
fOut1[1] = 0.2;
fOut1[2] = 0.3;
fOut1[3] = 0.4;
fOut1[4] = 0.5;
fOut1[5] = 0.6;
float fOut2[6];

fOut2[0] = 0;
fOut2[1] = 1;
fOut2[2] = 0;
fOut2[3] = 0;
fOut2[4] = 0;
fOut2[5] = 0;

float fOut3[6];
fOut3[0] = 0;
fOut3[1] = 0;
fOut3[2] = 1;
fOut3[3] = 0;
fOut3[4] = 0;
fOut3[5] = 0;

float fOut4[6];
fOut4[0] = 0;
fOut4[1] = 0;
fOut4[2] = 0;
fOut4[3] = 1;
fOut4[4] = 0;
fOut4[5] = 0;

//SimpleNet net;
ANN::BPNet net;
ANN::BPLayer layer1(3, ANN::ANLayerInput);
layer1.AddFlag(ANN::ANBiasNeuron);
ANN::BPLayer layer2(64, ANN::ANLayerHidden);
layer2.AddFlag(ANN::ANBiasNeuron);
ANN::BPLayer layer3(64, ANN::ANLayerHidden);
layer3.AddFlag(ANN::ANBiasNeuron);
ANN::BPLayer layer4(64, ANN::ANLayerHidden);
layer4.AddFlag(ANN::ANBiasNeuron);
ANN::BPLayer layer5(6, ANN::ANLayerOutput);

layer1.ConnectLayer(&layer2);
layer2.ConnectLayer(&layer3);
layer3.ConnectLayer(&layer4);
layer4.ConnectLayer(&layer5);

net.AddLayer(&layer1);
net.AddLayer(&layer2);
net.AddLayer(&layer3);
net.AddLayer(&layer4);
net.AddLayer(&layer5);

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
net.SetLearningRate(0.2);
net.SetMomentum(0.9);
net.SetWeightDecay(0);
net.SetTrainingSet(input);


//errors = net.TrainFromData(5000, 0.001);
std::cout<< net <<std::endl;

net.ExpToFS("TEST");
net.ImpFromFS("TEST");
net.SetTrainingSet(input);
std::cout<< net <<std::endl;
*/
//////////////////////////////////////////////////////////////////

float TR[16];
TR[0] 	= -1;
TR[1] 	= 1;
TR[2] 	= -1;
TR[3] 	= 1;
TR[4] 	= -1;
TR[5] 	= 1;
TR[6] 	= -1;
TR[7] 	= 1;
TR[8] 	= -1;
TR[9] 	= 1;
TR[10] 	= -1;
TR[11] 	= 1;
TR[12] 	= -1;
TR[13] 	= 1;
TR[14] 	= -1;
TR[15] 	= 1;


float TR2[16];
TR2[0] 	= -1;
TR2[1] 	= -1;
TR2[2] 	= -1;
TR2[3] 	= -1;
TR2[4] 	= -1;
TR2[5] 	= -1;
TR2[6] 	= -1;
TR2[7] 	= -1;
TR2[8] 	= 1;
TR2[9] 	= 1;
TR2[10] = 1;
TR2[11] = 1;
TR2[12] = 1;
TR2[13] = 1;
TR2[14] = 1;
TR2[15] = 1;

float TR3[16];
TR3[0] 	= 1;
TR3[1] 	= 1;
TR3[2] 	= 1;
TR3[3] 	= 1;
TR3[4] 	= 1;
TR3[5] 	= 1;
TR3[6] 	= 1;
TR3[7] 	= 1;
TR3[8] 	= 1;
TR3[9] 	= 1;
TR3[10] = 1;
TR3[11] = 1;
TR3[12] = 1;
TR3[13] = 1;
TR3[14] = 1;
TR3[15] = 1;

float fInpHF[16];
fInpHF[0] 	= 1;
fInpHF[1] 	= -1;
fInpHF[2] 	= -1;
fInpHF[3] 	= 1;
fInpHF[4] 	= -1;
fInpHF[5] 	= 1;
fInpHF[6] 	= -1;
fInpHF[7] 	= 1;
fInpHF[8] 	= -1;
fInpHF[9] 	= 1;
fInpHF[10] 	= 1;
fInpHF[11] 	= 1;
fInpHF[12] 	= -1;
fInpHF[13] 	= 1;
fInpHF[14] 	= -1;
fInpHF[15] 	= 1;

float fInpHF2[16];
fInpHF2[0] 	= 1;
fInpHF2[1] 	= 1;
fInpHF2[2] 	= 1;
fInpHF2[3] 	= 1;
fInpHF2[4] 	= 1;
fInpHF2[5] 	= 1;
fInpHF2[6] 	= -1;
fInpHF2[7] 	= 1;
fInpHF2[8] 	= 1;
fInpHF2[9] 	= 1;
fInpHF2[10] = 1;
fInpHF2[11] = -1;
fInpHF2[12] = 1;
fInpHF2[13] = 1;
fInpHF2[14] = 1;
fInpHF2[15] = -1;

float fInpHF3[16];
fInpHF3[0] 	= 1;
fInpHF3[1] 	= -1;
fInpHF3[2] 	= 1;
fInpHF3[3] 	= -1;
fInpHF3[4] 	= -1;
fInpHF3[5] 	= 1;
fInpHF3[6] 	= -1;
fInpHF3[7] 	= -1;
fInpHF3[8] 	= 1;
fInpHF3[9] 	= 1;
fInpHF3[10] = 1;
fInpHF3[11] = 1;
fInpHF3[12] = 1;
fInpHF3[13] = 1;
fInpHF3[14] = 1;
fInpHF3[15] = 1;

ANN::TrainingSet input;
input.AddInput(TR, 16);
input.AddInput(TR2, 16);
input.AddInput(TR3, 16);

ANN::HFNet hfnet;
hfnet.Resize(16,1);
hfnet.SetTrainingSet(input);
hfnet.PropagateBW();

hfnet.SetInput(fInpHF);
for(int k = 0; k < 1; k++) {
	hfnet.PropagateFW();

	for(int k = 0; k < hfnet.GetOutput().size(); k++) {
		std::cout<<"outp: "<<hfnet.GetOutput().at(k)<<std::endl;
	}
	std::cout<<std::endl;
}
hfnet.SetInput(fInpHF2);
for(int k = 0; k < 1; k++) {
	hfnet.PropagateFW();

	for(int k = 0; k < hfnet.GetOutput().size(); k++) {
		std::cout<<"outp: "<<hfnet.GetOutput().at(k)<<std::endl;
	}
	std::cout<<std::endl;
}
hfnet.SetInput(fInpHF3);
for(int k = 0; k < 1; k++) {
	hfnet.PropagateFW();

	for(int k = 0; k < hfnet.GetOutput().size(); k++) {
		std::cout<<"outp: "<<hfnet.GetOutput().at(k)<<std::endl;
	}
	std::cout<<std::endl;
}

hfnet.ExpToFS("hftest");
std::cout<<"TEST"<<std::endl;
hfnet.ImpFromFS("hftest");

hfnet.SetInput(fInpHF);
for(int k = 0; k < 1; k++) {
	hfnet.PropagateFW();

	for(int k = 0; k < hfnet.GetOutput().size(); k++) {
		std::cout<<"outp: "<<hfnet.GetOutput().at(k)<<std::endl;
	}
	std::cout<<std::endl;
}
hfnet.SetInput(fInpHF2);
for(int k = 0; k < 1; k++) {
	hfnet.PropagateFW();

	for(int k = 0; k < hfnet.GetOutput().size(); k++) {
		std::cout<<"outp: "<<hfnet.GetOutput().at(k)<<std::endl;
	}
	std::cout<<std::endl;
}
hfnet.SetInput(fInpHF3);
for(int k = 0; k < 1; k++) {
	hfnet.PropagateFW();

	for(int k = 0; k < hfnet.GetOutput().size(); k++) {
		std::cout<<"outp: "<<hfnet.GetOutput().at(k)<<std::endl;
	}
	std::cout<<std::endl;
}

/*
    thrust::host_vector<float> h_vec(1024);
    thrust::sequence(h_vec.begin(), h_vec.end()); // values = indices

    // transfer data to the device
    thrust::device_vector<float> d_vec = h_vec;

    unsigned int index;
    float val = hostGetMax(d_vec, index);
    std::cout <<  "Max index is:" << index <<std::endl;
    std::cout << "Value is: " << val <<std::endl;

    val = hostGetMin(d_vec, index);
	std::cout <<  "Max index is:" << index <<std::endl;
	std::cout << "Value is: " << val <<std::endl;
*/

	return 0;
}
