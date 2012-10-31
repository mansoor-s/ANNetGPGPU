/*
 * main.cpp
 *
 *  Created on: 12.04.2010
 *      Author: dgrat
 */

#include <ANNet>
#include <ANGPGPU>
#include <ANContainers>
#include <ANMath>
#include <math/ANRandom.h>

#include <ctime>
#include <iostream>


int main(int argc, char *argv[]) {
  float fInp1[3];
  fInp1[0] = 1;
  fInp1[1] = 1;
  fInp1[2] = 1;

  float fInp2[3];
  fInp2[0] = 0;
  fInp2[1] = 0;
  fInp2[2] = 0;

  float fInp3[3];
  fInp3[0] = 0;
  fInp3[1] = 1;
  fInp3[2] = 0;

  float fInp4[3];
  fInp4[0] = 0;
  fInp4[1] = 0;
  fInp4[2] = 1;


  float fOut1[6];
  fOut1[0] = 0;
  fOut1[1] = 1;
  fOut1[2] = 0.1;
  fOut1[3] = 0.2;
  fOut1[4] = 0.3;
  fOut1[5] = 0.4;

  float fOut2[6];
  fOut2[0] = 0;
  fOut2[1] = 1;
  fOut2[2] = 0;
  fOut2[3] = 0;
  fOut2[4] = 0;
  fOut2[5] = 0;

  float fOut3[6];
  fOut3[0] = 0;
  fOut3[1] = 1;
  fOut3[2] = 1;
  fOut3[3] = 0;
  fOut3[4] = 0;
  fOut3[5] = 0;

  float fOut4[6];
  fOut4[0] = 0;
  fOut4[1] = 1;
  fOut4[2] = 1;
  fOut4[3] = 1;
  fOut4[4] = 0;
  fOut4[5] = 0;

  //SimpleNet gpu;
  ANN::BPNetGPU gpu;
  ANN::BPNet cpu;
  ANN::BPLayer layer1(3, ANN::ANLayerInput);
  layer1.AddFlag(ANN::ANBiasNeuron);
  ANN::BPLayer layer2(2048, ANN::ANLayerHidden);
  layer2.AddFlag(ANN::ANBiasNeuron);
  ANN::BPLayer layer3(2048, ANN::ANLayerHidden);
  layer3.AddFlag(ANN::ANBiasNeuron);
  ANN::BPLayer layer4(6, ANN::ANLayerOutput);
  layer4.AddFlag(ANN::ANBiasNeuron);

  layer1.ConnectLayer(&layer2);
  layer2.ConnectLayer(&layer3);
  layer3.ConnectLayer(&layer4);

  gpu.AddLayer(&layer1);
  gpu.AddLayer(&layer2);
  gpu.AddLayer(&layer3);
  gpu.AddLayer(&layer4);

  cpu.AddLayer(&layer1);
  cpu.AddLayer(&layer2);
  cpu.AddLayer(&layer3);
  cpu.AddLayer(&layer4);

  ANN::TrainingSet input;
  input.AddInput(fInp1, 3);
  input.AddOutput(fOut1, 6);
  input.AddInput(fInp2, 3);
  input.AddOutput(fOut2, 6);
  input.AddInput(fInp3, 3);
  input.AddOutput(fOut3, 6);
  input.AddInput(fInp4, 3);
  input.AddOutput(fOut4, 6);

  gpu.SetLearningRate(0.05);
  gpu.SetMomentum(0.1);
  gpu.SetWeightDecay(0);
  gpu.SetTrainingSet(input);


  cpu.SetLearningRate(0.05);
  cpu.SetMomentum(0.1);
  cpu.SetWeightDecay(0);
  cpu.SetTrainingSet(input);

  bool b = false;
  float f;
  std::vector<float> errors;
/*
  errors = cpu.TrainFromData(10, 0, b, f);
  cpu.ExpToFS("Foo.bar");
  std::cout<<"cpu: \n"<<cpu<<std::endl;
*/
  errors = gpu.TrainFromData(100, 0.01, b, f);
  gpu.ExpToFS("Foo.bar");
  std::cout<<"gpu: \n"<<gpu<<std::endl;

  return 0;
}
