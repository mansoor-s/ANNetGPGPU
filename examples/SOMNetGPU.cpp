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
#include <math/ANRandom.h>
#include <gui/QSOMReader.h>
#include <gpgpu/ANKernels.h>

#include <ctime>
#include <iostream>


int main(int argc, char *argv[]) {
  QApplication a(argc, argv);

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
  int w1 = 32;
  int w2 = 4;

  ANN::SOMNet SOMap;
  SOMap.SetTrainingSet(input);
  SOMap.CreateSOM(3, 1, w1,w1);

  ANN::SOMNetGPU gpu(&SOMap);
  ANN::SOMNetGPU gpu2(&SOMap);
  //	ANN::SOMNetGPU gpu;
  //	gpu.SetTrainingSet(input);
  //	gpu.CreateSOM(3, 1, w1,w1);

  //SOMap.Training(9);

  SOMReader w(w1, w1, w2);
  //for(int x = 0; x < w1*w1; x++) {
  //	ANN::SOMNeuron *pNeur = (ANN::SOMNeuron*)((ANN::SOMLayer*)SOMap.GetOPLayer())->GetNeuron(x);
  //	vCol[0] = pNeur->GetConI(0)->GetValue();
  //	vCol[1] = pNeur->GetConI(1)->GetValue();
  //	vCol[2] = pNeur->GetConI(2)->GetValue();

  //	w.SetField(QPoint(pNeur->GetPosition()[0], pNeur->GetPosition()[1]), vCol );
  //}
  //w.Save("CPU.png");

  // GPU
  gpu.SetConscienceRate(0.1);
  gpu.Training(500);

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
/*
  gpu.ExpToFS("foo2.bar");
  gpu.ImpFromFS("foo2.bar");
  gpu2.ImpFromFS("foo2.bar");

  for(int x = 0; x < w1*w1; x++) {
	  ANN::SOMNeuron *pNeur = (ANN::SOMNeuron*)((ANN::SOMLayer*)gpu2.GetOPLayer())->GetNeuron(x);
	  vCol[0] = pNeur->GetConI(0)->GetValue();
	  vCol[1] = pNeur->GetConI(1)->GetValue();
	  vCol[2] = pNeur->GetConI(2)->GetValue();

	  w.SetField(QPoint(pNeur->GetPosition()[0], pNeur->GetPosition()[1]), vCol );
  }

  w.Save("GPU_2.png");
*/
  return 0;
}
