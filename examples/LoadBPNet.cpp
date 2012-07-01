/*
 * main.cpp
 *
 *  Created on: 12.04.2010
 *      Author: dgrat
 */

#include <ANNet>
#include <ANContainers>
#include <ANMath>
#include <math/ANRandom.h>

#include <ctime>
#include <iostream>


int main(int argc, char *argv[]) {
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

  ANN::TrainingSet input;
  input.AddInput(fInp1, 3);
  input.AddOutput(fOut1, 6);
  input.AddInput(fInp2, 3);
  input.AddOutput(fOut2, 6);
  input.AddInput(fInp3, 3);
  input.AddOutput(fOut3, 6);
  input.AddInput(fInp4, 3);
  input.AddOutput(fOut4, 6);
  
  //SimpleNet net;
  ANN::BPNet net;
  net.ImpFromFS("TEST");

  net.SetTrainingSet(input);
  std::cout<< net <<std::endl;

  return 0;
}
