/*
 * main.cpp
 *
 *  Created on: 12.04.2010
 *      Author: dgrat
 */

#include <ANNet>
#include <ANContainers>
#include <ANMath>

#include <ctime>
#include <iostream>


int main(int argc, char *argv[]) {
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

  return 0;
}
