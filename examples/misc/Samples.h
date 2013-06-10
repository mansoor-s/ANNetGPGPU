#ifndef NETSAMPLES_H
#define NETSAMPLES_H

#include <vector>


float fInp1[3] = {0,0,0};
float fInp2[3] = {0,1,0};
float fInp3[3] = {0,0,1};
float fInp4[3] = {1,0,1};
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
float fOut1[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
float fOut2[6] = {0, 1, 0, 0, 0, 0};
float fOut3[6] = {0, 0, 1, 0, 0, 0};
float fOut4[6] = {0, 0, 0, 1, 0, 0};
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
float ared[3] 		= {1,0,0};
float agreen[3] 	= {0,1,0};
float ablue[3] 		= {0,0,1};
float ayellow[3] 	= {1,1,0};
float aorange[3] 	= {1, 0.4, 0.25};
float apurple[3] 	= {1,0,1};
float adk_green[3] 	= {0, 0.5, 0.25};
float adk_blue[3] 	= {0, 0, 0.5};
float awhite[3] 	= {1,1,1};
float ablack[3] 	= {0,0,0};

std::vector<float> red (ared, ared + sizeof(ared) / sizeof(int) );
std::vector<float> green (agreen, agreen + sizeof(agreen) / sizeof(int) );
std::vector<float> blue (ablue, ablue + sizeof(ablue) / sizeof(int) );
std::vector<float> yellow (ayellow, ayellow + sizeof(ayellow) / sizeof(int) );
std::vector<float> orange (aorange, aorange + sizeof(aorange) / sizeof(int) );
std::vector<float> purple (apurple, apurple + sizeof(apurple) / sizeof(int) );
std::vector<float> dk_green (adk_green, adk_green + sizeof(adk_green) / sizeof(int) );
std::vector<float> dk_blue (adk_blue, adk_blue + sizeof(adk_blue) / sizeof(int) );
std::vector<float> black (ablack, ablack + sizeof(ablack) / sizeof(int) );
std::vector<float> white (awhite, awhite + sizeof(awhite) / sizeof(int) );
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
float TR1[16] = {-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,};
float TR2[16] = {-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1};
float TR3[16] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

float fInpHF1[16] = {-1,1,1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,};
float fInpHF2[16] = {-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1};
float fInpHF3[16] = {1,1,1,1,1,-11,1,1,1,1,1,1,-1,1,1,1};;

#endif