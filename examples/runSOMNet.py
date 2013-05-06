from ANPyNetCPU import *
red = vectorFLT(3)
black = vectorFLT(3)
white = vectorFLT(3)
green = vectorFLT(3)
blue = vectorFLT(3)

red[0] = 1
red[1] = 0
red[2] = 0

black[0] = 0
black[1] = 0
black[2] = 0

white[0] = 1
white[1] = 1
white[2] = 1

green[0] = 0
green[1] = 1
green[2] = 0

blue[0] = 0
blue[1] = 0
blue[2] = 1

trainSet = TrainingSet()
trainSet.AddInput(red);
trainSet.AddInput(black);
trainSet.AddInput(white);
trainSet.AddInput(blue);

widthMap = 4
heightMap = 1

inpWidth = 3
inpHeight = 1

SOM = SOMNet(inpWidth,inpHeight,widthMap,heightMap)
SOM.SetTrainingSet(trainSet);
SOM.Training(1000);

centroids = SOM.CalcCentroids()

for i in centroids:
  print i.GetOutput()