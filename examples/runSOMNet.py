from ANPyNetCPU import *
black 	= vectorFLT([0,0,0])
white 	= vectorFLT([1,1,1])
red 	= vectorFLT([1,0,0])
green 	= vectorFLT([0,1,0])
blue 	= vectorFLT([0,0,1])

trainSet = TrainingSet()
trainSet.AddInput(black)
trainSet.AddInput(white)
trainSet.AddInput(red)
trainSet.AddInput(green)
trainSet.AddInput(blue)

widthMap = 4
heightMap = 1

inpWidth = 3
inpHeight = 1

SOM = SOMNet(inpWidth,inpHeight,widthMap,heightMap)
SOM.SetTrainingSet(trainSet)
SOM.SetLearningRate(0.3)
SOM.Training(1000)

centroids = SOM.CalcCentroids()

for i in centroids:
  # print distances
  print "Distance: "
  print i.m_fEucDist
  
  #print centroid
  print "Centroid: "
  print i # is the same as: print i.m_vCentroid
  # or the same as:
  #for j in i.m_vCentroid:
  #  print j
  #print "Input: "

  # example to copy values from an vectorFLT to python list
  print "Input: "
  list = []
  for j in i.m_vInput:
    list.append(j)
  print list
    