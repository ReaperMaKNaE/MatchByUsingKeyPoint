import cv2
import FastFeatureDetector as FFD
import ShapeDetectOneFigure as SDOF

inputImage1 = './img/normal1.png'

SDOF.ShapeDetectForSingleImage(inputImage1)
'''
firstImage = './img/rotatedNormalImage.png'
secondImage = './img/rotatedDefectImage.png'

thresholdForNormal, thresholdForDefect = FFD.detectDefectUsingThreshold(firstImage, secondImage)

print('threshold Values : ', thresholdForNormal, ' and ', thresholdForDefect)
'''