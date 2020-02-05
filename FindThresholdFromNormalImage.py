import cv2
import FastFeatureDetector as FFD
import ShapeDetect as SD

inputImage1 = './img/normal1.png'
inputImage2 = './img/defect1.png'

SD.ShapeDetector(inputImage1, inputImage2)

firstImage = './img/rotatedNormalImage.png'
secondImage = './img/rotatedDefectImage.png'

thresholdForNormal, thresholdForDefect = FFD.detectDefectUsingThreshold(firstImage, secondImage)

print('threshold Values : ', thresholdForNormal, ' and ', thresholdForDefect)
