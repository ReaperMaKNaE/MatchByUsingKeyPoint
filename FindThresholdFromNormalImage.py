import ShapeDetectOneFigure as SDOF
import FastFeatureDetectForPixel as FFDFP
import FastFeatureDetector as FFD



#inputImage1 = 'normal1.jpg'
#CutinputImage1 = SDOF.ShapeDetectForSingleImage(inputImage1)

inputImage2 = 'defect4.jpg'
SDOF.ShapeDetectForSingleImage(inputImage2)

'''
inputImage3 = 'normal7.jpg'
SDOF.ShapeDetectForSingleImage(inputImage3)

inputImage3 = 'normal8.jpg'
SDOF.ShapeDetectForSingleImage(inputImage3)

firstImage = './img/Cutnormal1.png'
secondImage = './img/Cutnormal2.jpg'

thresholdForNormal, thresholdForDefect = FFD.detectDefectUsingThreshold(firstImage, secondImage)

print('threshold Values : ', thresholdForNormal, ' and ', thresholdForDefect)
'''