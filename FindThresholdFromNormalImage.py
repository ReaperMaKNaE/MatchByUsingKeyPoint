import ShapeDetectOneFigure as SDOF
import FastFeatureDetectForPixel as FFDFP



inputImage1 = 'normal1.png'
CutinputImage1 = SDOF.ShapeDetectForSingleImage(inputImage1)
normalPixel = FFDFP.GetPixelLocationForNormal(CutinputImage1)

print(normalPixel)

'''
inputImage2 = 'normal2.jpg'
SDOF.ShapeDetectForSingleImage(inputImage2)

inputImage3 = 'defect1.png'
SDOF.ShapeDetectForSingleImage(inputImage3)

inputImage3 = 'defect2.jpg'
SDOF.ShapeDetectForSingleImage(inputImage3)
'''


'''
firstImage = './img/rotatedNormalImage.png'
secondImage = './img/rotatedDefectImage.png'

thresholdForNormal, thresholdForDefect = FFD.detectDefectUsingThreshold(firstImage, secondImage)

print('threshold Values : ', thresholdForNormal, ' and ', thresholdForDefect)
'''