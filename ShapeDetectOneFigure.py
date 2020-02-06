'''

    Make Two Figures Set as One Figures Using Shape Detect

    This is only for Rectangle Shape. Take Contour of Figures and find rectangle which have smallest area.

    Make horizontal if the figures are tilting.

    Keypoint detection should spend too much time, so I use shape detection, make them match as fast.
'''

import cv2
import numpy as np
import math

PI = 3.141592

# 여백
whiteScreen = 10

def ShapeDetectForSingleImage(inputImage):

    # Call Image and Get Information for Image

    Image = cv2.imread('./img/' + inputImage)
    YPixels, XPixels, channels = Image.shape
    print('Y Pixels : ', YPixels, 'X Pixels : ', XPixels, 'channels : ', channels)

    # Back up Original Image

    ImageForCut = Image.copy()

    # Make image downscale and grayscale to get contour

    Image = cv2.resize(Image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    ret, binaryImage = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    binaryImage = cv2.dilate(binaryImage, None)

    # Find Contour of Image

    mode = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(binaryImage, mode, method)
    print('len(contours) = ', len(contours))

    maxLength = 0
    k = 0

    for i, cnt in enumerate(contours):
        perimeter = cv2.arcLength(cnt, closed = True)
        if perimeter > maxLength:
            maxLength = perimeter
            k = i
    print('maxLength = ', maxLength)
    cnt = contours[k]
    dst = Image.copy()
    cv2.drawContours(dst, [cnt], 0, (255, 0, 0), 3)

    # After Find Contour, Find Smallest area BOX.
    # The pixel number of BOX is saved at box
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    dst1 = dst.copy()
    cv2.drawContours(dst1, [box], 0, (0,0,255), 2)
    cv2.imshow('Image After Contour', dst1)
    cv2.imwrite('./img/Contoured' + inputImage, dst1)

    # Print Coordinate for Normal Image

    print(' Box Coordinate for normal : ', box)

    # Calculate With Mathematics
    # Calculate How many Rotate of Normal & Defect Image

    #
    # The order of components of box -> [Left Downside], [Left Upside], [Right Upside], [Right Downside]
    #

    slopeForHorizontalImage = (box[2][1] - box[1][1]) / (box[2][0] - box[1][0])

    if(slopeForHorizontalImage != 0):
        print('Images are not on horizontal... Start Make it horizontal.')

        AngleForImage = math.atan(slopeForHorizontalImage)

        ImageForSmallScreen = ImageForCut.copy()
        ImageForSmallScreen = cv2.resize(ImageForSmallScreen, dsize=(640, 480), interpolation=cv2.INTER_AREA)

        YPixelsForSmall, XPixelsForSmall, _ = ImageForSmallScreen.shape

        print('Angle of Image : ', AngleForImage)

        # If tilting angle of product has bigger than 0, below variable will be 0. if not, 1

        if(AngleForImage > 0):
            AngleForImage = -AngleForImage

        AngleForImageInDegree = AngleForImage*180/PI
        print('Angle in degree : ', AngleForImageInDegree)

        # Rotates Figures

        M1 = cv2.getRotationMatrix2D((YPixelsForSmall/2, XPixelsForSmall/2), AngleForImageInDegree, 1.0)
        rotatedImage = cv2.warpAffine(ImageForSmallScreen, M1, (640, 480))

        # Show Rotated Image

        cv2.imshow('Rotated Image : ', rotatedImage)

        # Rotates Image be cut

        rotatedImageForCut = cv2.warpAffine(ImageForCut, M1, (XPixels, YPixels))

        # Find Smallest area box

        grayRotatedImage = cv2.cvtColor(rotatedImageForCut, cv2.COLOR_BGR2GRAY)
        retRotatedImage, binaryRotatedImage = cv2.threshold(grayRotatedImage, 220, 255, cv2.THRESH_BINARY_INV)
        binaryRotatedImage = cv2.dilate(binaryRotatedImage, None)

        # Find Contours of Figures
        mode = cv2.RETR_EXTERNAL
        method = cv2.CHAIN_APPROX_SIMPLE
        contoursRotatedImage, hierarchyRotatedImage = cv2.findContours(binaryRotatedImage, mode, method)
        print('len(contoursRotatedImage) = ', len(contoursRotatedImage))

        maxLengthRotatedImage = 0
        kRotatedImage = 0
        for i, cntRotatedImage in enumerate(contoursRotatedImage):
            perimeter = cv2.arcLength(cntRotatedImage, closed=True)
            if perimeter > maxLengthRotatedImage:
                maxLengthRotatedImage = perimeter
                kRotatedImage = i
        print('maxLengthRotatedImage = ', maxLengthRotatedImage)
        cntRotatedImage = contoursRotatedImage[kRotatedImage]
        dst2RotatedImage = rotatedImageForCut.copy()
        cv2.drawContours(dst2RotatedImage, [cntRotatedImage], 0, (255, 0, 0), 3)

        # After Find Contour, Find Smallest area BOX.
        # The pixel number of BOX is saved at box_defect
        rectRotatedImage = cv2.minAreaRect(cntRotatedImage)
        boxRotatedImage = cv2.boxPoints(rectRotatedImage)
        boxRotatedImage = np.int32(boxRotatedImage)
        dst4RotatedImage = dst2RotatedImage.copy()
        cv2.drawContours(dst4RotatedImage, [boxRotatedImage], 0, (0, 0, 255), 2)
        print('Box point of rotated Image : ', boxRotatedImage)
        cv2.imshow('Box point for rotated Image ', dst4RotatedImage)

        # Get Pixel Information and Rotates

        XPixelsForCut = []
        YPixelsForCut = []

        XPixelsForCut.append((min([boxRotatedImage[2][1] - whiteScreen , boxRotatedImage[3][1] - whiteScreen]),
                              max([boxRotatedImage[1][1] + whiteScreen , boxRotatedImage[0][1] - whiteScreen])))

        YPixelsForCut.append((min([boxRotatedImage[1][0] - whiteScreen , boxRotatedImage[0][0] - whiteScreen]),
                             max([boxRotatedImage[2][0] + whiteScreen, boxRotatedImage[3][0] + whiteScreen])))

        print('min and max value for Image : ', XPixelsForCut, YPixelsForCut)

        print('round number = ', round(XPixelsForCut[0][0]),round(XPixelsForCut[0][1]),
                                 round(YPixelsForCut[0][0]),round(YPixelsForCut[0][1]), 'For Image')

        CutRotatedImage = rotatedImageForCut[round(XPixelsForCut[0][0]):round(XPixelsForCut[0][1]),
                                       round(YPixelsForCut[0][0]):round(YPixelsForCut[0][1])]

        rowsRotated, colsRotated, channelsRotated = CutRotatedImage.shape

        print('Pixel Size for images : ', rowsRotated , ' X', colsRotated, ' and channels : ', channelsRotated)

        cv2.imshow('CutRotatedImage', CutRotatedImage)

        cv2.imwrite('./img/Cut' + inputImage, CutRotatedImage)

        return './img/Cut'+inputImage

    else :
        XPixelsForCut = []
        YPixelsForCut = []

        XPixelsForCut.append((min([box[2][1] - whiteScreen, box[3][1] - whiteScreen])
                              * YPixels / 480,
                              max([box[3][1] + whiteScreen, box[0][1] - whiteScreen])
                              * YPixels / 480))

        YPixelsForCut.append((min([box[1][0] - whiteScreen, box[0][0] - whiteScreen])
                              * XPixels / 640,
                              max([box[2][0] + whiteScreen, box[3][0] + whiteScreen])
                              * XPixels / 640))

        print('min and max value for Image : ', XPixelsForCut, YPixelsForCut)

        print('round number = ', int(round(XPixelsForCut[0][0])),round(XPixelsForCut[0][1]),
                                     round(YPixelsForCut[0][0]),round(YPixelsForCut[0][1]), 'For Image')

        CutImage = ImageForCut[int(round(XPixelsForCut[0][0])):int(round(XPixelsForCut[0][1])),
                               int(round(YPixelsForCut[0][0])):int(round(YPixelsForCut[0][1]))]

        rowsCut, colsCut, channelsCut = CutImage.shape

        print('Pixel Size for images : ', rowsCut , ' X', colsCut, ' and channels : ', channelsCut)

        cv2.imwrite('./img/Cut' + inputImage, CutImage)

        return './img/Cut'+inputImage