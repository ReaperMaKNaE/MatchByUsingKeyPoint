'''

    Make Two Figures Set as One Figures Using Shape Detect

    This is only for Toner. Take Contour of Figures and find rectangle which have smallest area.

    Because two figures(normal and defect) rotate with different angle, Make them match by this module

    Keypoint detection should spend too much time, so I use shape detection, make them match as fast.
'''

import cv2
import numpy as np
import math

PI = 3.141592

# 여백
whiteScreen = 10

# Call Normal Image
normal_image = cv2.imread('./img/normal.jpg')
normal_image = cv2.resize(normal_image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(normal_image, cv2.COLOR_BGR2GRAY)
ret, blmage = cv2.threshold(gray,220, 255, cv2.THRESH_BINARY_INV)
blmage = cv2.dilate(blmage, None)
cv2.imshow('blmage', blmage)

# Backup normal image for cut
normalImageForCut = cv2.imread('./img/normal.jpg')
normalImageForCut = cv2.resize(normalImageForCut, dsize=(640, 480), interpolation=cv2.INTER_AREA)

# Find Contours of Figures
mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
contours, hierarchy = cv2.findContours(blmage, mode, method)
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
dst2 = normal_image.copy()
cv2.drawContours(dst2, [cnt], 0, (255, 0, 0), 3)

# After Find Contour, Find Smallest area BOX.
# The pixel number of BOX is saved at box
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int32(box)
dst4 = dst2.copy()
cv2.drawContours(dst4, [box], 0, (0,0,255), 2)
cv2.imshow('dst4', dst4)

# Call Defect Image
defect = cv2.imread('./img/defect1.jpg')
defect = cv2.resize(defect, dsize=(640, 480), interpolation=cv2.INTER_AREA)
gray_defect = cv2.cvtColor(defect, cv2.COLOR_BGR2GRAY)
ret_defect, bImage_defect = cv2.threshold(gray_defect,220, 255, cv2.THRESH_BINARY_INV)
bImage_defect = cv2.dilate(bImage_defect, None)
cv2.imshow('bImage_defect', bImage_defect)

# Backup Defect Image for Image Matching
defectImageForMatching = cv2.imread('./img/defect1.jpg')
defectImageForMatching = cv2.resize(defectImageForMatching, dsize=(640, 480), interpolation=cv2.INTER_AREA)

# Find Contours of Figures
mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
contours_defect, hierarchy = cv2.findContours(bImage_defect, mode, method)
print('len(contours_defect) = ', len(contours_defect))

maxLength_defect = 0
k_defect = 0
for i, cnt in enumerate(contours_defect):
    perimeter = cv2.arcLength(cnt, closed = True)
    if perimeter > maxLength_defect:
        maxLength_defect = perimeter
        k_defect = i
print('maxLength_defect = ', maxLength_defect)
cnt_defect = contours_defect[k_defect]
dst2_defect = defect.copy()
cv2.drawContours(dst2_defect, [cnt], 0, (255, 0, 0), 3)

# After Find Contour, Find Smallest area BOX.
# The pixel number of BOX is saved at box_defect
rect_defect = cv2.minAreaRect(cnt_defect)
box_defect = cv2.boxPoints(rect_defect)
box_defect = np.int32(box_defect)
dst4_defect = dst2_defect.copy()
cv2.drawContours(dst4_defect, [box_defect], 0, (0,0,255), 2)
cv2.imshow('dst4_defect', dst4_defect)

# Calculate With Mathematics
# Calculate How many Rotate of Normal & Defect Image

#
# The order of components of box -> [Left Downside], [Left Upside], [Right Upside], [Right Downside]
#

slopeForHorizontalNormalImage = (box[2][1] - box[1][1]) / (box[2][0] - box[1][0])
slopeForHorizontalDefectImage = (box_defect[2][1] - box_defect[1][1]) / (box_defect[2][0] - box_defect[1][0])

if(slopeForHorizontalNormalImage != slopeForHorizontalDefectImage):
    print('Two Images are not matched. Start Matching...')
    print('Standard image is Normal Image.')

    AngleForNormalImage = math.atan(slopeForHorizontalNormalImage)
    AngleForDefectImage = math.atan(slopeForHorizontalDefectImage)

    print('Angle of Normal Image : ', AngleForNormalImage, ' Angle of Defect Image = ', AngleForDefectImage)

    # If normal image has bigger angle, below variable will be 0. if not, 1
    whichImageHasBiggerAngle = 0

    if(AngleForNormalImage > AngleForDefectImage):
        AngleDiscrepancy = AngleForNormalImage - AngleForDefectImage
    else :
        AngleDiscrepancy = AngleForDefectImage - AngleForNormalImage
        whichImageHasBiggerAngle = 1

    AngleDiscrepancyInDegree = AngleDiscrepancy*180/PI

    print('Difference between two angle : ', AngleDiscrepancy)
    print('Rotate defect Image as ', AngleDiscrepancyInDegree, 'degree')

    rows, cols, channels = defectImageForMatching.shape
    if whichImageHasBiggerAngle == 0:
        M1 = cv2.getRotationMatrix2D((rows/2, cols/2), -AngleDiscrepancyInDegree, 1.0)
    else:
        M1 = cv2.getRotationMatrix2D((rows/2, cols/2), AngleDiscrepancyInDegree, 1.0)
    rotatedDefectImage = cv2.warpAffine(defectImageForMatching, M1, (640, 480))

    cv2.imshow('rotatedDefectImage', rotatedDefectImage)

    # Find Smallest area box
    gray_rotatedDefect = cv2.cvtColor(rotatedDefectImage, cv2.COLOR_BGR2GRAY)
    ret_rotatedDefect, bImage_rotatedDefect = cv2.threshold(gray_rotatedDefect, 220, 255, cv2.THRESH_BINARY_INV)
    bImage_rotatedDefect = cv2.dilate(bImage_rotatedDefect, None)
    cv2.imshow('bImage_rotatedDefect', bImage_rotatedDefect)

    # Find Contours of Figures
    mode = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_SIMPLE
    contours_rotatedDefect, hierarchy_rotatedDefect = cv2.findContours(bImage_rotatedDefect, mode, method)
    print('len(contours_rotatedDefect) = ', len(contours_rotatedDefect))

    maxLength_rotatedDefect = 0
    k_rotatedDefect = 0
    for i, cnt in enumerate(contours_rotatedDefect):
        perimeter = cv2.arcLength(cnt, closed=True)
        if perimeter > maxLength_rotatedDefect:
            maxLength_rotatedDefect = perimeter
            k_rotatedDefect = i
    print('maxLength_defect = ', maxLength_rotatedDefect)
    cnt_rotatedDefect = contours_rotatedDefect[k_rotatedDefect]
    dst2_rotatedDefect = rotatedDefectImage.copy()
    cv2.drawContours(dst2_rotatedDefect, [cnt], 0, (255, 0, 0), 3)

    # After Find Contour, Find Smallest area BOX.
    # The pixel number of BOX is saved at box_defect
    rect_rotatedDefect = cv2.minAreaRect(cnt_rotatedDefect)
    box_rotatedDefect = cv2.boxPoints(rect_rotatedDefect)
    box_rotatedDefect = np.int32(box_rotatedDefect)
    dst4_rotatedDefect = dst2_rotatedDefect.copy()
    cv2.drawContours(dst4_rotatedDefect, [box_rotatedDefect], 0, (0, 0, 255), 2)
    cv2.imshow('dst4_rotatedDefect', dst4_rotatedDefect)
    print('Box point of rotated Image : ', box_rotatedDefect)

    CutRotatedDefectImage = rotatedDefectImage[min([box_rotatedDefect[2][1] - whiteScreen , box_rotatedDefect[3][1] - whiteScreen]):
                                              max([box_rotatedDefect[3][1] + whiteScreen , box_rotatedDefect[0][1] + whiteScreen]),
                                              min([box_rotatedDefect[1][0] - whiteScreen , box_rotatedDefect[0][0] - whiteScreen]):
                                              max([box_rotatedDefect[2][0] + whiteScreen , box_rotatedDefect[3][0] + whiteScreen])]

    print('min and max value For defect image: ', min([box_rotatedDefect[1][0] - whiteScreen , box_rotatedDefect[0][0] - whiteScreen]),
                                              max([box_rotatedDefect[2][0] + whiteScreen , box_rotatedDefect[3][0] + whiteScreen]),
                                              min([box_rotatedDefect[2][1] - whiteScreen , box_rotatedDefect[3][1] - whiteScreen]),
                                              max([box_rotatedDefect[3][1] + whiteScreen , box_rotatedDefect[0][1] + whiteScreen]))

    resizedRotatedDefectImage = cv2.resize(CutRotatedDefectImage, dsize=(640, 480), interpolation=cv2.INTER_AREA)

    CutNormalImage = normalImageForCut[min([box[2][1] - whiteScreen , box[3][1] - whiteScreen]):
                                       max([box[3][1] + whiteScreen , box[0][1] + whiteScreen]),
                                       min([box[1][0] - whiteScreen , box[0][0] - whiteScreen]):
                                       max([box[2][0] + whiteScreen , box[3][0] + whiteScreen])]


    print('min and max value for Normal Image: ', min([box[1][0] - whiteScreen , box[0][0] - whiteScreen]),
                                       max([box[2][0] + whiteScreen , box[3][0] + whiteScreen]),
                                       min([box[2][1] - whiteScreen , box[3][1] - whiteScreen]),
                                       max([box[3][1] + whiteScreen , box[0][1] + whiteScreen]))

    resizedNormalImage = cv2.resize(CutNormalImage, dsize = (640,480), interpolation = cv2.INTER_AREA)

    cv2.imwrite('./img/rotatedNormalImage.png', resizedNormalImage)
    cv2.imwrite('./img/rotatedDefectImage.png', resizedRotatedDefectImage)

    cv2.imshow("Normal Image Cut " , resizedNormalImage)
    cv2.imshow("Defect Image Cut " , resizedRotatedDefectImage)

cv2.waitKey(0)