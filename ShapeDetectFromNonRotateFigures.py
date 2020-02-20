#
# Cosmetic/defect 폴더에는 Raw images for Defect products 들이 들어있음.
# Cosmetic/normal 폴더에는 Raw images for OK products 들이 들어있음
# Cosmetic/normalCut 폴더에는 CutImageInXAxis를 통해 잘린 Images for OK products 들이 들어있음.
# CosmeticCut/defect 폴더에는 이 python file을 통해 잘린 images for defect products들이 들어있음
# CosmeticCut/normal 폴더에는 CutImageInXAxis를 통해 잘린 image에서 검출된 OK Products 들이 들어있음
#

import cv2
import numpy as np
import math
import os

# 여백 설정
whiteScreen = 0

#
# 사진에서 Image를 얻기 위해 아래와 같은 함수 선언
#
# 전체적인 Mechanism은 맨 아래에 path로 설정한 directory 내부에 있는 파일 이름을 모두 저장하고
# 저장한 file 이름을 반복문(for)을 통해 아래 함수에 지속적으로 넣음
# 그리고 그 해당하는 Image에 Contour를 따고, 사각형을 그려 그 크기만큼 이미지를 잘라냄.
# 잘라낸 이미지는 저장
#
def ShapeDetectForSingleImage(inputImage):

    # Call Image and Get Information for Image
    #
    # 어떤 이미지를 불러왔는지 확인하기 위해 print.
    print('input Image : ', inputImage)

    # 이미지를 불러와서 Y pixel, X pixel, 채널을 받아옴.
    Image = cv2.imread('./Cosmetic/normalCut/' + inputImage)
    YPixels, XPixels, channels = Image.shape
    print('Y Pixels : ', YPixels, 'X Pixels : ', XPixels, 'channels : ', channels)

    # Back up Original Image
    # 잘라서 저장할 이미지를 backup 해놓기 위해 copy함.
    ImageForCut = Image.copy()

    # Make image downscale and grayscale to get contour
    Image = cv2.resize(Image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    ImageCheck = Image.copy()
    gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

    #
    # Defect Image와 Normal Image는 imshow, waitKey를 통해 어떤 contour가 계속 되는지 확인해야함.
    # defect image의 경우 143~145면 나름 이쁘게 잡힘. (이미지 몇개는 제대로 안되어서 지웠음)
    # normal Image의 경우 좌/우를 일정 ratio로 자른 이후 contour 범위가 152~155이면 나름 이쁘게 잡힘.
    #

    # For Defect Image
    #ret, binaryImage = cv2.threshold(gray, 143, 145, cv2.THRESH_BINARY_INV)
    # For normal Image
    ret, binaryImage = cv2.threshold(gray, 152, 155, cv2.THRESH_BINARY_INV)
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

    x, y, w, h = cv2.boundingRect(cnt)
    print('x, y, w, h is ', x, ' and ', y, ' and ',  w, ' and ',  h)
    img1 = cv2.rectangle(ImageCheck, (x, y), (x + w, y + h),(0,0,255), 5)

    #
    # Contour한 이미지를 rectangle로 감싼 형태를 아래 주석을 해제하면 확인이 가능(저장될 이미지)
    #
    #cv2.imshow('img1', img1)

    #
    # 아래는 픽셀계산. 원래 기존의 이미지를 640*480으로 dsize하여 계산하였기 때문에,
    # 비율을 맞춰주기 위해 아래와 같이 곱해서 계산함.
    # 아래 int(round) 형태를 사용한 이유는, 비율을 곱하다보니 float형태로 저장이 되기에,
    # round를 하여 반올림하고 int형태로 해야 정확한 픽셀값이 나와 이미지를 잘라낼 수 있음.
    #
    XPixelsForCut = []

    YPixelsForCut = []

    XPixelsForCut.append( ( x * XPixels / 640, ( x + w ) * XPixels / 640 ) )

    YPixelsForCut.append( ( y * YPixels / 480, ( y + h ) * YPixels / 480 ) )

    print('min and max value for Image : ', XPixelsForCut, YPixelsForCut)

    print('round number = ', int(round(XPixelsForCut[0][0])),round(XPixelsForCut[0][1]),
                                round(YPixelsForCut[0][0]),round(YPixelsForCut[0][1]), 'For Image')

    CutImage = ImageForCut[int(round(YPixelsForCut[0][0])):int(round(YPixelsForCut[0][1])),
                        int(round(XPixelsForCut[0][0])):int(round(XPixelsForCut[0][1]))]

    #
    # 잘린 이미지의 정보를 확인하기 위해 shape을 쓰고 출력 및 이미지 저장.
    #

    rowsCut, colsCut, channelsCut = CutImage.shape

    print('Pixel Size for images : ', rowsCut , ' X', colsCut, ' and channels : ', channelsCut)
    cv2.imwrite('./CosmeticCut/normal/' + inputImage, CutImage)

    return 0

#
# Image File들이 들어있는 Directory 설정 및 Image File들을 file_list에 저장
#
path = "./Cosmetic/normalCut/"

file_list = os.listdir(path)

#
# 저장에 실패한 경우 1인데, 이전에 남아있던 잔존이라 의미없는 값. 변수 rotationFail 관련은 무시하여도 무방.
#

rotationFail = 0

for i in range(len(file_list)):
    print(file_list[i])
    check = ShapeDetectForSingleImage('{}'.format(file_list[i]))
    if(check == 1 ):
        rotationFail += 1
#
# 얼마나 많이 저장을 실패하였는가를 의미. 무시하여도 무방.
#
print('Fail Rotation : ', rotationFail)