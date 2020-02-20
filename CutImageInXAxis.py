import cv2
import numpy as np
import math
import os

# 여백 설정
whiteScreen = 0

#
# Normal Image의 경우 주로 defect가 되는 것이 우측부분이었음.
# 이로인해, 우측부분과 좌측부분을 일정부분 잘라내고 contour를 통한 normal image 검출에서
# 상당히 효과적으로 동작하는 것을 확인함.
# Ratio의 경우는 좌/우의 비율.
# 기존의 이미지는 [0, 100] 으로 설정하면 이미지 그대로 저장됨
# 아래의 경우 [10, 75]로, 좌측 10%, 우측 25%를 잘라내는 경우.
# 이와 같이 잘라낸 처리를 진행 후 OK product 검출 결과, 높은 정확도를 보여주었음.
#

def ShapeDetectForSingleImage(inputImage):
    # Call Image and Get Information for Image
    print('input Image : ', inputImage)

    Image = cv2.imread('./Cosmetic/normal/' + inputImage)
    YPixels, XPixels, channels = Image.shape
    print('Y Pixels : ', YPixels, 'X Pixels : ', XPixels, 'channels : ', channels)

    Ratio = [10, 75]
    ConvertedXmin = Ratio[0] * XPixels / 100
    ConvertedXmax = Ratio[1] * XPixels / 100

    ImageCut = Image[0:YPixels, int(round(ConvertedXmin)):int(round(ConvertedXmax))]

    cv2.imwrite('./Cosmetic/normalCut/' + inputImage, ImageCut)
    return 0

# 이하 ShapeDetectFromNonRotateFigures와 동일

path = "./Cosmetic/normal/"

file_list = os.listdir(path)

rotationFail = 0

for i in range(len(file_list)):
    print(file_list[i])
    check = ShapeDetectForSingleImage('{}'.format(file_list[i]))
    if(check == 1 ):
        rotationFail += 1

print('Fail Rotation : ', rotationFail)
