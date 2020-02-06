import cv2
import numpy as np
import math

def GetPixelLocationForNormal(inputImage, thresholdValue = 20):
    # Call Image
    Image = cv2.imread(inputImage)

    # Backup Color Image
    Image_color = Image.copy()

    # Resize both Image
    Image = cv2.resize(Image, dsize=(640,480), interpolation = cv2.INTER_AREA)
    Image_color = cv2.resize(Image_color, dsize=(640,480), interpolation = cv2.INTER_AREA)

    # Make One of them Grayscale
    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

    # Find Key point of Normal Image and show it with smaller threshold Value
    fastF_normalImage =cv2.FastFeatureDetector.create(threshold=thresholdValue)
    kp_normalImage = fastF_normalImage.detect(Image)
    dst_normalImage = cv2.drawKeypoints(Image, kp_normalImage, None, color=(0,0,255))
    print('len(kp_normalImage)=', len(kp_normalImage))
    cv2.imshow('dst_normalImage', dst_normalImage)
    cv2.imwrite('./img/dst_normalImage.png', dst_normalImage)

    # Set 2 Lists to save pixel values for keypoint
    list_kp_normalImage = []

    # Save pixel values for keypoint
    for keypoint in kp_normalImage:
        list_kp_normalImage.append(keypoint.pt)

    # Initialize distances Value
    distanceUnder1 = 0
    distanceUnder2 = 0
    distanceUnder3 = 0
    distanceOver = 0

    # Initialize Threshold Distance Values
    thresholdDistance1 = 10
    thresholdDistance2 = 30
    thresholdDistance3 = 50

    # Make 2 Lists For save close 3 points
    triangleForNormalImage = []

    # Find 3 close points and save them to triangleForNormalImage
    for i in range(len(kp_normalImage)-2):
        for j in range(i+1, len(kp_normalImage)-1):
            for m in range(j+1, len(kp_normalImage)):
                x1_distance = list_kp_normalImage[i][0] - list_kp_normalImage[j][0]
                y1_distance = list_kp_normalImage[i][1] - list_kp_normalImage[j][1]

                first_distance = math.sqrt(x1_distance ** 2 + y1_distance ** 2)

                x2_distance = list_kp_normalImage[j][0] - list_kp_normalImage[m][0]
                y2_distance = list_kp_normalImage[j][1] - list_kp_normalImage[m][1]

                second_distance = math.sqrt(x2_distance **2 + y2_distance ** 2)

                total_distance = first_distance + second_distance

                if (total_distance < thresholdDistance1):
                    distanceUnder1 += 1
                    triangleForNormalImage.append(list_kp_normalImage[j])
                elif (total_distance < thresholdDistance2):
                    distanceUnder2 += 1
                elif (total_distance < thresholdDistance3):
                    distanceUnder3 += 1
                else:
                    distanceOver += 1

    # Show How many points are close
    print('Sum of distance between 3 points are under ', thresholdDistance1, ' : ', distanceUnder1)
    print('Sum of distance between 3 points are under ', thresholdDistance2, ' : ', distanceUnder2)
    print('Sum of distance between 3 points are under ', thresholdDistance3, ' : ', distanceUnder3)
    print('Over ', thresholdDistance3, ' : ', distanceOver)

    # Remove Same Points in triangleForNormalImage
    triangleForNormalImage = list(set(triangleForNormalImage))
    print(triangleForNormalImage)
    print('# of the components for under ', thresholdDistance1,' : ', len(triangleForNormalImage))
    imgcheck = Image_color.copy()
    for i in range(len(triangleForNormalImage)):
        cv2.circle(imgcheck, (int(float(triangleForNormalImage[i][0])), int(float(triangleForNormalImage[i][1]))), 10, (0,0,255), 3)
    cv2.imshow('imgcheck', imgcheck)

    # Ready for pop
    lengthOfTriangle = len(triangleForNormalImage)
    popList = []

    # Find close values and save the index to popList
    for i in range(lengthOfTriangle-1):
        for j in range(i+1, lengthOfTriangle):
            distanceXToCheckHowClose = triangleForNormalImage[i][0] - triangleForNormalImage[j][0]
            distanceYToCheckHowClose = triangleForNormalImage[i][1] - triangleForNormalImage[j][1]

            distance = math.sqrt( distanceXToCheckHowClose ** 2 + distanceYToCheckHowClose ** 2 )

            if distance < 10 :
                popList.append(j)

    popList = list(set(popList))
    popList.sort()
    print('Pop List : ', popList)
    popNum = 0

    for i in range(len(popList)):
        triangleForNormalImage.pop(popList[i]-popNum)
        print('After remove, component of triangle : ', triangleForNormalImage)
        print('remove order : ', popList[i]-popNum)
        popNum += 1

    for i in range(len(triangleForNormalImage)):
        cv2.circle(Image_color, (int(float(triangleForNormalImage[i][0])), int(float(triangleForNormalImage[i][1]))), 10, (0,0,255), 3)

    # Draw Lines For compare distance
    cv2.line(Image_color, (10,10), (40,10), (0,255,0), 5)

    print('After remove close pixels : ', triangleForNormalImage)

    cv2.imwrite('./img/withCircle_Normal_1.png', Image_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return triangleForNormalImage