import cv2
import math

firstImage = './img/rotatedNormalImage.png'
secondImage = './img/rotatedDefectImage.png'

thresholdValue1 = 20
thresholdValue2 = 30

defect = 0
popListDistance = 10
threshold1 = 140
threshold2 = 150
defectDetectDistance = 20

while (defect == 0):
    if (threshold1 < thresholdValue1 & threshold2 < thresholdValue2):
        break

    # Call Normal Image and Pre-processing
    normalImage = cv2.imread(firstImage)
    normalImage = cv2.resize(normalImage, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    normalImage = cv2.cvtColor(normalImage, cv2.COLOR_BGR2GRAY)

    # Backup Color Normal Image
    normalImage_color = cv2.imread(firstImage)
    normalImage_color = cv2.resize(normalImage_color, dsize=(640, 480), interpolation=cv2.INTER_AREA)

    # Call Defect Image and Pre-processing
    defectImage = cv2.imread(secondImage)
    defectImage = cv2.resize(defectImage, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    defectImage = cv2.cvtColor(defectImage, cv2.COLOR_BGR2GRAY)

    # Backup Color Defect Image
    defectImage_color = cv2.imread(secondImage)
    defectImage_color = cv2.resize(defectImage_color, dsize=(640, 480), interpolation=cv2.INTER_AREA)

    # Backup Color Defect Image
    defectImage_color_Defect = cv2.imread(secondImage)
    defectImage_color_Defect = cv2.resize(defectImage_color_Defect, dsize=(640, 480), interpolation=cv2.INTER_AREA)

    # Find Key point of Normal Image and show it with smaller threshold Value
    fastF_normalImage = cv2.FastFeatureDetector.create(threshold=threshold1)
    kp_normalImage = fastF_normalImage.detect(normalImage)
    dst_normalImage = cv2.drawKeypoints(normalImage, kp_normalImage, None, color=(0, 0, 255))
    print('len(kp_normalImage)=', len(kp_normalImage))
    cv2.imshow('dst_normalImage', dst_normalImage)
    cv2.imwrite('./img/dst_normalImage.png', dst_normalImage)

    # Find Key point of Defect Image and show it with bigger threshold Value
    fastF_defectImage = cv2.FastFeatureDetector.create(threshold=threshold2)
    kp_defectImage = fastF_defectImage.detect(defectImage)
    dst_defectImage = cv2.drawKeypoints(defectImage, kp_defectImage, None, color=(0, 0, 255))
    print('len(kp_defectImage)=', len(kp_defectImage))
    cv2.imshow('dst_defectImage', dst_defectImage)
    cv2.imwrite('./img/dst_defectImage.png', dst_defectImage)

    # Set 2 Lists to save pixel values for keypoint
    list_kp_normalImage = []
    list_kp_defectImage = []

    # Save pixel values for keypoint
    for keypoint in kp_normalImage:
        list_kp_normalImage.append(keypoint.pt)

    for keypoint in kp_defectImage:
        list_kp_defectImage.append(keypoint.pt)

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
    triangleForDefectImage = []

    # Find 3 close points and save them to triangleForNormalImage
    for i in range(len(kp_normalImage) - 2):
        for j in range(i + 1, len(kp_normalImage) - 1):
            for m in range(j + 1, len(kp_normalImage)):
                x1_distance = list_kp_normalImage[i][0] - list_kp_normalImage[j][0]
                y1_distance = list_kp_normalImage[i][1] - list_kp_normalImage[j][1]

                first_distance = math.sqrt(x1_distance ** 2 + y1_distance ** 2)

                x2_distance = list_kp_normalImage[j][0] - list_kp_normalImage[m][0]
                y2_distance = list_kp_normalImage[j][1] - list_kp_normalImage[m][1]

                second_distance = math.sqrt(x2_distance ** 2 + y2_distance ** 2)

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
    print('# of the components for under ', thresholdDistance1, ' : ', len(triangleForNormalImage))
    imgcheck = normalImage_color.copy()
    for i in range(len(triangleForNormalImage)):
        cv2.circle(imgcheck, (int(float(triangleForNormalImage[i][0])), int(float(triangleForNormalImage[i][1]))), 10,
                   (0, 0, 255), 3)
    cv2.imshow('imgcheck', imgcheck)

    # Ready for pop
    lengthOfTriangle = len(triangleForNormalImage)
    popList = []

    # Find close values and save the index to popList
    for i in range(lengthOfTriangle - 1):
        for j in range(i + 1, lengthOfTriangle):
            distanceXToCheckHowClose = triangleForNormalImage[i][0] - triangleForNormalImage[j][0]
            distanceYToCheckHowClose = triangleForNormalImage[i][1] - triangleForNormalImage[j][1]

            distance = math.sqrt(distanceXToCheckHowClose ** 2 + distanceYToCheckHowClose ** 2)

            if distance < 10:
                popList.append(j)

    popList = list(set(popList))
    popList.sort()
    print('Pop List : ', popList)
    popNum = 0

    for i in range(len(popList)):
        triangleForNormalImage.pop(popList[i] - popNum)
        print('After remove, component of triangle : ', triangleForNormalImage)
        print('remove order : ', popList[i] - popNum)
        popNum += 1

    for i in range(len(triangleForNormalImage)):
        cv2.circle(normalImage_color,
                   (int(float(triangleForNormalImage[i][0])), int(float(triangleForNormalImage[i][1]))), 10,
                   (0, 0, 255), 3)

    # Draw Lines For compare distance
    cv2.line(normalImage_color, (10, 10), (40, 10), (0, 255, 0), 5)

    print('After remove close pixels : ', triangleForNormalImage)

    cv2.imwrite('./img/withCircle_Normal_1.png', normalImage_color)

    # Initialize distances Value Again
    distanceUnder1 = 0
    distanceUnder2 = 0
    distanceUnder3 = 0
    distanceOver = 0

    # Find 3 close points and save them to triangleForDefectImage
    for i in range(len(kp_defectImage) - 3):
        for j in range(i + 1, len(kp_defectImage) - 2):
            for m in range(j + 1, len(kp_defectImage) - 1):
                for k in range(m+1, len(kp_defectImage)):
                    x1_distance = list_kp_defectImage[i][0] - list_kp_defectImage[j][0]
                    y1_distance = list_kp_defectImage[i][1] - list_kp_defectImage[j][1]

                    first_distance = math.sqrt(x1_distance ** 2 + y1_distance ** 2)

                    x2_distance = list_kp_defectImage[j][0] - list_kp_defectImage[m][0]
                    y2_distance = list_kp_defectImage[j][1] - list_kp_defectImage[m][1]

                    second_distance = math.sqrt(x2_distance ** 2 + y2_distance ** 2)

                    x3_distance = list_kp_defectImage[m][0] - list_kp_defectImage[k][0]
                    y3_distance = list_kp_defectImage[m][1] - list_kp_defectImage[k][1]

                    third_distance = math.sqrt(x3_distance ** 2 + y3_distance ** 2)

                    total_distance = first_distance + second_distance + third_distance

                    if (total_distance < thresholdDistance1):
                        distanceUnder1 += 1
                        triangleForDefectImage.append(list_kp_defectImage[j])
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

    # Remove Same Points in triangleForDefectImage
    triangleForDefectImage = list(set(triangleForDefectImage))
    print('defect Point : ', triangleForDefectImage)
    print('# of the components for under ', thresholdDistance1, ' : ', len(triangleForDefectImage))

    imgcheck_defect = defectImage_color.copy()
    for i in range(len(triangleForDefectImage)):
        cv2.circle(imgcheck_defect,
                   (int(float(triangleForDefectImage[i][0])), int(float(triangleForDefectImage[i][1]))), 10,
                   (0, 0, 255), 3)
    cv2.imshow('imgcheck_defect', imgcheck_defect)

    # Ready for pop
    lengthOfTriangle = len(triangleForDefectImage)
    popList = []

    # Find close values and save the index to popList
    for i in range(lengthOfTriangle - 1):
        for j in range(i + 1, lengthOfTriangle):
            distanceXToCheckHowClose = triangleForDefectImage[i][0] - triangleForDefectImage[j][0]
            distanceYToCheckHowClose = triangleForDefectImage[i][1] - triangleForDefectImage[j][1]

            distance = math.sqrt(distanceXToCheckHowClose ** 2 + distanceYToCheckHowClose ** 2)

            if distance < popListDistance:
                popList.append(j)

    popList = list(set(popList))
    popList.sort()
    print('Pop List : ', popList)
    popNum = 0

    for i in range(len(popList)):
        triangleForDefectImage.pop(popList[i] - popNum)
        print('After remove, component of triangle : ', triangleForDefectImage)
        print('remove order : ', popList[i] - popNum)
        popNum += 1

    # Draw Keypoints for defectImage
    for i in range(len(triangleForDefectImage)):
        cv2.circle(defectImage_color,
                   (int(float(triangleForDefectImage[i][0])), int(float(triangleForDefectImage[i][1]))), 10,
                   (0, 0, 255), 3)

    print('After remove close pixels : ', triangleForDefectImage)

    # Draw Lines For compare distance
    cv2.line(defectImage, (10, 10), (40, 10), (0, 255, 0), 5)

    cv2.imwrite('./img/withCircle_Defect_1.png', defectImage_color)

    # Record Defect Part
    DefectPart = []

    for i in range(len(triangleForDefectImage)):
        DefectPart.append(1)

    for i in range(len(triangleForDefectImage)):
        for j in range(len(triangleForNormalImage)):
            distanceXBetweenNormalAndDefect = triangleForDefectImage[i][0] - triangleForNormalImage[j][0]
            distanceYBetweenNormalAndDefect = triangleForDefectImage[i][1] - triangleForNormalImage[j][1]

            distanceBetweenNormalAndDefect = math.sqrt(
                distanceXBetweenNormalAndDefect ** 2 + distanceYBetweenNormalAndDefect ** 2)

            if (distanceBetweenNormalAndDefect < defectDetectDistance):
                DefectPart[i] = 0

    defectIndex = []

    for i in range(len(triangleForDefectImage)):
        if (DefectPart[i] == 1):
            defect = 1
            defectIndex.append(i)

    if defect == 1:
        print('Defect is Detected!!')
        for i in range(len(defectIndex)):
            print('Defect Point : ', triangleForDefectImage[defectIndex[i]])
            cv2.circle(defectImage_color_Defect,
                       (int(float(triangleForDefectImage[i][0])), int(float(triangleForDefectImage[i][1]))), 20,
                       (0, 255, 255), 3)
            cv2.line(defectImage_color_Defect, (10, 10), (10 + defectDetectDistance, 10), (0, 255, 0), 5)
            cv2.imwrite('./img/defectImage_color_Defect.png', defectImage_color_Defect)
            cv2.imshow('defectImage_color_defect', defectImage_color_Defect)
    else:
        threshold1 -= 5
        threshold2 -= 5

    cv2.waitKey(0)
    cv2.destroyAllWindows()
