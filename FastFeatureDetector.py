'''
 Defect Detection Using OpenCV

 Find KeyPoint of the Images and find close 3 points in some distance and draw circle which have center point as one of them

 Normal Image : Find Keypoint with smaller threshold value

 Defect Image : Find Keypoint with Bigger threshold value

 if there are new keypoint at defect images, which is not close to normal image's,
 then there are defect.

'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

# Call Normal Image and Pre-processing
normalImage = cv2.imread('./img/normal.jpg')
normalImage = cv2.resize(normalImage, dsize=(640,480), interpolation = cv2.INTER_AREA)
normalImage = cv2.cvtColor(normalImage, cv2.COLOR_BGR2GRAY)

# Call Defect Image and Pre-processing
defectImage = cv2.imread('./img/defect1.jpg')
defectImage = cv2.resize(defectImage, dsize=(640,480), interpolation = cv2.INTER_AREA)
defectImage = cv2.cvtColor(defectImage,cv2.COLOR_BGR2GRAY)

# Find Key point of Normal Image and show it with smaller threshold Value
fastF_normalImage =cv2.FastFeatureDetector.create(threshold=70)
kp_normalImage = fastF_normalImage.detect(normalImage)
dst_normalImage = cv2.drawKeypoints(normalImage, kp_normalImage, None, color=(0,0,255))
print('len(kp_normalImage)=', len(kp_normalImage))
cv2.imshow('dst_normalImage',  dst_normalImage)

# Find Key point of Defect Image and show it with bigger threshold Value
fastF_defectImage =cv2.FastFeatureDetector.create(threshold=90)
kp_defectImage = fastF_defectImage.detect(defectImage)
dst_defectImage = cv2.drawKeypoints(defectImage, kp_defectImage, None, color=(0,0,255))
print('len(kp_defectImage)=', len(kp_defectImage))
cv2.imshow('dst_defectImage',  dst_defectImage)

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
thresholdDistance1 = 8
thresholdDistance2 = 30
thresholdDistance3 = 50

# Make 2 Lists For save close 3 points
triangleForNormalImage = []
triangleForDefectImage = []

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

            if(total_distance < thresholdDistance1) :
                distanceUnder1 += 1
                triangleForNormalImage.append(list_kp_normalImage[j])
            elif(total_distance < thresholdDistance2) :
                distanceUnder2 += 1
            elif(total_distance < thresholdDistance3) :
                distanceUnder3 += 1
            else :
                distanceOver += 1

# Show How many points are close
print('Sum of distance between 3 points are under ', thresholdDistance1 ,' : ', distanceUnder1)
print('Sum of distance between 3 points are under ', thresholdDistance2 ,' : ', distanceUnder2)
print('Sum of distance between 3 points are under ', thresholdDistance3 ,' : ', distanceUnder3)
print('Over ', thresholdDistance3 ,' : ', distanceOver)

# Remove Same Points in triangleForNormalImage
triangleForNormalImage = list(set(triangleForNormalImage))
print(triangleForNormalImage)
print('# of the components for under ', thresholdDistance1,' : ', len(triangleForNormalImage))

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
    cv2.circle(normalImage, (int(float(triangleForNormalImage[i][0])), int(float(triangleForNormalImage[i][1]))), 10, (0,0,255), 3)

print('After remove close pixels : ', triangleForNormalImage)

cv2.imwrite('./img/withCircle_normal.png', normalImage)
cv2.imshow('normalImage', normalImage)

# Initialize distances Value Again
distanceUnder1 = 0
distanceUnder2 = 0
distanceUnder3 = 0
distanceOver = 0

# Find 3 close points and save them to triangleForDefectImage
for i in range(len(kp_defectImage)-2):
    for j in range(i+1, len(kp_defectImage)-1):
        for m in range(j+1, len(kp_defectImage)):
            x1_distance = list_kp_defectImage[i][0] - list_kp_defectImage[j][0]
            y1_distance = list_kp_defectImage[i][1] - list_kp_defectImage[j][1]

            first_distance = math.sqrt(x1_distance ** 2 + y1_distance ** 2)

            x2_distance = list_kp_defectImage[j][0] - list_kp_defectImage[m][0]
            y2_distance = list_kp_defectImage[j][1] - list_kp_defectImage[m][1]

            second_distance = math.sqrt(x2_distance **2 + y2_distance ** 2)

            total_distance = first_distance + second_distance

            if(total_distance < thresholdDistance1) :
                distanceUnder1 += 1
                triangleForDefectImage.append(list_kp_defectImage[j])
            elif(total_distance < thresholdDistance2) :
                distanceUnder2 += 1
            elif(total_distance < thresholdDistance3) :
                distanceUnder3 += 1
            else :
                distanceOver += 1

# Show How many points are close
print('Sum of distance between 3 points are under ', thresholdDistance1 ,' : ', distanceUnder1)
print('Sum of distance between 3 points are under ', thresholdDistance2 ,' : ', distanceUnder2)
print('Sum of distance between 3 points are under ', thresholdDistance3 ,' : ', distanceUnder3)
print('Over ', thresholdDistance3 ,' : ', distanceOver)

# Remove Same Points in triangleForDefectImage
triangleForDefectImage = list(set(triangleForDefectImage))
print(triangleForDefectImage)
print('# of the components for under ', thresholdDistance1,' : ', len(triangleForDefectImage))

# Ready for pop
lengthOfTriangle = len(triangleForDefectImage)
popList = []

# Find close values and save the index to popList
for i in range(lengthOfTriangle-1):
    for j in range(i+1, lengthOfTriangle):
        distanceXToCheckHowClose = triangleForDefectImage[i][0] - triangleForDefectImage[j][0]
        distanceYToCheckHowClose = triangleForDefectImage[i][1] - triangleForDefectImage[j][1]

        distance = math.sqrt( distanceXToCheckHowClose ** 2 + distanceYToCheckHowClose ** 2 )

        if distance < 10 :
            popList.append(j)

popList = list(set(popList))
popList.sort()
print('Pop List : ', popList)
popNum = 0

for i in range(len(popList)):
    triangleForDefectImage.pop(popList[i]-popNum)
    print('After remove, component of triangle : ', triangleForDefectImage)
    print('remove order : ', popList[i]-popNum)
    popNum += 1

for i in range(len(triangleForDefectImage)):
    cv2.circle(defectImage, (int(float(triangleForDefectImage[i][0])), int(float(triangleForDefectImage[i][1]))), 10, (0,0,255), 3)

print('After remove close pixels : ', triangleForDefectImage)

cv2.imwrite('./img/withCircle_defect1.png', defectImage)
cv2.imshow('defectImage', defectImage)

cv2.waitKey(0)