# 0901.py
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
 
src = cv2.imread('./img/defect1.jpg')
src = cv2.resize(src, dsize=(640,480), interpolation = cv2.INTER_AREA)
gray= cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

#1
##fastF = cv2.FastFeatureDetector_create()
##fastF =cv2.FastFeatureDetector.create()
fastF =cv2.FastFeatureDetector.create(threshold=90) # 100
kp = fastF.detect(gray) 
dst = cv2.drawKeypoints(gray, kp, None, color=(0,0,255))
print('len(kp)=', len(kp))
cv2.imshow('dst',  dst)

list_kp1 = []

for keypoint in kp:
    list_kp1.append(keypoint.pt)

distanceUnder10 = 0
distanceUnder30 = 0
distanceUnder50 = 0
distanceOver50 = 0

triangleToIndicateDefect=[]

for i in range(len(kp)-2):
    for j in range(i+1, len(kp)-1):
        for m in range(j+1, len(kp)):
            x1_distance = list_kp1[i][0] - list_kp1[j][0]
            y1_distance = list_kp1[i][1] - list_kp1[j][1]

            first_distance = math.sqrt(x1_distance ** 2 + y1_distance ** 2)

            x2_distance = list_kp1[j][0] - list_kp1[m][0]
            y2_distance = list_kp1[j][1] - list_kp1[m][1]

            second_distance = math.sqrt(x2_distance **2 + y2_distance ** 2)

            total_distance = first_distance + second_distance

            if(total_distance < 8) :
                distanceUnder10 = distanceUnder10 + 1
                triangleToIndicateDefect.append(list_kp1[j])
            elif(total_distance < 30) :
                distanceUnder30 = distanceUnder30 + 1
            elif(total_distance < 50) :
                distanceUnder50 = distanceUnder50 + 1
            else :
                distanceOver50 = distanceOver50 + 1

print('Sum of distance between 3 points are under 10 : ', distanceUnder10)
print('Sum of distance between 3 points are under 30 : ', distanceUnder30)
print('Sum of distance between 3 points are under 50 : ', distanceUnder50)
print('Sum of distance between 3 points are Over 50 : ',distanceOver50)

triangleToIndicateDefect = list(set(triangleToIndicateDefect))
print(triangleToIndicateDefect)
print('# of the components for under 10 : ', len(triangleToIndicateDefect))

for i in range(len(triangleToIndicateDefect)):
    cv2.circle(src, (int(float(triangleToIndicateDefect[i][0])), int(float(triangleToIndicateDefect[i][1]))), 10, (0,0,255), 3)

cv2.imwrite('./img/withCircle_defect1.png', src)
cv2.imshow('src', src)

cv2.waitKey(0)
cv2.destroyAllWindows()