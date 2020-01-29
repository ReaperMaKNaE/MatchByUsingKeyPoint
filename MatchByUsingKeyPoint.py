import cv2
import matplotlib.pyplot as plt
import numpy as np

img0 = cv2.imread('img/normal.jpg', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('img/defect1.jpg', cv2.IMREAD_GRAYSCALE)
img0 = cv2.resize(img0, dsize=(640, 480), interpolation=cv2.INTER_AREA)
img1 = cv2.resize(img1, dsize=(640, 480), interpolation=cv2.INTER_AREA)
img0 = cv2.Canny(img0, 100, 150)
img1 = cv2.Canny(img1, 100, 150)

detector = cv2.ORB_create(100)
kps0, fea0 = detector.detectAndCompute(img0, None)
kps1, fea1 = detector.detectAndCompute(img1, None)

matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, False)
matches01 = matcher.knnMatch (fea0, fea1, k=2)
matches10 = matcher.knnMatch (fea1, fea0, k=2)

def ratio_test(matches, ratio_thr):
    good_matches = []
    for m in matches:
        ratio = m[0].distance / m[1].distance
        if ratio < ratio_thr:
            good_matches.append(m[0])
    return good_matches

#SMALL -> AGGRESSIVE
RATIO_THR = 0.7
good_matches01 = ratio_test(matches01, RATIO_THR)
good_matches10 = ratio_test(matches10, RATIO_THR)

good_matches10_ = {(m.trainIdx, m.queryIdx) for m in good_matches10}
final_matches = [m for m in good_matches01 if (m.queryIdx, m.trainIdx) in good_matches10_]

dbg_img = cv2.drawMatches(img0, kps0, img1, kps1, final_matches, None)

###get pixel data
# Initialize lists
list_kp1 = []
list_kp2 = []
# For each match...
for mat in final_matches:
    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx
    # x - columnsd
    # y - rows
    # Get the coordinates
    (x1,y1) = kps0[img1_idx].pt
    (x2,y2) = kps1[img2_idx].pt
    # Append to each list
    list_kp1.append((x1, y1))
    list_kp2.append((x2, y2))
    
# 공통점들의 픽셀 좌표 출력
i = 0
list_new1 = []
list_new2 = []
for i in range(4):
    print(list_kp1[i])
    print(list_kp2[i])
    list_new1.append(list_kp1[i]) 
    list_new2.append(list_kp2[i])
###test to get last pixel data
plt.figure()
plt.imshow(dbg_img [:, :, [2,1,0]])
cv2.imwrite('img/common.jpg', dbg_img)
plt.tight_layout()
plt.show()

 ###################################################################### 원근법 짬뽕 시작

#원근법 img0가 되어 img1을 변형함.
show_img = np.copy(img1)
original_img = np.copy(img0)
#src_pts 는 [(x, y), (x ,y), (x, y)]의 구성임
#src_pts = select_points(show_img, 4)
src_pts = np.array(list_new1, np.float32)
#좌하, 좌상, 우상, 우하 순서임
dst_pts = np.array(list_new2, np.float32)
print(src_pts)
print(dst_pts)

perspective_m = cv2.getPerspectiveTransform(src_pts, dst_pts) 
unwarped_img = cv2.warpPerspective(img1, perspective_m, (640, 480))
cv2.imwrite('img/changed.jpg', unwarped_img)
cv2.imwrite('img/criteria.jpg', original_img)

cv2.imshow('result', np.hstack((original_img, unwarped_img)))
cv2.imshow('substract', cv2.subtract(unwarped_img, original_img))
cv2.imwrite('img/substract.jpg', cv2.subtract(unwarped_img, original_img))
k = cv2.waitKey()

cv2.destroyAllWindows()
