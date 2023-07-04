import cv2
import numpy as np
from camera_calibration import calibrate
def optimize_keypoint(sift, img, bg_subtracttion_img):
    keypoints, descriptors = sift.detectAndCompute(img, None)
    valid_kp = []
    valid_des = []
    for i, point in enumerate(keypoints):
        x, y = int(point.pt[0]), int(point.pt[1])
        if bg_subtracttion_img[y, x] > 0:
            valid_kp.append(point)
            valid_des.append(descriptors[i])
    valid_des = np.array(valid_des)
    return valid_kp, valid_des

def find_nearest_keypoint(target_point, keypoints):
    distances = []
    for point in keypoints:
        x, y = point.pt
        distance = ((x - target_point[0]) ** 2 + (y - target_point[1]) ** 2) ** 0.5
        distances.append(distance)
    sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
    nearest_keypoint = keypoints[sorted_indices[0]]
    return nearest_keypoint

img1=cv2.imread('./test_images/test1/source3.png', cv2.IMREAD_GRAYSCALE)
img2=cv2.imread('./test_images/test1/source4.png', cv2.IMREAD_GRAYSCALE)

bg_subtraction_img1=cv2.imread('./test_images/bg-subtract/result2.png', cv2.IMREAD_GRAYSCALE)
bg_subtraction_img2=cv2.imread('./test_images/bg-subtract/result3.png', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
# keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
# keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
keypoints_1, descriptors_1 = optimize_keypoint(sift, img1, bg_subtraction_img1)
keypoints_2, descriptors_2 = optimize_keypoint(sift, img2, bg_subtraction_img2)

print(descriptors_1.shape)

img1_sift = cv2.drawKeypoints(img1,keypoints_1,img1)
img2_sift = cv2.drawKeypoints(img2,keypoints_2,img2)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)
matched_img = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)

matched_kp1 = []
for match in matches[:50]:
    kp1_idx = match.queryIdx
    matched_kp1.append(keypoints_1[kp1_idx])

matched_kp2 = []
for match in matches[:50]:
    kp2_idx = match.queryIdx
    matched_kp2.append(keypoints_2[kp2_idx])

max_value = np.amax(bg_subtraction_img1)
max_loc = np.where(bg_subtraction_img1 == max_value)
head_point = (max_loc[1][0], max_loc[0][0])
foot_point = (max_loc[1][0], max_loc[0][-1])

phl = find_nearest_keypoint(head_point, matched_kp1)
phr = find_nearest_keypoint(head_point, matched_kp2)

ytop = head_point[1]
dis_phead = abs(int(phl.pt[0]) - int(phr.pt[0]))

ret, mtx, dist, rvecs, tvecs= calibrate('./test_images/chess','chess','png',0.015)
cy = mtx[1][2]
Tx = 70

yman = (cy-ytop) * Tx / dis_phead
print(yman)

img_color = cv2.cvtColor(bg_subtraction_img1, cv2.COLOR_GRAY2BGR)
cv2.circle(img_color, head_point, 5, (0, 0, 255), 2)
cv2.circle(img_color, foot_point, 5, (0, 0, 255), 2)
cv2.circle(img_color, (int(phl.pt[0]), int(phl.pt[1])), 5, (0, 0, 255), 2)
cv2.circle(img_color, (int(phr.pt[0]), int(phr.pt[1])), 5, (0, 0, 255), 2)

cv2.imshow('image', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()