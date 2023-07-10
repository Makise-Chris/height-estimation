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

def find_largest_contour(img):
    ret, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    bg = np.zeros_like(img)
    cv2.fillPoly(bg, [max_contour], color=(255, 255, 255))
    return bg

img1=cv2.imread('./test_images/rectified/rectified0.png')
img2=cv2.imread('./test_images/rectified/rectified1.png')

bg_subtraction_img1=cv2.imread('./test_images/bg-subtract/result0.png', cv2.IMREAD_GRAYSCALE)
bg_subtraction_img2=cv2.imread('./test_images/bg-subtract/result1.png', cv2.IMREAD_GRAYSCALE)

bg_subtraction_img1 = find_largest_contour(bg_subtraction_img1)
bg_subtraction_img2 = find_largest_contour(bg_subtraction_img2)

sift = cv2.SIFT_create()
keypoints_1, descriptors_1 = optimize_keypoint(sift, img1, bg_subtraction_img1)
keypoints_2, descriptors_2 = optimize_keypoint(sift, img2, bg_subtraction_img2)

img1_sift = cv2.drawKeypoints(img1,keypoints_1,img1)
img2_sift = cv2.drawKeypoints(img2,keypoints_2,img2)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)
matched_img = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)

# cv2.imshow('result', matched_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

matched_kp1 = []
for match in matches[:40]:
    kp1_idx = match.queryIdx
    matched_kp1.append(keypoints_1[kp1_idx])

matched_kp2 = []
for match in matches[:40]:
    kp2_idx = match.trainIdx
    matched_kp2.append(keypoints_2[kp2_idx])

max_value = np.amax(bg_subtraction_img1)
max_loc = np.where(bg_subtraction_img1 == max_value)
head_point = (max_loc[1][0], max_loc[0][0])
foot_point = (max_loc[1][0], max_loc[0][-1])

max_value1 = np.amax(bg_subtraction_img2)
max_loc1 = np.where(bg_subtraction_img2 == max_value1)
head_point1 = (max_loc1[1][0], max_loc1[0][0])
foot_point1 = (max_loc1[1][0], max_loc1[0][-1])


phl = find_nearest_keypoint(head_point, matched_kp1)
phr = find_nearest_keypoint(head_point1, matched_kp2)

ytop = head_point[1]
dis_phead = abs(int(phl.pt[0]) - int(phr.pt[0]))
print(dis_phead)

param_file = cv2.FileStorage()
param_file.open('stereoMap.xml', cv2.FileStorage_READ)
mtxL = param_file.getNode('cameraMtxL').mat()
cy = mtxL[1][2]
Tx = 125

yman = (cy-ytop) * Tx / dis_phead
print(yman + 650)

img_color = cv2.cvtColor(bg_subtraction_img1, cv2.COLOR_GRAY2BGR)
cv2.circle(img_color, head_point, 5, (0, 0, 255), 2)
cv2.circle(img_color, foot_point, 5, (0, 0, 255), 2)
cv2.circle(img_color, (int(phl.pt[0]), int(phl.pt[1])), 5, (0, 0, 255), 2)
cv2.circle(img_color, (int(phr.pt[0]), int(phr.pt[1])), 5, (0, 0, 255), 2)

cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("result", 500, 600)
cv2.imshow('result', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()