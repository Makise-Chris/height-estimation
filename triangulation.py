import cv2
import numpy as np
import matplotlib.pyplot as plt
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


def triangulate(uvs1, uvs2, P1, P2):
    uvs1 = np.array(uvs1)
    uvs2 = np.array(uvs2)

    # RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    # P1 = mtx1 @ RT1  # projection matrix for C1
    #
    # RT2 = np.concatenate([R, T], axis=-1)
    # P2 = mtx2 @ RT2  # projection matrix for C2
    def DLT(P1, P2, point1, point2):

        A = [point1[1] * P1[2, :] - P1[1, :],
             P1[0, :] - point1[0] * P1[2, :],
             point2[1] * P2[2, :] - P2[1, :],
             P2[0, :] - point2[0] * P2[2, :]
             ]
        A = np.array(A).reshape((4, 4))
        # print('A: ')
        # print(A)

        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices=False)

        print('Triangulated point: ')
        print(Vh[3, 0:3] / Vh[3, 3])
        return Vh[3, 0:3] / Vh[3, 3]

    p3ds = []
    for uv1, uv2 in zip(uvs1, uvs2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)
    print(p3ds)

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

cv2.imshow('match', matched_img)
cv2.waitKey(0)

matched_kp1 = []
for match in matches[:50]:
    kp1_idx = match.queryIdx
    matched_kp1.append(keypoints_1[kp1_idx])

matched_kp2 = []
for match in matches[:50]:
    kp2_idx = match.trainIdx
    matched_kp2.append(keypoints_2[kp2_idx])

max_value = np.amax(bg_subtraction_img1)
max_loc = np.where(bg_subtraction_img1 == max_value)
head_point = (max_loc[1][0], max_loc[0][0])
foot_point = (max_loc[1][0], max_loc[0][-1])

# phl = find_nearest_keypoint(head_point, matched_kp1)
# phr = find_nearest_keypoint(head_point, matched_kp2)

phl = find_nearest_keypoint(head_point, matched_kp1)
phr = None
for match in matches[:50]:
    if keypoints_1[match.queryIdx] == phl:
        phr = keypoints_2[match.trainIdx]
        break

param_file = cv2.FileStorage()
param_file.open('stereoMap.xml', cv2.FileStorage_READ)
P1 = param_file.getNode('projMtxL').mat()
P2 = param_file.getNode('projMtxR').mat()
mtx1 = param_file.getNode('calibrateMtxL').mat()
mtx2 = param_file.getNode('calibrateMtxR').mat()
R = param_file.getNode('rot').mat()
T = param_file.getNode('trans').mat()

triangulate([phl.pt], [phr.pt], P1, P2)

# ytop = head_point[1]
# dis_phead = abs(int(phl.pt[0]) - int(phr.pt[0]))
# print(dis_phead)
#
# param_file = cv2.FileStorage()
# param_file.open('stereoMap.xml', cv2.FileStorage_READ)
# mtxL = param_file.getNode('projMtxL').mat()
# cy = mtxL[1][2]
# Tx = 125
#
# yman = (cy-ytop) * Tx / dis_phead
# print(yman + 650)
#
img_color = cv2.cvtColor(bg_subtraction_img2, cv2.COLOR_GRAY2BGR)
# cv2.circle(img_color, head_point, 5, (0, 0, 255), 2)
# cv2.circle(img_color, foot_point, 5, (0, 0, 255), 2)
cv2.circle(img_color, (int(phl.pt[0]), int(phl.pt[1])), 5, (0, 0, 255), 2)
cv2.circle(img_color, (int(phr.pt[0]), int(phr.pt[1])), 5, (0, 0, 255), 2)

cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("result", 500, 600)
cv2.imshow('result', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()