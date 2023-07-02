import cv2
import numpy as np
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
matched_img = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:70], img2, flags=2)
cv2.imshow('image', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()