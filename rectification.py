import cv2
from stereo_vision import  rectify

img1=cv2.imread('./test_images/test1/source0.png')
img2=cv2.imread('./test_images/test1/source1.png')

img1, img2 = rectify(img1, img2)

cv2.imwrite(f"./test_images/rectified/rectified0.png", img1)
cv2.imwrite(f"./test_images/rectified/rectified1.png", img2)
