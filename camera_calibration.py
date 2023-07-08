import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
def calibrate(dirpath, prefix, image_format, square_size, width=9, height=6):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    #objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]

    images = glob.glob(dirpath+'/' + prefix + '*.' + image_format)

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img", 500, 600)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners =  cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    error = reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)

    return [ret, mtx, dist, rvecs, tvecs]

def reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    total_error = 0
    for i in range(len(objpoints)):
        # Tính toán lại tọa độ 3D của điểm góc trên bảng phân cách thành tọa độ ảnh
        reprojected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        # Tính toán khoảng cách giữa các điểm ảnh được tính toán lại và các điểm ảnh ban đầu
        error = cv2.norm(imgpoints[i], reprojected, cv2.NORM_L2) / len(reprojected)
        # Tổng hợp các giá trị bình phương khoảng cách
        total_error += error ** 2
    mean_error = np.sqrt(total_error / len(objpoints))
    print("Reprojection error: ", mean_error)
    return mean_error