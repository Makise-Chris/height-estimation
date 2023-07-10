import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrate(dirpathL, dirpathR, prefix, image_format, square_size, width=9, height=6):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgPointsL = []  # 2d points in image plane.
    imgPointsR = []

    if dirpathL[-1:] == '/':
        dirpathL = dirpathL[:-1]
    if dirpathR[-1:] == '/':
        dirpathR = dirpathR[:-1]

    imagesLeft = glob.glob(dirpathL+'/' + prefix + '*.' + image_format)
    imagesRight = glob.glob(dirpathR + '/' + prefix + '*.' + image_format)

    for imgLeft, imgRight in zip(imagesLeft, imagesRight):
        imgL = cv2.imread(imgLeft)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

        imgR = cv2.imread(imgRight)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        retL, cornersL =  cv2.findChessboardCorners(grayL, (width, height), None)
        retR, cornersR = cv2.findChessboardCorners(grayR, (width, height), None)

        # If found, add object points, image points (after refining them)
        if retL and retR:
            objpoints.append(objp)

            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            imgPointsL.append(cornersL)

            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            imgPointsR.append(cornersR)

            # Draw and display the corners
            cv2.drawChessboardCorners(imgL, (width, height), cornersL, retL)
            cv2.drawChessboardCorners(imgR, (width, height), cornersR, retR)
            cv2.imshow('img L', imgL)
            cv2.imshow('img R', imgR)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgPointsL, grayL.shape[::-1], None, None)
    heightL, widthL, channelsL = imgL.shape
    newCameraMtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (widthL, heightL), 1, (widthL, heightL))

    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgPointsR, grayR.shape[::-1], None, None)
    heightR, widthR, channelsR = imgR.shape
    newCameraMtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (widthR, heightR), 1, (widthR, heightR))

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    retStereo, newCameraMtxL, distL, newCameraMtxR, distR, rot, trans, essentialMtx, fundamentalMtx = cv2.stereoCalibrate(objpoints, imgPointsL, imgPointsR, mtxL, distL,
mtxR, distR, grayL.shape[::-1], criteria = criteria, flags = flags)

    rectL, rectR, projMtxL, projMtxR, Q, roiL, roiR = cv2.stereoRectify(newCameraMtxL, distL, newCameraMtxR, distR, grayL.shape[::-1], rot, trans)

    stereoMapL = cv2.initUndistortRectifyMap(newCameraMtxL, distL, rectL, projMtxL, grayL.shape[::-1], cv2.CV_16SC2)
    stereoMapR = cv2.initUndistortRectifyMap(newCameraMtxR, distR, rectR, projMtxR, grayR.shape[::-1], cv2.CV_16SC2)

    print("Saving parameters!")
    param_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_WRITE)
    param_file.write('stereoMapL_x', stereoMapL[0])
    param_file.write('stereoMapL_y', stereoMapL[1])
    param_file.write('stereoMapR_x', stereoMapR[0])
    param_file.write('stereoMapR_y', stereoMapR[1])
    param_file.write('cameraMtxL', projMtxL)
    print(newCameraMtxL)
    print(projMtxL)

calibrate('./test_images/stereoLeft', './test_images/stereoRight', 'image', 'png', 24, 9, 6)