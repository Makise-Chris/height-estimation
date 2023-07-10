import cv2

def rectify(imgL, imgR):
    param_file = cv2.FileStorage()
    param_file.open('stereoMap.xml', cv2.FileStorage_READ)

    stereoMapL_x = param_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = param_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = param_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = param_file.getNode('stereoMapR_y').mat()

    imgL = cv2.remap(imgL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4)
    imgR = cv2.remap(imgR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4)

    cv2.imshow("Img Left", imgL)
    cv2.imshow("Img Right", imgR)
    cv2.waitKey(0)

    # capLeft=cv2.VideoCapture(f"http://192.168.0.101:4747/video")
    # capRight=cv2.VideoCapture(f"http://192.168.0.102:4747/video")
    #
    # while (capLeft.isOpened() and capRight.isOpened()):
    #     retL, imgL = capLeft.read()
    #     retR, imgR = capRight.read()
    #
    #     if retL and retR:
    #         imgL = cv2.remap(imgL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4)
    #         imgR = cv2.remap(imgR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4)
    #
    #         cv2.imshow("Img Left", imgL)
    #         cv2.imshow("Img Right", imgR)
    #
    #         k = cv2.waitKey(1)
    #
    #         if k == ord('q'):
    #             break
    #
    # capLeft.release()
    # capRight.release()
    cv2.destroyAllWindows()

    return imgL, imgR