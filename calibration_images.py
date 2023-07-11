import cv2

cap=cv2.VideoCapture(f"http://192.168.97.245:4747/video")
cap2=cv2.VideoCapture(f"http://192.168.97.243:4747/video")
num=0

while cap.isOpened():
    ret, img=cap.read()
    ret2, img2=cap2.read()

    if ret == True:
        k=cv2.waitKey(5)

        if k==ord('q'):
            break
        elif k==ord('s'):
            cv2.imwrite('./test_images/stereoLeft/imageL' + str(num) + '.png', img)
            cv2.imwrite('./test_images/stereoRight/imageR' + str(num) + '.png', img2)
            print('Images saved')
            num+=1
        elif k==ord('o'):
            cv2.imwrite('./test_images/test1/source' + str(num) + '.png', img)
            cv2.imwrite('./test_images/test1/source' + str(num+1) + '.png', img2)
            print('Images saved')
            num+=1
        cv2.imshow('Img left', img)
        cv2.imshow('Img Right', img2)

cap.release()
cap2.release()
cv2.destroyAllWindows()
