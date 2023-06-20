import cv2
import numpy as np

source = cv2.imread('./test_images/test1/source2.png')
result_b = cv2.equalizeHist(source[:,:,0])
result_g = cv2.equalizeHist(source[:,:,1])
result_r = cv2.equalizeHist(source[:,:,2])

source = np.stack((result_b,result_g,result_r), axis=2)

# Define boundary rectangle containing the foreground object
height, width, _ = source.shape
left_margin_proportion = 0.3
right_margin_proportion = 0.3
up_margin_proportion = 0.1
down_margin_proportion = 0.1

boundary_rectangle = (
    int(width * left_margin_proportion),
    int(height * up_margin_proportion),
    int(width * (1 - right_margin_proportion)),
    int(height * (1 - down_margin_proportion)),
)

# Set the seed for reproducibility purposes
cv2.setRNGSeed(0)

# Initialize GrabCut mask image, that will store the segmentation results
mask = np.zeros((height, width), np.uint8)
mask[:] = cv2.GC_PR_BGD

gray_image = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
gray_image = cv2.equalizeHist(gray_image)

binarized_image = cv2.adaptiveThreshold(
    gray_image,
    maxValue=1,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY,
    blockSize=9,
    C=7,
)
mask[binarized_image == 0] = cv2.GC_FGD
mask = cv2.medianBlur(mask, 3)

# Arrays used by the algorithm internally
background_model = np.zeros((1, 65), np.float64)
foreground_model = np.zeros((1, 65), np.float64)

number_of_iterations = 5

cv2.grabCut(
    img=source,
    mask=mask,
    rect=boundary_rectangle,
    bgdModel=background_model,
    fgdModel=foreground_model,
    iterCount=number_of_iterations,
    mode=cv2.GC_INIT_WITH_RECT,
)

mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype("uint8")
segmented_image = source.copy() * mask[:, :, np.newaxis]
mask *= 255

cv2.imshow('image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()