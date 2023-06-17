import numpy as np
from util import *

class ForegroundExtraction:
    def __init__(self, source, foregroundSeed, backgroundSeed, threshold):
        # Keep a copy of source
        self.source = source.copy()

        # This is where the output is stored
        self.cutOut = source.copy()

        # Store the user marked foreground and background seeds
        self.foregroundSeed = foregroundSeed
        self.backgroundSeed = backgroundSeed

        # Used to create an image pyramid
        self.iDown = [self.source]
        self.fDown = [self.foregroundSeed]
        self.bDown = [self.backgroundSeed]

        # Keep a list of contour points for items being cut out
        self.cuts = None

        # Number of objects being cut out
        self.numCuts = None

        # Refined foreground and background
        self.foreground = None
        self.background = None

        # Images stored for debugging purposes
        self._contour = None
        self._contourSource = None
        self._initialMask = None
        self._refinedMask = np.zeros(self.source.shape[:2], dtype=np.uint8)
        self._boundary = None

        self.numRows = source.shape[0]
        self.numCols = source.shape[1]
        self.minImageSize = 500000
        self.timesDownsampled = 0

        # Kernel used for convolution
        self.kernel = np.ones((3,3), np.uint8)

        # Hyper parameters to tweak
        self.erodeIterations = 8
        self.dilateIterations = 8
        self.patchRadius = 10
        self.colorWeight = 0.5
        self.locationWeight = 0.5
        self.contourSizeThreshold = threshold

    def run(self):
        # We down sample the large image to a much more managable size
        # and we perform Grabcut on the downsampled image to get a rough
        # esimate of the cutout mask. Using this estimate, we can obtain a more
        # refined foreground and background seed.
        self.downSample()
        self.cuts, self.foreground, self.background, self._contour , self._contourSource, self._boundary = self.getBoundaries()

        # Using the refined foreground and background seeds, we take the estimated
        # cutout mask and refine the contour by examining patches that lie on the contour
        # and cut out pixels with respect to the refined foreground and background seeds
        self.refineBoundary(self._boundary[2])

        # Cut out the original image using the refined mask
        self.cutOut[self._refinedMask==0] = 0
        return self.cutOut, self._refinedMask, self._contour, self._contourSource, self._initialMask

    # Constructs an image pyramid by scaling down the image and foreground and background seeds
    def downSample(self):
        imageSize = self.numRows * self.numCols
        while imageSize > self.minImageSize:
            image = cv.pyrDown(self.iDown[-1])
            self.iDown.append(image)

            foreground = cv.pyrDown(self.fDown[-1])
            self.fDown.append(foreground)

            background = cv.pyrDown(self.bDown[-1])
            self.bDown.append(background)

            imageSize = image.shape[0] * image.shape[1]
            self.timesDownsampled += 1

    # Get a rough estimate of the cutout mask
    def getBoundaries(self):

        # Perform the cut
        mask = self.cut(self.iDown[-1], self.fDown[-1], self.bDown[-1])

        # Upsample the mask back to the size of the original image
        for i in range(self.timesDownsampled):
            if mask.shape[0] != self.iDown[-1 * (i + 1)].shape[0]:
                mask = mask[:-1,:]
            if mask.shape[1] != self.iDown[-1 * (i + 1)].shape[1]:
                mask = mask[:,:-1]
            mask = cv.pyrUp(mask)

        # Store a copy of the upsampled unrefined mask
        self._initialMask = mask.copy()

        # We know pixels inside the contour of the unrefined mask
        # should be a part of the foreground and that pixels outside of the
        # contour should be a part of the background so we can generate a better
        # foreground and background seed by eroded and dilating the unrefined mask
        mask = cv.erode(mask, self.kernel, iterations=self.erodeIterations)
        erodedMask = cv.erode(mask, self.kernel, iterations=self.erodeIterations)
        dilatedMask = cv.dilate(mask, self.kernel, iterations=self.dilateIterations)

        boundary = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        erodedBoundary = cv.findContours(erodedMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        dilatedBoundary = cv.findContours(dilatedMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # Create image representations of the new foreground and background seeds
        contour = np.ones(self.source.shape, dtype=np.uint8) * 255
        contourSource = self.source.copy()
        foreground = np.zeros(self.source.shape[:2], dtype=np.uint8)
        background = np.zeros(self.source.shape[:2], dtype=np.uint8)

        cuts = []
        self.numCuts = min((len(boundary[0]), len(erodedBoundary[0]), len(dilatedBoundary[0])))
        # Only deal with the largest contours to avoid artifacts
        biggestContours = np.array([cv.contourArea(x) for x in boundary[0]]).argsort()[-self.numCuts:]
        biggestErodedContours = np.array([cv.contourArea(x) for x in erodedBoundary[0]]).argsort()[-self.numCuts:]
        biggestDilatedContours = np.array([cv.contourArea(x) for x in dilatedBoundary[0]]).argsort()[-self.numCuts:]

        for i in range(0, self.numCuts):
            points = boundary[0][biggestContours[i]]
            erodedPoints = erodedBoundary[0][biggestErodedContours[i]]
            dilatedPoints = dilatedBoundary[0][biggestDilatedContours[i]]

            if (cv.contourArea(points) > self.contourSizeThreshold):
                # Red
                for p in points:
                    contour[p[0][1], p[0][0]] = [0, 0, 255]
                    contourSource[p[0][1], p[0][0]] = [0, 0, 255]

                # Green
                for p in erodedPoints:
                    foreground[p[0][1], p[0][0]] = 255
                    contour[p[0][1], p[0][0]] = [0, 255, 0]
                    contourSource[p[0][1], p[0][0]] = [0, 255, 0]

                # Blue
                for p in dilatedPoints:
                    background[p[0][1], p[0][0]] = 255
                    contour[p[0][1], p[0][0]] = [255, 0, 0]
                    contourSource[p[0][1], p[0][0]] = [255, 0, 0]

                cuts.append((points, erodedPoints, dilatedPoints))
        return cuts, foreground, background, contour, contourSource, ([boundary, erodedBoundary, dilatedBoundary])

    # Refines the contour by examining patches on the contour
    def refineBoundary(self, boundary):
        for i in range(len(boundary[0])):
            if cv.contourArea(boundary[0][i]) > self.contourSizeThreshold:
                cv.drawContours(self._refinedMask, boundary[0], i, 255, thickness=-1)

    # Performs Grabcut on an image given a foreground and background seed
    def cut(self, image, foreground, background):
        mask = np.ones(image.shape[:2], dtype=np.uint8) * cv.GC_PR_BGD
        mask[background!=255] = cv.GC_BGD
        mask[foreground!=255] = cv.GC_FGD

        bgdModel = np.zeros((1, 65), dtype=np.float64)
        fgdModel = np.zeros((1, 65), dtype=np.float64)

        cv.grabCut(image,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        mask *= 255
        return mask


# Read images
source = readSource('./test_images/test1/source.png')
foregroundSeed = readMask('./test_images/test1/foreground.png')
backgroundSeed = readMask('./test_images/test1/background.png')
threshold = 20000

assert source is not None
assert foregroundSeed is not None
assert backgroundSeed is not None

# Run algorithm
extraction = ForegroundExtraction(source, foregroundSeed, backgroundSeed, threshold)
result, _refinedMask, _contour, _contourSource, _initialMask = extraction.run()

# Figure out the paths for the debugging images
output = 'test_images/test6/out.png'
splitString = output.split('.')
_refinedMaskPath = str(splitString[0]) + '_refinedMask.'
_contourPath = str(splitString[0]) + '_contour.'
_contourSourcePath = str(splitString[0]) + '_contourSource.'
_initialMaskPath = str(splitString[0]) + '_initialMask.'

for i in range(1, len(splitString)):
    _refinedMaskPath += str(splitString[i])
    _contourPath += str(splitString[i])
    _contourSourcePath += str(splitString[i])
    _initialMaskPath += str(splitString[i])

# Write the images
writeImage(output, result)
writeImage(_refinedMaskPath, _refinedMask)
writeImage(_contourPath, _contour)
writeImage(_contourSourcePath, _contourSource)
writeImage(_initialMaskPath, _initialMask)