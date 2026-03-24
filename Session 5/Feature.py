import cv2
import numpy as np
import matplotlib.pyplot as plt

catur = cv2.imread('./chessboard.jpg')

if (catur is None):
    print('Image not found')
    exit()

catur = cv2.resize(catur, (0, 0), fx=0.5, fy=0.5)
gray = cv2.cvtColor(catur, cv2.COLOR_BGR2GRAY) # 
gray = np.float32(gray)

xtra1 = cv2.imread('./images/1.jpg')
xtra2 = cv2.imread('./images/2.png')
xtra3 = cv2.imread('./images/3.jpg')

if (xtra1 is None or xtra2 is None or xtra3 is None):
    print('One or more images not found')
    exit()

gray1 = cv2.cvtColor(xtra1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(xtra2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(xtra3, cv2.COLOR_BGR2GRAY)

# Show
def ShowImagePLT(image, title='Image'):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def ShowImageCV2(image, title='Image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Problem
def HarrisCorner(image, tres=0.01):
    harris = cv2.cornerHarris(image, 2, 5, 0.04)
    res = catur.copy()
    res[harris > tres * harris.max()] = [0, 0, 255]
    return res

def SubpixelCorner(image):
    harris = cv2.cornerHarris(image, 2, 5, 0.04)
    _, tres = cv2.threshold(harris, 0.01 * harris.max(), 255, cv2.THRESH_BINARY)
    tres = np.uint8(tres)

    _, _, _, centroids = cv2.connectedComponentsWithStats(tres)
    centroids = np.float32(centroids)

    criteria = (
        cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 
        100, 
        0.001
    )

    enhanced_corners = cv2.cornerSubPix(image, centroids, (2, 2), (-1, -1), criteria)

    res = catur.copy()
    enhanced_corners = np.int16(enhanced_corners)

    for corner in enhanced_corners:
        corner_y = corner[1]
        corner_x = corner[0]
        res[corner_y, corner_x] = [0, 255, 0]
    
    return res

def FastFeatureDetector(image_gray, ori_image):
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(image_gray, None)
    res = ori_image.copy()
    cv2.drawKeypoints(ori_image, keypoints, res, color=[0, 0, 255])
    return res

def ORBFeatureDetector(image_gray, ori_image):
    orb = cv2.ORB_create()
    keypoints = orb.detect(image_gray, None)
    res = ori_image.copy()
    cv2.drawKeypoints(ori_image, keypoints, res, color=[255, 0, 0])
    return res


harris = HarrisCorner(gray)
subpix = SubpixelCorner(gray)

fast_result = FastFeatureDetector(gray1, xtra1)
orb_result = ORBFeatureDetector(gray1, xtra1)

# ShowImageCV2(harris, 'Harris')
# ShowImageCV2(subpix, 'Subpixel')
# ShowImageCV2(fast_result, 'FAST Result')
# ShowImageCV2(orb_result, 'ORB Result')

res_harris = [
    HarrisCorner(gray1),
    HarrisCorner(gray2),
    HarrisCorner(gray3)
]

res_subpix = [
    SubpixelCorner(gray1),
    SubpixelCorner(gray2),
    SubpixelCorner(gray3)
]

res_fast = [
    FastFeatureDetector(gray1, xtra1),
    FastFeatureDetector(gray2, xtra2),
    FastFeatureDetector(gray3, xtra3)
]

res_orb = [
    ORBFeatureDetector(gray1, xtra1),
    ORBFeatureDetector(gray2, xtra2),
    ORBFeatureDetector(gray3, xtra3)
]

plt.figure(figsize=(12, 8))
