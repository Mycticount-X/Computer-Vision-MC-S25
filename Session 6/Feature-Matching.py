import cv2
import numpy as np
import matplotlib.pyplot as plt

def feature_matching(img, scene):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    # Create Surf
    surf = cv2.xfeatures2d.SIFT_create()
    # surf.setHessianThreshold(15000)

    # Keypoint & Descriptor
    keyp_img, desc_img = surf.detectAndCompute(img, None)
    keyp_scene, desc_scene = surf.detectAndCompute(scene, None)

    # Create Flann
    idx_param = dict(algorithm=0)
    src_param = dict(checks=50)
    flann = cv2.FlannBasedMatcher(idx_param, src_param)

    # Match
    matched = flann.knnMatch(desc_img, desc_scene, k=2)

    valid = [[0,0] for _ in range(len(matched))]
    for i, (FB, SB) in enumerate(matched):
        if (FB.distance < 0.7 * SB.distance):
            valid[i] = [1,0]

    # Draw
    result = cv2.drawMatchesKnn(
        img, 
        keyp_img,
        scene,
        keyp_scene,
        matched,
        None,
        matchesMask=valid
    )

    plt.imshow(result)
    plt.show()



if __name__ == "__main__":
    img = cv2.imread('./kitkat.png')
    scene = cv2.imread('./kitkat_scene.jpg')
    feature_matching(img, scene)
