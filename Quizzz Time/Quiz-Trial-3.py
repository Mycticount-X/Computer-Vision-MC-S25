import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def Preprocessing (img, blurtype='gaussian', ksize=3):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)

    if blurtype == 'gaussian':
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif blurtype == 'median':
        img = cv2.medianBlur(img, ksize)
    elif blurtype == 'avg':
        img = cv2.blur(img, (ksize, ksize))
    elif blurtype == 'filter2d':
        kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
        img = cv2.filter2D(img, -1, kernel)
    elif blurtype == 'bilateral':
        img = cv2.bilateralFilter(img, ksize, 75, 75)
    else:
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    return img

def Identify (targ_path: str, dataset_path: str, algo='AKAZE', blurtype='gaussian', ksize=3):    
    img_targ_ori = cv2.imread(targ_path)
    if img_targ_ori is None:
        print('Failed to Load Image')
        return None
    
    img_targ = Preprocessing(img_targ_ori, blurtype, ksize)
    
    # Desc Algo
    if algo == 'AKAZE':
        descriptor = cv2.AKAZE.create()
    elif algo == 'ORB':
        descriptor = cv2.ORB.create()
    elif algo == 'SIFT' or algo == 'SURF':
        descriptor = cv2.SIFT.create()
    else:
        descriptor = cv2.AKAZE.create()

    # KP & Desc
    kp_targ, desc_targ = descriptor.detectAndCompute(img_targ, None)

    # Matcher
    if algo == 'ORB':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, False)
    else:
        desc_targ = np.float32(desc_targ)

        idx_params = dict(algorithm=1, trees=5)
        src_params = dict(checks=30)
        matcher = cv2.FlannBasedMatcher(idx_params, src_params)
    
    # Best Match
    best_match = 0
    best_data = {}

    for filename in os.listdir(dataset_path):
        if not filename.endswith(('.jpg', '.png', '.jpeg')):
            continue

        ref_path = os.path.join(dataset_path, filename)
        img_ref_ori = cv2.imread(ref_path)
        img_ref = Preprocessing(img_ref_ori, blurtype, ksize)

        # KP & Desc
        kp_ref, desc_ref = descriptor.detectAndCompute(img_ref, None)

        if desc_ref is None or len(desc_ref) < 2:
            continue

        if algo != 'ORB':
            desc_ref = np.float32(desc_ref)

        # Lowe's Ratio
        matches = matcher.knnMatch(desc_targ, desc_ref, k=2)
        good_matches = []

        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Best Match
        if len(good_matches) > best_match:
            best_match = len(good_matches)
            best_data = {
                'name': filename,
                'image': img_ref_ori,
                'matches': good_matches,
                'keypoints': kp_ref
            }

    # Visualize
    if best_match > 0:
        print('PoVeKamon Found')
        print()

        result = cv2.drawMatches(
            img_targ_ori, kp_targ,
            best_data['image'], best_data['keypoints'],
            best_data['matches'], None,
            (255, 0, 255), (0, 0, 255)
        )

        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 5))
        plt.imshow(result)
        plt.title(f'Target vs {best_data["name"]}  -  {len(best_data["matches"])} match(s)')
        plt.axis(False)
        plt.show()
    else:
        print('(!) The Image not match with any PoVeKamon in Database!')



if __name__ == '__main__':
    print('Version 1')
    Identify('./Images-Pokemon/Object.png', './Images-Pokemon/Data', 'AKAZE', 'median', 5)
    Identify('./Images-Pokemon/Object2.png', './Images-Pokemon/Data', 'AKAZE', 'median', 5)
    print('')

    print('Version 2')
    Identify('./Images-Pokemon/Object.png', './Images-Pokemon/Data', 'AKAZE', 'gaussian', 3)
    Identify('./Images-Pokemon/Object2.png', './Images-Pokemon/Data', 'AKAZE', 'gaussian', 3)