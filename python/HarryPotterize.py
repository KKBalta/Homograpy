import numpy as np
import cv2
import skimage.io
import skimage.color
from matchPics import matchPics
from helper import plotMatches
from planarH import computeH_ransac, compositeH

DISPLAY = True

if __name__ == "__main__":
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')

    # Get matches and corresponding locations
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk)
    print(f"Found {len(matches)} matches between cover and desk")

    # Display matches for debugging
    if DISPLAY:
        plotMatches(cv_cover, cv_desk, matches, locs1, locs2, showImg=True)

    # Create set of points (x1, x2) corresponding to various matches
    # Swap from (y,x) to (x,y) format
    x1 = locs1[matches[:, 0]][:, [1, 0]]  # cv_cover points
    x2 = locs2[matches[:, 1]][:, [1, 0]]  # cv_desk points

    # Find H and inliners using ransac
    H, inliers = computeH_ransac(x1, x2)
    print(f"Number of inliers: {np.sum(inliers)} out of {len(matches)} matches")
    print("Computed Homography:")
    print(H)

    # Normalize H
    H = H / H[2, 2]

    # Resize hp cover to that of the cv_cover shape before transforming
    resizeShape = (cv_cover.shape[1], cv_cover.shape[0])
    hp_cover = cv2.resize(hp_cover, dsize=resizeShape)

    # Get composite image
    compositeImg = compositeH(H, hp_cover, cv_desk)

    cv2.imshow("Composite Image", compositeImg)
    cv2.waitKey(0)

    
    if DISPLAY:
        # For cross-checking only; cv2 has its inputs swapped.
        H_ground_truth, inliners = cv2.findHomography(x2, x1, method=cv2.RANSAC)
        print(f"inlier count: {np.sum(inliers)}")
        print("OpenCV Homography:")
        print(H_ground_truth)

        compositeImgtrue = compositeH(H_ground_truth, hp_cover, cv_desk)
        cv2.imshow(" Ground Truth CompositeH", compositeImgtrue)
        cv2.waitKey(0)


        imgShape = (cv_desk.shape[1], cv_desk.shape[0])
        warpedImg = cv2.warpPerspective(hp_cover, np.linalg.inv(H), dsize=imgShape)
        cv2.imshow("My Output", warpedImg)
        cv2.waitKey(0)

        warpedImg = cv2.warpPerspective(hp_cover, np.linalg.inv(H_ground_truth), dsize=imgShape)
        cv2.imshow("Ground Truth (cv2)", warpedImg)
        cv2.waitKey(0)
