import os
import scipy
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matchPics import matchPics

from helper import plotMatches

resultsdir = "../results/rotTest"

"""
Q3.5
"""
if __name__ == "__main__":
	os.makedirs(resultsdir, exist_ok=True)

	# Read the image and convert to grayscale, if necessary
	originalImg = cv2.imread("../data/cv_cover.jpg")
	rotImg = originalImg.copy()

	# Histogram count for matches
	nMatches = []
	angles = [ (i+1)*10 for i in range(36) ]

	for i in range(36):

		angle = (i+1)*10
		print(f"Rotating image by {angle} degrees")
		# Rotate Image
		rotImg = scipy.ndimage.rotate(originalImg,angle,reshape=False)

		# Compute features, descriptors and Match features
		matches, locs1, locs2 = matchPics(originalImg, rotImg)

		# Update histogram
		nMatches.append(len(matches)) 
		print(f"Number of matches: {len(matches)} at angle {angle}")

		# Save all results
		saveTo = os.path.join(resultsdir, f"rot{angles}.png")
		plotMatches(originalImg, rotImg, matches, locs1, locs2, saveTo=saveTo, showImg=True)
		print(f"Saved to {saveTo}")


	# Display histogram
	plt.tight_layout()
	plt.bar(x=angles, height=nMatches, width=5)
	plt.show()
