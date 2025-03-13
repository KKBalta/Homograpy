Homography Project

This project implements various homography-based computer vision techniques, including feature detection, matching, homography computation, and applications such as image warping and augmented reality.

Project Structure

Homograpy/
├── data/            # Input data (images and videos)
├── figs/            # Image figures for documentation
├── python/          # Source code
│   ├── ar.py             # Augmented reality implementation
│   ├── briefRotTest.py   # Tests feature matching under rotation
│   ├── HarryPotterize.py # Replace book cover using homography
│   ├── helper.py         # Utility functions for feature detection
│   ├── loadSaveVid.py    # Video I/O functions
│   ├── matchPics.py      # Image feature matching functions
│   ├── panaroma.py       # Panorama stitching (stub)
│   ├── planarH.py        # Homography calculation implementations
│   ├── q3_4.py           # Feature matching example script
│   └── test_computeH.py  # Test for homography computation
├── requirements.txt      # Required Python packages
└── results/         # Output directory for results

Installation

To set up the environment, install the required dependencies:

pip install -r requirements.txt

Features and Implementations

1. Feature Detection and Matching

The project utilizes FAST corner detection and BRIEF descriptors for feature extraction and matching:

corner_detection detects corners using the FAST algorithm.

computeBrief computes BRIEF descriptors for features.

matchPics matches features between images.

2. Homography Computation

The project implements multiple homography computation methods with increasing robustness:

computeH: Direct Linear Transform (DLT) for computing homography.

computeH_norm: Normalized DLT for improved numerical stability.

computeH_ransac: RANSAC-based homography computation to handle outliers.

3. Image Warping and Compositing

compositeH applies a homography to overlay one image onto another.

Applications

Feature Matching Visualization

To run the feature matching demonstration:

cd python
python q3_4.py

This matches features between book cover and desk images and visualizes the matches.

Harry Potter Book Cover Replacement

Replace a book cover in an image with another using homography:

cd python
python HarryPotterize.py

Feature Matching under Rotation

Test the robustness of feature matching under rotation:

cd python
python briefRotTest.py

This rotates an image at different angles and plots the number of matches to assess descriptor robustness.

Augmented Reality

The ar.py script overlays video content onto a book cover in a video sequence, implementing an augmented reality effect.

cd python
python ar.py

Testing

Test the homography computation with:

cd python
python test_computeH.py

This verifies the correctness of the homography computation by mapping known points.

Dependencies

This project relies on several Python libraries:

OpenCV (cv2)

NumPy

SciPy

scikit-image

Matplotlib

All dependencies are listed in requirements.txt.

Future Work

Complete the panorama stitching functionality in panaroma.py.

Optimize the augmented reality implementation for real-time performance.

License

This project is open-source and free to use for educational and research purposes.

For any questions or contributions, feel free to reach out!

