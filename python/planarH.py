import numpy as np
import cv2

"""
Q3.6
Compute the homography between two sets of points

@param[in] x1 Set of points x1 in hetereogeneous coords
@param[in] x2 Set of points x2 in hetereogeneous coords

@return H2to1 Closest 3x3 H matrix (least squares)
"""
def computeH(x1, x2):

    # Check if the number of points is the same
    assert(x1.shape[1] == 2 and x2.shape[1] == 2)

    n = x1.shape[0]
    MatrixA = np.zeros((2*n, 9))
    # Create A_i for each point correspondence    
    for i in range(n):
        x1i = x1[i][0]
        y1i = x1[i][1]
        x2i = x2[i][0]
        y2i = x2[i][1]

        # For x1i = h11*x2i + h12*y2i + h13
        #     y1i = h21*x2i + h22*y2i + h23
        # Rearranging to form Ah = 0:
        # x2i*h11 + y2i*h12 + h13 - x1i*x2i*h31 - x1i*y2i*h32 - x1i*h33 = 0
        # x2i*h21 + y2i*h22 + h23 - y1i*x2i*h31 - y1i*y2i*h32 - y1i*h33 = 0
        
        MatrixA[2*i] = np.array([x2i, y2i, 1, 0, 0, 0, -x1i*x2i, -x1i*y2i, -x1i])
        MatrixA[2*i+1] = np.array([0, 0, 0, x2i, y2i, 1, -y1i*x2i, -y1i*y2i, -y1i])

    # Perform SVD
    U, S, Vt = np.linalg.svd(MatrixA)

    # Solution is the last column of V
    H2to1 = Vt[-1]

    # Reshape to 3x3
    H2to1 = H2to1.reshape(3,3)

    H2to1 = H2to1 / H2to1[2,2] # Normalize

    # Round the elements to 6 decimal places
    H2to1 = np.round(H2to1, 6)
    
    return H2to1


"""
Q3.7
Normalise the coordinates to reduce noise before computing H

@param[in] x1 Set of points x1 in hetereogeneous coords
@param[in] x2 Set of points x2 in hetereogeneous coords

@return H2to1 Closest 3x3 H matrix (least squares)
"""
def computeH_norm(_x1, _x2):
    x1 = np.array(_x1)
    x2 = np.array(_x2)

    # Compute the centroid (mean) of the points
    mean1 = np.mean(x1, axis=0)
    mean2 = np.mean(x2, axis=0)

    # Shift the origin of the points to the centroid
    x1norm = x1 - mean1
    x2norm = x2 - mean2

    dist1 = np.sqrt(np.sum(x1norm**2, axis=1))
    dist2 = np.sqrt(np.sum(x2norm**2, axis=1))

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    max_dist1 = np.max(dist1)
    max_dist2 = np.max(dist2)

    norm1 = max_dist1 / np.sqrt(2)
    norm2 = max_dist2 / np.sqrt(2)

    x1norm /= norm1
    x2norm /= norm2

    # Similarity transform for x1
    T1 = np.array([
        [1/norm1, 0, -mean1[0]/norm1],
        [0, 1/norm1, -mean1[1]/norm1],
        [0, 0, 1]
    ])
    
    # Inverse of T1 (for denormalization)
    T1_inv = np.array([
        [norm1, 0, mean1[0]],
        [0, norm1, mean1[1]],
        [0, 0, 1]
    ])

    # Similarity transform for x2 - THIS WAS THE ISSUE
    # Now using norm2 and mean2 instead of norm1 and mean1
    T2 = np.array([
        [1/norm2, 0, -mean2[0]/norm2],
        [0, 1/norm2, -mean2[1]/norm2],
        [0, 0, 1]
    ])
    
    # Inverse of T2 (not needed but included for completeness)
    T2_inv = np.array([
        [norm2, 0, mean2[0]],
        [0, norm2, mean2[1]],
        [0, 0, 1]
    ])

    # Compute homography with normalized coordinates
    H2to1_norm = computeH(x1norm, x2norm)

    # Denormalization: H2to1 = T1_inv * H2to1_norm * T2
    H2to1 = T1_inv @ H2to1_norm @ T2

    # Normalize
    H2to1 = H2to1 / H2to1[2, 2]
    
    return H2to1


"""
Q3.8
Run RANSAC on set of matched points x1, x2.
Reduces effect of outliers by finding inliers.
Returns best fitting homography H and best inlier set.

@param[in] x1 Set of points x1 in hetereogeneous coords
@param[in] x2 Set of points x2 in hetereogeneous coords
@param[in] threshold
    # TODO: Find out what the threshold is
    # Note that threshold is squared sum of difference
    # to avoid extra sqrt computation, so threshold
    # will be number of pixels away, SQUARED
    threshold = 10  # ~3 pixels away

@return bestH2to1
@return bestInlier Vector of length N with a 1 at matches, 0 elsewhere
"""
def computeH_ransac(_x1, _x2, nSamples=1000, threshold=10):
    
    x1 = np.array(_x1)
    x2 = np.array(_x2)

    nPoints = len(x1)
    assert(nPoints == len(x2))
    
    bestH2to1 = None
    best_inlier_count = 0
    best_inliers = np.zeros(nPoints, dtype=int)

    # Convert to homogeneous coordinates for calculations
    x1_homog = np.hstack((x1, np.ones((nPoints, 1))))
    x2_homog = np.hstack((x2, np.ones((nPoints, 1))))

    # Sample N times
    for i in range(nSamples):
        # Choose 4 random unique points
        indexes = np.random.choice(np.arange(nPoints), size=4, replace=False)
        
        try:
            # Compute homography from the minimal sample
            h = computeH_norm(x1[indexes], x2[indexes])  # Using normalized version for better results
            
            # Apply homography to all points in x2
            x1_est = np.dot(h, x2_homog.T).T
            
            # Skip if any invalid values
            if np.any(np.isnan(x1_est)) or np.any(np.isinf(x1_est)):
                continue
                
            # Convert back from homogeneous coordinates
            third_comp = x1_est[:, 2:3]
            if np.any(np.abs(third_comp) < 1e-10):
                continue
                
            x1_est = x1_est[:, :2] / third_comp
            
            # Calculate the error for each point (squared Euclidean distance)
            errors = np.sum((x1[:, :2] - x1_est) ** 2, axis=1)
            
            # Count inliers (points with error less than threshold)
            current_inliers = errors < threshold
            inlier_count = np.sum(current_inliers)
            
            # Update best model if we found more inliers
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                bestH2to1 = h
                best_inliers = current_inliers.astype(int)
                
                # Early termination if we have a very good model
                if best_inlier_count > 0.8 * nPoints:
                    break        
        except Exception:
            continue
    
    # Recompute homography using all inliers for better accuracy
    if best_inlier_count >= 4:
        inlier_indices = np.where(best_inliers)[0]
        bestH2to1 = computeH_norm(x1[inlier_indices], x2[inlier_indices])
    
    return bestH2to1, best_inliers

"""
Q3.9
Create a composite image after warping the template image on top
of the image using the homography

Note that the homography we compute is from the image to the template;
x_template = H2to1*x_photo
"""

def compositeH(H2to1, template, img, alreadyInverted=False):
    imgShape = (img.shape[1], img.shape[0])  # (width, height)
    
    # Invert Homography if not already inverted
    if not alreadyInverted:
        H1to2 = np.linalg.inv(H2to1)
    else:
        H1to2 = H2to1
    
    # Create a binary mask (single-channel) the same size as the template
    mask = np.ones((template.shape[0], template.shape[1]), dtype=np.uint8) * 255
    
    # Warp the mask to determine where the template should go in the image
    warped_mask = cv2.warpPerspective(mask, H1to2, imgShape)
    
    # Convert to binary mask
    _, warped_mask_bin = cv2.threshold(warped_mask, 1, 255, cv2.THRESH_BINARY)
    
    # Warp template using the homography
    templateWarped = cv2.warpPerspective(template, H1to2, imgShape)
    
    # Create an inverse mask (black where the template goes)
    inv_warped_mask = cv2.bitwise_not(warped_mask_bin)
    
    # Ensure the image and masks have the correct channels
    if len(img.shape) == 3 and img.shape[2] == 3:
        inv_warped_mask = cv2.cvtColor(inv_warped_mask, cv2.COLOR_GRAY2BGR)
        warped_mask_bin = cv2.cvtColor(warped_mask_bin, cv2.COLOR_GRAY2BGR)
    
    # Remove padding from the image using the inverse mask
    img_no_pad = cv2.bitwise_and(img, inv_warped_mask)
    
    # Merge the warped template onto the image
    composite_img = cv2.add(img_no_pad, templateWarped)

    return composite_img