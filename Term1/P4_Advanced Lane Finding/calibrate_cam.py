import numpy as np
import cv2
import glob

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in the real world space
imgpoints = [] # 2d points in image plane

def get_obj_img_points():
    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')
    
    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        
        # If found, add object pints, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
    
    return objpoints, imgpoints

# Calibrate Camera
def calibrate_camera():
    import pickle
    import os
    mtx = []
    dist = []
    
    if os.path.exists('mtx_dist_pickle.p'):
        f = pickle.load(open('mtx_dist_pickle.p', 'rb'))
        mtx = f['mtx']
        dist = f['dist']
    else:
        # Calibrate camera and obtain mtx and dist coefficients
        img = cv2.imread('./test_images/test1.jpg')
        img_size = (img.shape[1], img.shape[0])
        
        objpoints, imgpoints = get_obj_img_points()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        
        dist_pickle = {}
        dist_pickle['mtx'] = mtx
        dist_pickle['dist'] = dist
        pickle.dump(dist_pickle, open('mtx_dist_pickle.p', 'wb'))
    return mtx, dist