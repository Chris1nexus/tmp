
import cv2
import numpy as np
import glob
import os
def calibrate(dirpath):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    CHESS_SQUARE_LEN = 0.027
 
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*CHESS_SQUARE_LEN
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(os.path.join(dirpath,"*.jpg"))
    gray = None
    for fname in images:
        print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (9,6), corners2, ret)
            #cv2.imshow('img', img)
            #cv2.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]

def saveCoefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", mtx)
    cv_file.write("dist_coeff", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()



if __name__ == '__main__':
    ret, matrix_coefficients, distortion_coefficients, rvecs, tvecs = calibrate('calib_images')   
    if not os.path.exists( "./params"):
        os.makedirs( "./params")
    print('rvecs')
    for item in rvecs:
        print(item)
        print()

    print('tvecs')
    for item in tvecs:
        print(item)
        print()
    saveCoefficients(matrix_coefficients, distortion_coefficients, "./params/calibration_coefficients.yaml")


    #h,  w = gray.shape[:2]
    #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix_coefficients, distortion_coefficients, (w,h), 1, (w,h))
    # undistort
    #dst = cv2.undistort(gray, matrix_coefficients, distortion_coefficients, None, newcameramtx)

