import cv2 as cv
import numpy as np 

def Binarization(image, thresholdValue):

    base_path = "./images"

    img = cv.imread(f"{base_path}/{image}", 0)
    cv.imshow("Original Image", img)

    gray_hist = cv.calcHist([img], [0], None, [256], [0, 256])

    blank = np.zeros(img.shape[:2], dtype='uint8')

    #! Simple Thresholding
    threshold, simple_thresh = cv.threshold(img, thresholdValue, 255, cv.THRESH_BINARY)
    cv.imshow('Simple Thresholded', simple_thresh)

    #! Simple Thresholding
    threshold, otsu_thresh = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    cv.imshow('Otsu Thresholded', otsu_thresh)

    #! Adaptive Thresholding
    adaptive_thresh_gaussian = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 29, 9)
    cv.imshow('Adaptive Thresholded With Gaussian', adaptive_thresh_gaussian)

    adaptive_thresh_mean_c = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 3)
    cv.imshow('Adaptive Thresholded With Mean', adaptive_thresh_mean_c)

    # equ = cv.equalizeHist(img)
    # # stacking images side-by-side 
    # res = np.hstack((img, equ)) 
    # cv.imshow("Result", res) 
    
    returnedObject = {
        "Simple Thresholded Image" : simple_thresh,
        "Otsu Thresholded Image" : otsu_thresh,
        "Adaptive Thresholded With Gaussian" : adaptive_thresh_gaussian,
        "Adaptive Thresholded With Mean" : adaptive_thresh_mean_c,
    }

    return returnedObject