import os
import sys
import argparse
from modules.histogram import Histogram
from modules.binarization import Binarization
from modules.morphologic import (
    Dilation,
    Erosion,
    Opening,
    Closing
)
from modules.hist_equalization import HistogramEqualization

import cv2 as cv

parser = argparse.ArgumentParser(description="Image Process on Medical Images")

parser.add_argument('--image', type = str, default = 'ultrason-fetus.tif', help = 'Image Name')
parser.add_argument('--histogram', action = 'store_true', help = 'Calc Histogram')
parser.add_argument('--binarization', type = int, choices=list(range(1, 256)), required=False, help = 'Make Binarization')
parser.add_argument('--morphologic', choices=["dilation", "erosion", "opening", "closing"], type = str, default = None, help = 'Make Morphologic Operation')
parser.add_argument('--equalization', action = 'store_true', help = 'Make Histogram Equalization')

args = parser.parse_args()

if __name__ == "__main__":
    if args.histogram:
        Histogram(args.image)
    if args.equalization:
        equalizedImage = HistogramEqualization(args.image)
    if args.binarization:
        binaryImg = Binarization(args.image, args.binarization)
        if args.morphologic:
            if args.morphologic == 'dilation':
                cv.imshow("Simple Dilated", Dilation(binaryImg["Simple Thresholded Image"]))
                cv.imshow("Otsu Dilated", Dilation(binaryImg["Otsu Thresholded Image"]))
                cv.imshow("Adaptive With Gaussian Dilated", Dilation(binaryImg["Adaptive Thresholded With Gaussian"]))
                cv.imshow("Adaptive With Mean Dilated", Dilation(binaryImg["Adaptive Thresholded With Mean"]))
            elif args.morphologic == 'erosion':
                cv.imshow("Simple Eroded", Erosion(binaryImg["Simple Thresholded Image"]))
                cv.imshow("Otsu Eroded", Erosion(binaryImg["Otsu Thresholded Image"]))
                cv.imshow("Adaptive With Gaussian Eroded", Erosion(binaryImg["Adaptive Thresholded With Gaussian"]))
                cv.imshow("Adaptive With Mean Eroded", Erosion(binaryImg["Adaptive Thresholded With Mean"]))
            elif args.morphologic == 'opening':
                cv.imshow("Simple Opened", Opening(binaryImg["Simple Thresholded Image"]))
                cv.imshow("Otsu Opened", Opening(binaryImg["Otsu Thresholded Image"]))
                cv.imshow("Adaptive With Gaussian Opened", Opening(binaryImg["Adaptive Thresholded With Gaussian"]))
                cv.imshow("Adaptive With Mean Opened", Opening(binaryImg["Adaptive Thresholded With Mean"]))
            elif args.morphologic == 'closing':
                cv.imshow("Simple Closed", Closing(binaryImg["Simple Thresholded Image"]))
                cv.imshow("Otsu Closed", Closing(binaryImg["Otsu Thresholded Image"]))
                cv.imshow("Adaptive With Gaussian Closed", Closing(binaryImg["Adaptive Thresholded With Gaussian"]))
                cv.imshow("Adaptive With Mean Closed", Closing(binaryImg["Adaptive Thresholded With Mean"]))

    cv.waitKey(0)
    cv.destroyAllWindows()
