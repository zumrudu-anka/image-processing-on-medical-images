import cv2 as cv
# import numpy as np
import matplotlib.pyplot as plt

def HistogramEqualization(image):

    img = cv.imread(f"./images/{image}", 0)
    cv.imshow("Original Image", img)

    # creating a Histograms Equalization 
    # of a image using cv2.equalizeHist()
    equ = cv.equalizeHist(img)
    cv.imshow("Equalized Image", equ)
    
    gray_hist = cv.calcHist([equ], [0], None, [256], [0, 256])
    
    plt.style.use('seaborn-dark-palette')
    plt.figure(figsize = (6, 4.55))
    
    img_name = image.split(".")[0]

    plt.title(f'{img_name} - Equalized Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    plt.plot(gray_hist)
    plt.xlim([0, 256])
    plt.show()

    # # stacking images side-by-side 
    # res = np.hstack((img, equ))

    return equ
