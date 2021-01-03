import cv2 as cv
import matplotlib.pyplot as plt
import os

def Histogram(image):
    base_path = "./images"

    img = cv.imread(f"{base_path}/{image}", 0)
    cv.imshow("Original Image", img)

    gray_hist = cv.calcHist([img], [0], None, [256], [0, 256])
    
    plt.style.use('seaborn-dark-palette')
    plt.figure(figsize = (6, 4.55))
    plt.title('Grayscale Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    plt.plot(gray_hist)
    plt.xlim([0, 256])
    plt.show()


    cv.waitKey(0) 
    cv.destroyAllWindows()