import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from skimage import data
from skimage.filters import threshold_multiotsu

def MultiThreshold(imageName, thresholdValues = None, classCount = 3):

    base_path = "./images"

    # Setting the font size for all plots.
    matplotlib.rcParams['font.size'] = 9

    # The input image.
    image = cv.imread(f"{base_path}/{imageName}", 0)

    # Applying multi-Otsu threshold for the default value, generating
    # three classes.

    thresholds = threshold_multiotsu(image, classCount)

    if thresholdValues:
        if isinstance(thresholdValues, list):
            thresholds = thresholdValues
        else:
            print("Threshold Values Must Be List!!!")
            print("The Program Continues With Multi Otsu Method!!!")

    # Using the threshold values, we generate the three regions.
    regions = np.digitize(image, bins = thresholds)

    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 5))

    # Plotting the original image.
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    # Plotting the histogram and the two thresholds obtained from
    # multi-Otsu.
    ax[1].hist(image.ravel(), bins=255)
    ax[1].set_title('Histogram')
    for thresh in thresholds:
        ax[1].axvline(thresh, color='r')

    # Plotting the Multi Otsu result.
    ax[2].imshow(regions, cmap='jet',)
    ax[2].set_title('Multi-Otsu result')
    ax[2].axis('off')

    plt.subplots_adjust()

    plt.show()