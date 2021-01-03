import cv2 as cv

def Dilation(image):
    dilated = cv.dilate(image, (3, 3), iterations = 3)
    # 1. param is image
    # 2. param is kernel
    # 3. param is iterations
    return dilated

def Erosion(image):
    eroded = cv.erode(image, (3, 3), iterations = 3)
    # 1. param is image
    # 2. param is kernel
    # 3. param is iterations
    return eroded

def Opening(image):
    eroded = Erosion(image)
    opened = Dilation(eroded)
    return opened

def Closing(image):
    dilated = Dilation(image)
    closed = Erosion(dilated)
    return closed
