import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


def imagePlot(image, plotTitle="plot of the image"):
    """ Plots a .jpg image or a matrix containing its values """
    
    if isinstance(image, str) and image.endswith(".jpg"):
        img = plt.imread(image)
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise ValueError("Not a correct form of image")
        
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.colorbar(fraction=0.03)
    plt.show()
    plt.title(plotTitle)
    return

def createSobelFilter(filtertype):
    """ creates the kernel of Sobel filter """
    if not (filtertype == 'vertical' or filtertype == 'horizontal'):
        raise ValueError("Not a correct 'filtertype' value")
        
    kernMatrix = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
    kernMatrix = np.array(kernMatrix)
    if filtertype == 'horizontal':
        return kernMatrix.T
    return kernMatrix

def createBlurFilter(filtertype):
    """ Creates the kernel of a blur filter: Either gaussian or moving average"""
    
    if filtertype == 'gaussian':
        kernMatrix = [[1/25, 3/25, 1/25],
                      [3/25, 9/25, 3/25],
                      [1/25, 3/25, 1/25]]
    elif filtertype == 'moving_average':
        kernMatrix = [[1/9, 1/9, 1/9],
                      [1/9, 1/9, 1/9],
                      [1/9, 1/9, 1/9]]
    else:
        raise ValueError("Not a correct 'filtertype' value")
        
    return kernMatrix

def imageFilter(imagepath, blur=0, filtertype='vertical'):
    """ Convolutes the Sobel filter with a given .jpg image.
        filtertype distinguishes between vertical and horizontal 
        filters. Correct values are 'horizontal' and 'vertical'. 
        If blur = 1, then the filter is changed from edge detector
        to blur filter and calculated the 'moving_average' or 
        'gaussian' blurs"""
    if blur == 0:
        filterMatrix = createSobelFilter(filtertype)
    elif blur == 1:
        filterMatrix = createBlurFilter(filtertype)
    else:
        raise ValueError("Wrong value of the parameter 'blur' ")
    img = plt.imread(imagepath)
    filteredImage = scipy.signal.convolve2d(img, filterMatrix)
    return filteredImage
    

def main():
    img = "natIMG.jpg"
    imagePlot(img, "Original Image")
    filteredImage = imageFilter(img, 0, "vertical")
    imagePlot(filteredImage, "Image with vertical filter")
    filteredImage = imageFilter(img, 0, "horizontal")
    imagePlot(filteredImage, "Image with horizontal filter")
    filteredImage = imageFilter(img, 1, "moving_average")
    imagePlot(filteredImage, "Image with moving_average filter")
    filteredImage = imageFilter(img, 1, "gaussian")
    imagePlot(filteredImage, "Image with gaussian filter")
    
if __name__ == "__main__":
    main()