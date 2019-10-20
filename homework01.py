import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def gaussian(delta, mu, xRange):
    """ Creates the univariate Gaussian probability density function """
    
    if not (isinstance(xRange, np.ndarray) or isinstance(xRange, list)):
        raise ValueError("Range must be a two elemental array or list of "
                         + "minimum and maximum range of the variable")
        return
    xpts = np.linspace(xRange[0], xRange[1], 1000, endpoint=True)
    gaus = (1 / delta*np.sqrt(2*np.pi)) *  np.exp((xpts - mu)**2 / -2*(delta**2))
    return [xpts, gaus]

def plotGaussian(delta, mu, xRange):
    """ Plots the univatiate Gaussian probability density function """
    
    [xpts, gaus] = gaussian(delta, mu, xRange)
    plt.figure()
    plt.plot(xpts, gaus)
    plt.xlabel('x')
    plt.ylabel('pdf(x)', rotation='horizontal')
    plt.title('The standard normal distribution')
    plt.show()
    plt.grid(True)
    return

def bivariateMeshgrid(xRange, yRange):
    """ Creates a meshgrid for the spasified xRange and yRange """
    if not (isinstance(xRange, np.ndarray) or isinstance(xRange, list)):
        raise ValueError("Range must be a two elemental array or list of "
                         + "minimum and maximum range of the variable")
        return
    if not (isinstance(yRange, np.ndarray) or isinstance(yRange, list)):
        raise ValueError("Range must be a two elemental array or list of "
                         + "minimum and maximum range of the variable")
        return
    x = np.linspace(xRange[0], xRange[1], 50)
    y = np.linspace(yRange[0], yRange[1], 50)
    xx, yy = np.meshgrid(x, y)
    return xx, yy

def bivariateGaussAuxVariable(deltaX, deltaY, muX, muY, ro, xRange, yRange):
    """ Creates the auxilary variable for the bivariate Gaussian probability
        density function """

    x, y =   bivariateMeshgrid(xRange, yRange)  
    z = ((x - muX)/deltaX)**2 + ((y - muY)/deltaY)**2 - (2*ro*(x - muX)*(y - muY) / (deltaX*deltaY))
    return [x, y, z]

def bivariateGaussian(deltaX, deltaY, muX, muY, ro, xRange, yRange):
    """ Creates the bivariate Gaussian probability density function """
    
    x, y, z = bivariateGaussAuxVariable(deltaX, deltaY, muX, muY, ro, xRange, yRange)
    gaus = (1 / 2*np.pi*deltaX*deltaY*np.sqrt(1 - ro**2)) * np.exp(z / -2*(1 - ro**2))
    return [x, y, gaus]
    
def plotBivariateGaussian(deltaX, deltaY, muX, muY, ro, xRange, yRange):
    """ Plots the bivariate Gaussian probability density function """
    [x, y, gaus] = bivariateGaussian(deltaX, deltaY, muX, muY, ro, xRange, yRange)
    
    fig = plt.figure()
    axis = Axes3D(fig)
    surf = axis.plot_surface(x, y, gaus, rstride=1, cstride=1,
                             cmap=cm.RdBu, linewidth=0)
    plt.title("The bivariate normal distribution")
    colBar = fig.colorbar(surf, shrink=0.5)
    colBar.set_label('pdf(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    return surf

def plotBivariateGaussianContour(deltaX, deltaY, muX, muY, ro, xRange, yRange):
    """ Creates the 2D contour plot of the bivariate Gaussian probability 
        density function """

    [x, y, gaus] = bivariateGaussian(deltaX, deltaY, muX, muY, ro, xRange, yRange)
    plt.figure()
    plt.contourf(x, y, gaus)
    colBar = plt.colorbar()
    colBar.set_label('pdf(x, y)')
    plt.title('The bivariate normal distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return

    
def main():
    plotGaussian(1, 0, [-8, 8])
    plotBivariateGaussian(1, 1, 0, 0, 0, [-8, 8], [-8, 8])
    plotBivariateGaussianContour(1, 1, 0, 0, 0, [-8, 8], [-8, 8])
    
if __name__ == '__main__':
    main()
           