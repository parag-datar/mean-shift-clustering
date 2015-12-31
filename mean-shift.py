# Author Parag Jayant Datar
# Date : 12 Dec 2015
# Mean Shift Discontinuity Preserving Filter

import numpy as np
from scipy import ndimage
import math
from random import randint
import matplotlib.pyplot as plt
import cv2

# Mode = 1 indicates that thresholding should be done based on H
# Mode = 2 indicates that thresholding should be done based on Hs and Hr
Mode = 2
# Path to the input image
imgPath = 'Butterfly.jpg'
# Set appropriate values for H,Hs,Hr and Iter
H = 90
Hr = 90
Hs = 90
Iter = 100

# Globals
img = cv2.imread(imgPath,cv2.IMREAD_COLOR)
opImg = np.zeros(img.shape,np.uint8)
boundaryImg = np.zeros(img.shape,np.uint8)


# Method getNeighbors
# It searches the entire Feature matrix to find
# neighbors of a pixel
# param--seed : Row of Feature matrix (a pixel)
# param--matrix : Feature matrix
# param--mode : mode=1 uses 1 value of threshold that is H
#               mode=2 uses 2 values of thresholds
#                      Hr for threshold in range domain
#                      Hs for threshold in spatial domain
# returns--neighbors : List of indices of F which are neighbors to the seed
def getNeighbors(seed,matrix,mode=1):
    neighbors = []
    nAppend = neighbors.append
    sqrt = math.sqrt
    for i in range(0,len(matrix)):
        cPixel = matrix[i]
        # if mode is 1, we threshold using H
        if (mode == 1):
            d = sqrt(sum((cPixel-seed)**2))
            if(d<H):
                 nAppend(i)
        # otherwise, we threshold using H
        else:
            r = sqrt(sum((cPixel[:3]-seed[:3])**2))
            s = sqrt(sum((cPixel[3:5]-seed[3:5])**2))
            if(s < Hs and r < Hr ):
                nAppend(i)
    return neighbors

# Method markPixels
# Deletes the pixel from the Feature matrix
# Marks the pixel in the output image with the mean intensity
# param--neighbors : Indices of pixels (from F) to be marked
# param--mean : Range and spatial properties for the pixel to be marked
# param--matrix : Feature matrix
# param--cluster : Cluster number
def markPixels(neighbors,mean,matrix,cluster):
    for i in neighbors:
        cPixel = matrix[i]
        x=cPixel[3]
        y=cPixel[4]
        opImg[x][y] = np.array(mean[:3],np.uint8)
        boundaryImg[x][y] = cluster
    return np.delete(matrix,neighbors,axis=0)

# Method calculateMean
# Calculates mean of all the neighbors and returns a 
# mean vector
# param--neighbors : List of indices of pixels (from F)
# param--matrix : Feature matrix
# returns--mean : Vector of mean of spatial and range properties
def calculateMean(neighbors,matrix):
    neighbors = matrix[neighbors]
    r=neighbors[:,:1]
    g=neighbors[:,1:2]
    b=neighbors[:,2:3]
    x=neighbors[:,3:4]
    y=neighbors[:,4:5]
    mean = np.array([np.mean(r),np.mean(g),np.mean(b),np.mean(x),np.mean(y)])
    return mean

# Method createFeatureMatrix
# Creates a Feature matrix of the image 
# as list of [r,g,b,x,y] for each pixel
# param--img : Image for which we wish to comute Feature matrix
# return--F : Feature matrix
def createFeatureMatrix(img):
    h,w,d = img.shape
    F = []
    FAppend = F.append
    for row in range(0,h):
        for col in range(0,w):
            r,g,b = img[row][col]
            FAppend([r,g,b,row,col])
    F = np.array(F)
    return F

# Method performMeanShift
# The heart of the code. This function performs the
# mean shift discontinuity preserving filtering for an image
# param--img : Image we wish to filter
def performMeanShift(img):
    clusters = 0
    F = createFeatureMatrix(img)
    # Actual mean shift implementation
    # Iterate over our Feature matrix until it is exhausted
    while(len(F) > 0):
        print 'remPixelsCount : ' + str(len(F))
        # Generate a random index between 0 and Length of 
        # Feature matrix so that we can choose a random
        # Seed
        randomIndex = randint(0,len(F)-1)
        seed = F[randomIndex]
        # Cache the seed as our initial mean
        initialMean = seed
        # Group all the neighbors based on the threshold H
        # H can be a single value or two values or range and
        # spatial fomain
        neighbors = getNeighbors(seed,F,Mode)
        print('found neighbors :: '+str(len(neighbors)))
        # If we get only 1 neighbor, which is the pixel itself,
        # We can directly mark it in our output image without calculating the shift
        # This condition helps us speed up a bit if we come across regions of single
        # pixel
        if(len(neighbors) == 1):
            F=markPixels([randomIndex],initialMean,F,clusters)
            clusters+=1
            continue
        # If we have multiple pixels, calculate the mean of all the columns
        mean = calculateMean(neighbors,F)
        # Calculate mean shift based on the initial mean
        meanShift = abs(mean-initialMean)
        # If the mean is below an acceptable value (Iter),
        # then we are lucky to find a cluster
        # Else, we generate a random seed again
        if(np.mean(meanShift)<Iter):
            F = markPixels(neighbors,mean,F,clusters)
            clusters+=1
    return clusters

# Method main
def main():
    clusters = performMeanShift(img)
    origlabelledImage, orignumobjects = ndimage.label(opImg)

    cv2.imshow('Origial Image',img)
    cv2.imshow('OP Image',opImg)
    cv2.imshow('Boundry Image',boundaryImg)
    
    cv2.imwrite('temp.jpg',opImg)
    temp = cv2.imread('temp.jpg',cv2.IMREAD_COLOR)
    labels, numobjects = ndimage.label(temp)
    fig, ax = plt.subplots()
    ax.imshow(labels)
    ax.set_title('Labeled objects')
    
    print 'Number of clusters formed : ', clusters


if __name__ == "__main__":
    main()