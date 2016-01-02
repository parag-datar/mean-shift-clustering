# mean-shift-clustering
Dependencies: Numpy, OpenCv.


A python script to segment images using the mean shift discontinuity preserving filter and clustering.

1) Edit values for the following global variables in the top of the file :-

Mode = 1 indicates that thresholding should be done based on H

Mode = 2 indicates that thresholding should be done based on Hs and Hr

Mode = 2 by default

Path to the input image

imgPath = 'Butterfly.jpg' (example)

Set appropriate values for H,Hs,Hr and Iter

H = 90

Hr = 90

Hs = 90

Iter = 100

2) Run the script mean-shift.py
