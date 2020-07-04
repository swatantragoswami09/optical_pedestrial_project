#!/usr/bin/python
# -*- coding: utf-8 -*-

# Python 2/3 compatibility
from __future__ import print_function


import numpy as np
import cv2
import time
import sys





#np.set_printoptions(threshold='nan')Read a new frameRead a new frame




def draw_flow(img, flow, step=8):
    #from the beginning to position 2 (excluded channel info at position 3)
    h, w = img.shape[:2]
    #print(img.shape)->(360,480)
    #int, int

    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    #array, array

    fx, fy = flow[y,x].T
    #array, array
    
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    #array

    lines = np.int32(lines + 0.5)
    #array
    
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    (h, w) = flow.shape[:2]
    (fx, fy) = (flow[:, :, 0], flow[:, :, 1])
       
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
     

    '''
    for i in range(len(v)):
    	for j in range(len(v[i])): 		
    		if v[i][j]<1:
    			v[i][j]=0
    		else:
    			v[i][j]=0XFF
    
    '''
    
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 0xFF   
    hsv[..., 2] = v
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('hsv', bgr)
    return bgr

count=0

maxAllowed=2000
minAllowed=50

if __name__ == '__main__':
 
    try:
        fn = sys.argv[1]
    except:
        fn = 0

    cam = cv2.VideoCapture(fn)
    #prev- ndarray, ret-bool
    (ret, prev) = cam.read()

    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while cam.isOpened():
        (ret, img) = cam.read()
        org = img
        vis = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	        
        blackAndWhite = gray
        """
        Computes a dense optical flow using the Gunnar Farneback’s algorithm.
        cv2.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, 
        poly_n=size of pixel neighbourhood, poly_sigma=deviation, flags) → flow	
        """
        #flow-ndarray
        flow = cv2.calcOpticalFlowFarneback(prevgray,gray,None,0.5,5,5,3,5,1.1,cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)
        prevgray = gray

        dotted_flow = draw_flow(gray, flow)
        cv2.imshow('flow', dotted_flow)
        
        HSV = draw_hsv(flow)
        gray1 = cv2.cvtColor(HSV, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray1,0, 0xFF,cv2.THRESH_BINARY)[1]
        thresh = cv2.medianBlur(thresh, 5)  #kernel size = 5                   
        #thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=1)
        cv2.imshow('thresh',thresh)
        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#retrieval mode, contour approximation method	
        # loop over the contours
        """
        if len(cnts)>0:   
            maxAllowed = maxAvg = max(maxAvg, sum([cv2.contourArea(cnt) for cnt in cnts])/len(cnts))
        """
        
        modifiedCnts = [cnt for cnt in cnts if minAllowed < cv2.contourArea(cnt) < maxAllowed]
        
        	
        cv2.drawContours(vis, modifiedCnts, -1, (0,255,0), 1)        
        textNoOfPeople = "People: " +str(len(modifiedCnts))
        cv2.putText(vis, textNoOfPeople, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)  
        count += 1
        
    
        #print ('Read a new frame')
        #if len(modifiedCnts)>=30:
        #    cv2.imwrite("contour%d.jpg" % count, vis)
        #cv2.imwrite("original%d.jpg" % count, org)
        #cv2.imwrite("dotted_flow%d.jpg" % count, dotted_flow)            
        #cv2.imwrite("BinaryImage%d.jpg" % count, thresh)
        cv2.imshow('Image', vis)
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break    
    cv2.destroyAllWindows()
