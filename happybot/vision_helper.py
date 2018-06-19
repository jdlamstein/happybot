"""Vision module for image analysis.

This module provides helper functions to identify landmarks.
"""

import cv2
import numpy as np

class vision:
    """
    Functions filter the image for a select color and get the contours of that binary image. 
    
    Args:
        img: Frame from video. 
        
    Attributes:
        img: Frame from video.
        window_x (int): Midpoint of columns.
        window_area (int): Area of frame in pixels.
        gray (int): Grayscale converted frame.
        hsv (int): HSV converted frame.
        mask (int): Binary image filtered by select color range.
    """
    def __init__(self, img):
        self.img = img
        self.window_x = np.shape(img)[1]//2
        self.window_area = np.shape(img)[0] * np.shape(img)[1]
        print('window_area', self.window_area)
        self.gray =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.mask = []
          
    def colorMask(self):
        """Filter HSV image for one color.
        
        Additional color ranges are commented."""
#        lower_brown = np.array([0, 20, 0]) ##30,84.5,20
#        upper_brown = np.array([50,200, 200])
#        lower_green = np.array([40,10,100])
#        upper_green = np.array([100,255,255])
#        lower_green_night = np.array([40,50,50])
#        upper_green_night = np.array([100,255,255])
#        lower_blue_night = np.array([95,20,20])
#        upper_blue_night = np.array([125,255,255])
#        lower_red = np.array([170,50,70])
#        upper_red = np.array([190, 255,255])
        lower_blue = np.array([100,150,20])
        upper_blue = np.array([120,255,255])

        self.mask = cv2.inRange(self.hsv,lower_blue,upper_blue)

    def get_contours(self):
        """Find contours in masked binary image. Get largest contour. 
        
        Returns:
            max_cnt (int): Coordinates of largest contour detected. 
            max_area (int): Value of the area of the largest contour. 
            """
        
        max_area = 0
        max_cnt = []
        # Get contours. 
        bin, contours, _hierarchy = cv2.findContours(self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # Get area of contours.
            cnt_area = cv2.contourArea(cnt)
            if cnt_area > max_area:
                max_area = cnt_area
                # Find largest area of detected contours. 
                if max_area > 0.01 * self.window_area:
                    max_cnt = cnt.reshape(-1,2)
                else:
                    max_area = -1
            
        if np.any(max_cnt):             
            cv2.drawContours(self.img,max_cnt.reshape(-1,1,2),-1,(0,255,0),3)

        return max_cnt, max_area