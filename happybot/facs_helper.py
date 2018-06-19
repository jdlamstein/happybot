# -*- coding: utf-8 -*-
"""This module processes facial landmarks and converts facial landmarks to facial features."""

__name__= 'facs_helper'

import numpy as np
import os
import glob
import cv2

class facialActions:
    """Get facial landmarks and find displacements of facial landmarks over time to determine
    facial features.
    
    Args:
        vec (int): 68 facial landmarks.
        img: Image.
        
    Attributes:
        newFeatures (float): Facial displacements of key landmarks. 
        
        (The following are *coordinates* from slices of the array 'vec'.
        All are type 'int'.)
        brow_ell: Left eyebrow.
        brow_r: Right eyebrow.
        eye_ell: Left eye.
        eye_r: Right eye.
        nose_line: Coordinates from tip of nose to brow. 
        nose_arc: Coordinates outlining arc from one nostril to the other. 
        lip_tu: Top upper lip. 
        lip_bl: Bottom lower lip.
        lip_tl: Top lower lip.
        lip_bu: Bottom upper lip. 
        jaw: Outline of jaw from ear to ear. 
        
        (The following are not used in the robot, but may be used to process CK+ database.
        These variables are density of lines calculated from Canny edges divided by the 
        area in the window. All have type 'float.' )
        furrow: Density of lines from canny edges between brows. 
        wrinkle_ell: Density left of left eye. 
        wrinkle_r: Density right of right eye. 
        brow_ri: Density above right inner eyebrow.
        brow_li: Density above left inner eyebrow. 
        brow_ro: Density above right outer eyebrow.
        brow_lo: Density above left outer eyebrow. 
    """
    def __init__(self,vec,img):
        dist = 10
        dist_eye = 10
        dist_shift = 10
        dist_shift_brow = 10
        self.newFeatures = []

# Declare key facial distances, ell means left, r is for right, u is for upper, b is for bottom
# u is for upper, l is for lower, i is for inner, and o is for outer. 
        self.brow_ell = vec[17:22,:]
        self.brow_r = vec[22:27,:]
        self.eye_ell = vec[36:42,:]
        self.eye_r = vec[42:48,:]  
        self.nose_line= vec[27:31,:]
        self.nose_arc = vec[31:36,:]
        self.lip_tu = vec[48:54,:]
        self.lip_bl = vec[54:60,:]
        self.lip_tl = vec[60:64,:]
        self.lip_bu = vec[64:68,:]
        self.jaw = vec[0:17,:]
        
        
        # Regions of interest can detect wrinkles between the brow
        # and on the corner of the eye. These are transient 
        # features as young people do not have as many wrinkles as
        # older people. The Canny edge detector finds lines, and the
        # algorithm computes the density over the sample area. 
        roi = img[self.nose_line[0,1] - dist:self.nose_line[0,1]+dist, self.nose_line[0,0]-dist: self.nose_line[0,0]+dist]
        roi_ell = img[self.eye_ell[0,1] - dist_eye:self.eye_ell[0,1]+dist_eye, self.eye_ell[0,0]-dist_eye - dist_shift: self.eye_ell[0,0]+dist_eye - dist_shift]
        roi_r = img[self.eye_r[3,1] - dist_eye:self.eye_r[3,1]+dist_eye, self.eye_r[3,0] - dist_eye + dist_shift: self.eye_r[3,0]+dist_eye + dist_shift]
        roi_brow_ri = img[self.brow_r[0,1] - dist - dist_shift_brow:self.brow_r[0,1]+dist - dist_shift_brow, self.brow_r[0,0] - dist: self.brow_r[0,0]+dist]
        roi_brow_li = img[self.brow_ell[4,1] - dist - dist_shift_brow:self.brow_ell[4,1]+dist - dist_shift_brow, self.brow_ell[4,0] - dist: self.brow_ell[4,0]+dist]
        roi_brow_ro = img[self.brow_r[4,1] - dist - dist_shift_brow:self.brow_r[4,1]+dist - dist_shift_brow, self.brow_r[4,0] - dist: self.brow_r[4,0]+dist]
        roi_brow_lo = img[self.brow_ell[0,1] - dist - dist_shift_brow:self.brow_ell[0,1]+dist - dist_shift_brow, self.brow_ell[0,0] - dist: self.brow_ell[0,0]+dist]
        canny = cv2.Canny(roi,50,200)
        canny_eye_r = cv2.Canny(roi_r, 50,200)
        canny_eye_ell = cv2.Canny(roi_ell, 50, 200)
        canny_brow_ri = cv2.Canny(roi_brow_ri, 50, 200)
        canny_brow_li = cv2.Canny(roi_brow_li, 100, 200)
        canny_brow_ro = cv2.Canny(roi_brow_ro, 100, 200)
        canny_brow_lo = cv2.Canny(roi_brow_lo, 100, 200)
        self.furrow = np.sum( (0 if canny is None else canny)/255) / dist**2
        self.wrinkle_ell = np.sum( (0 if canny_eye_ell is None else canny_eye_ell)/255) / dist_eye**2
        self.wrinkle_r = np.sum( (0 if canny_eye_r is None else canny_eye_r)/255) / dist_eye**2
        self.brow_ri = np.sum( (0 if canny_brow_ri is None else canny_brow_ri)/255) / dist**2
        self.brow_li = np.sum( (0 if canny_brow_li is None else canny_brow_li)/255) / dist**2
        self.brow_ro = np.sum( (0 if canny_brow_ro is None else canny_brow_ro)/255) / dist**2
        self.brow_lo = np.sum( (0 if canny_brow_lo is None else canny_brow_lo)/255) / dist**2
  
    def detectFeatures (self):
        """Get upper facial features, which are displacements over time of facial landmarks.
        Refer to https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
        
        Returns: 
            D: Distance between eyebrows.
            blo: Height between outer corner of left eye and outer left eyebrow.
            bli: Height between inner corner of left eye and inner left eyebrow.
            bri: Height between outer corner of right eye and outer right eyebrow.
            bro: Height between inner corner of right eye and inner right eyebrow.
            hl1: Height of top left eyelid from pupil.
            hr1: Height of top right eyelid from pupil.
            hl2: Height of bottom left eyelid from pupil.
            hr2: Height of bottom right eyelid from pupil.
            hl3: Height of top left eyelid from pupil using alternate coordinate.
            hr3: Height of top right eyelid from pupil using alternate coordinate.
            bl: Distance from top left eyebrow to brim of nose.
            br: Distance from top right eyebrow to brim of nose.
            n_arc: Height from brim of nose to corner of nose.
            hl_0: Height of left eye from eyelid to eyelid.
            hr_0: Height of right eye from eyelid to eyelid.
        
        """        
        D = abs(self.brow_r[0,0] - self.brow_ell[4,0]) # distance between eyebrows
        blo = abs( (self.brow_ell[0,1] + self.brow_ell[1,1])/2- self.eye_ell[0,1]) # height between outer corner of left eye and outer left eyebrow
        bli = abs( (self.brow_ell[4,1] + self.brow_ell[3,1])/2 - self.eye_ell[3,1]) # height between inner corner of left eye and inner left eyebrow

        bri = abs( (self.brow_r[0,1] + self.brow_r[1,1])/2 - self.eye_r[0,1]) # height between outer corner of right eye and outer right eyebrow

        bro = abs( (self.brow_r[4,1] + self.brow_r[3,1])/2 - self.eye_r[3,1]) # height between inner corner of right eye and inner right eyebrow

        hl1 = (1+  abs(self.eye_ell[0,1] - self.eye_ell[2,1])  )# Height of top left eyelid from pupil
        hr1 = (1+ abs(self.eye_r[3,1] - self.eye_r[1,1])  ) # Height of top right eyelid from pupil
        hl2 = (1+ abs(self.eye_ell[0,1] - self.eye_ell[4,1]) ) # Height of bottom left eyelid from pupil
        hr2 = (1+ abs(self.eye_r[3,1] - self.eye_r[5,1]) )  # Height of bottom right eyelid from pupil
        hl3 = (1 + abs(self.eye_ell[2,1] - self.nose_line[0,1]) )
        hr3 = (1 + abs(self.eye_r[1,1] - self.nose_line[0,1]) ) 

        bl = abs(self.brow_ell[2,1] - self.nose_line[0,1]) # distance from top left eyebrow to brim of nose
        br = abs(self.brow_r[2,1] - self.nose_line[0,1]) # distance from top right eyebrow to brim of nose
        n_arc = abs( (self.nose_arc[0,1] + self.nose_arc[4,1])/2 - self.nose_line[0,1]) # height from brim of nose to corner of nose
        hl_0 = abs( (self.eye_ell[1,1] + self.eye_ell[2,1])/2 - (self.eye_ell[4,1] + self.eye_ell[5,1])/2   ) # Height of left eye from eyelid to eyelid
        hr_0 = abs( (self.eye_r[1,1] + self.eye_r[2,1])/2 - (self.eye_r[4,1] + self.eye_r[5,1])/2    ) # Height of right eye from eyelid to eyelid

        self.newFeatures = [D,blo, bli,bro, bri, hl1,hr1,hl2,hr2, self.furrow, self.wrinkle_ell, self.wrinkle_r, bl, br, n_arc, hl_0, hr_0, self.brow_ri, self.brow_li, self.brow_ro, self.brow_lo, hl3, hr3]
        return self.newFeatures
    
    def detectLowerFeatures (self):
        """Get features of lower face. Features are distance of between key landmarks.
        
        Returns:
            h1: Height of top lip from corner of mouth.
            h2: Height of bottom lip from corner of mouth.
            w: Width of mouth from corner to corner of lips. 
            D_ell:  Height from left eye to left corner of mouth.
            D_r: Height from right eye to right corner of mouth.
            D_top: Height of top lip to bridge of nose.
            D_b: Height of bottom lip to bridge of nose. 
            
        """

        h1 = abs(self.lip_tu[3,1] - (self.lip_tu[0,1] + self.lip_bl[0,1]) / 2 ) # Height of top lip from corner of mouth
        h2 = abs(self.lip_bl[3,1] - (self.lip_tu[0,1] + self.lip_bl[0,1]) / 2 ) # Height of bottom lip from corner of mouth
        w = abs(self.lip_tu[0,0] - self.lip_bl[0,0]) # Width of mouth from corner to corner of lips. 
        D_ell = abs(self.lip_tu[0,1] - self.nose_line[0,1]) # Height from left eye to left corner of mouth.
        D_r = abs(self.lip_bl[0,1] - self.nose_line[0,1]) # Height from right eye to right corner of mouth.
        D_top = abs(self.lip_tu[3,1] - self.nose_line[0,1]) # Height of top lip to bridge of nose.
        D_b = abs(self.lip_bl[3,1] - self.nose_line[0,1]) # Height of bottom lip to bridge of nose. 
    
        self.newLowerFeatures = [h1,h2,w,D_ell, D_r, D_top, D_b]
        return self.newLowerFeatures
        
    def UpperFaceFeatures (self, old, new):
        """Motion of upper facial features comparing new frame to old frame.
        
        Not all values are returned for the robot. Canny edges, due to lighting, 
        were disrupting results. Performance on faces in the wild 
        improved with fewer arguments.
        
        Note that all displacements over time are scaled by the initial neutral position.
        This attempts to keep the analysis consistent for analyzing faces of different
        size and keeping the analysis scale invariant when the face is closer or farther away.
        It works okay, but the distance of the face does matter because the CK+ database
        provides faces all at the same distance from the camera. 
        
        Args:
            old: Upper static facial features from function detectFeatures.
            new: Upper static facial features from function detectFeatures.
        
        Returns:
            (all floats)
            r_D: Change in Distance between eyebrows.
            r_blo: Change in height between outer corner of left eye and outer left eyebrow.
            r_bli: Change in height between inner corner of left eye and inner left eyebrow.
            r_bri: Change in height between outer corner of right eye and outer right eyebrow.
            r_bro: Change in height between inner corner of right eye and inner right eyebrow.
            r_hl1: Change in height of top left eyelid from pupil.
            r_hr1: Change in height of top right eyelid from pupil.
            r_hl2: Change in height of bottom left eyelid from pupil.
            r_hr2: Change in height of bottom right eyelid from pupil.
            r_hl3: Change in height of bottom left eyelid from pupil.
            r_hr3: Change in height of bottom right eyelid from pupil.
            r_el: Change in left eye height. 
            r_er: Change in right eye height. 
            r_furrow: Change in density of lines from canny edges between brows. 
            r_wrinkle_ell: Change in density left of left eye. 
            r_wrinkle_r: Change in density right of right eye. 
            r_bl: Change in distance from top left eyebrow to brim of nose.
            r_br: Change in distance from top right eyebrow to brim of nose.
            r_n_arc: Change in height from brim of nose to corner of nose.
            r_hl_0: Change in height of left eye from eyelid to eyelid.
            r_hr_0: Change in height of right eye from eyelid to eyelid.
        """
        D_brow = (new[0] - old[0]) / (old[0] ) #D
        r_blo =(new[1] - old[1]) / (old[1]) #blo
        r_bli = (new[2] - old[2]) / (old[2]) #bli
        r_bro = (new[3] - old[3]) / (old[3]) #bro
        r_bri = (new[4] - old[4]) / (old[4]) #bri
        r_hl1 = (new[5] - old[5]) /( old[5]) # hl1
        r_hr1 = (new[6] - old[6]) / (old[6]) # hr1
        r_hl2 = - (new[7] - old[7]) / (old[7]) # hl2
        r_hr2 = - (new[8] - old[8]) /( old[8]) # hr2
        r_el = ((new[5] + new[7] )  - (old[5] + old[7])) / (old[5] + old[7] ) # left eye height
        r_er = ((new[6] + new[8] )  - (old[6] + old[8]) )/ (old[6] + old[8] ) # right eye height
        r_furrow = (new[9] - old[9]) / (old[9] + 1) # furrow
        r_wrinkle_ell = (new[10] - old[10]) / (old[10] + 1) # wrinkle left eye outer corner
        r_wrinkle_r = (new[11] - old[11]) / (old[11] + 1) # wrinkle right eye outer corner
        
        
        r_bl = (new[12] - old[12]) / (old[12] ) # bl
        r_br = (new[13] - old[13]) / (old[13] ) # br
        r_n_arc = (new[14] - old[14]) / (old[14] ) # n_arc
        r_hl_0 = (new[15] - old[15]) / (old[15] ) # hl_0
        r_hr_0 = (new[16] - old[16]) / (old[16] ) # hr_0
        
        r_brow_ri = (new[17] - old[17]) / (old[17] +1 ) # wrinkle above inner right eyebrow
        r_brow_li = (new[18] - old[18]) / (old[18] +1 ) # wrinkle above inner left eyebrow
        r_brow_ro = (new[19] - old[19]) / (old[19] +1 ) # wrinkle above outer right eyebrow
        r_brow_lo = (new[20] - old[20]) / (old[20] +1 ) # wrinkle about outer left eyebrow
        
        r_hl3 = (new[21] - old[21]) / old[21]
        r_hr3 = (new[22] - old[22]) / old[22]


# If you want to use different input parameters for the neural network of top facial features, change the 
# output of this function. 

##        return D_brow, r_blo, r_bli, r_bro, r_bri, r_hl1, r_hr1, r_hl2, r_hr2, r_el,r_er,r_bl, r_br, r_n_arc, r_hl_0, r_hr_0,  r_furrow, r_wrinkle_ell, r_wrinkle_r, r_brow_ri, r_brow_li, r_brow_ro, r_brow_lo, r_hl3, r_hr3
##        return D_brow, r_blo, r_bli, r_bro, r_bri, r_hl1, r_hr1, r_hl2, r_hr2, r_el,r_er,r_bl, r_br, r_n_arc, r_hl_0, r_hr_0,  r_furrow, r_wrinkle_ell, r_wrinkle_r
        return D_brow, r_blo, r_bli, r_bro, r_bri, r_hl1, r_hr1, r_hl2, r_hr2, r_el,r_er


    def LowerFaceFeatures (self, old, new):
        """Motion of lower facial features comparing new frame to old frame.
        
        Note that all displacements over time are scaled by the initial neutral position.
        This attempts to keep the analysis consistent for analyzing faces of different
        size and keeping the analysis scale invariant when the face is closer or farther away.
        It works okay, but the distance of the face does matter because the CK+ database
        provides faces all at the same distance from the camera. 
        
        Args: 
            old: Lower facial features of single frame from function detectLowerFeatures.
            new: Lower facial features of single frame from function detectLowerFeatures. 
        
        Returns:
            r_h (float): Change in lip height.
            r_w (float): Change in lip width. 
            r_ell (float):  Change in height of left lip corner to nose. 
            r_r (float): Change in height of right lip corner to nose.
            r_top (float): Change in height of top lip to bridge of nose.
            r_btm (float): Change in height of bottom lip to bridge of nose.      
        """
        
        #[h1,h2,w,D_ell, D_r, D_top, D_b]
        r_h = ( (new[0] + new[1]) - (old[0] + old[1]) ) /  (old[0] + old[1]) # lip height
        r_w = (new[2] - old[2]) / old[2] # lip width
        r_ell = - (new[3] - old[3]) / old[3] # left lip corner height to nose
        r_r = - (new[4] - old[4]) / old[4] # right lip corner height to nose
        r_top = - (new[5] - old[5]) / old[5] # top lip height to nose 
        r_btm = - (new[6] - old[6]) / old[6] # bottom lip height to nose    
        
        return r_h, r_w, r_ell, r_r, r_top, r_btm
    
    def checkProfile (self, tol):
        """Check that face is looking straight-on at camera.
        
        Check that left jaw is approximately equal to right jaw. Check that distance from eye
        to nose is approximately equal for left and right side.
        
        Args:
            tol (int): Tolerance for how much left side can differ from right side. 
            
        Returns:
            jawBool (bool): True if left and right jaw are the same within tolerance. False otherwise.
            eyeBool (bool):True if left and right jaw are the same within tolerance. False otherwise.
        """
        jawBool = abs( abs(self.nose_line[0,0] - self.jaw[0,0]) -  abs(self.jaw[-1,0] - self.nose_line[0,0]) ) < tol
        eyeBool = abs( abs(self.eye_ell[0,0] - self.nose_line[0,0] ) -  abs(self.eye_r[3,0] - self.nose_line[0,0]) ) < tol 
        
        return jawBool, eyeBool

class facsFolder:
    """Process facial action units from the CK+ database.
    Args:
        facs_folder (str): Path to FACS folder in CK+ database.
        facsList (list): All action units in CK+ database. 
        facsTop (list): Target AUs in top part of face.
        facsBtm (list): Target AUs in bottom part of face.
        facsPaper (list): Target AUs by paper by Tian. 
    
    Attributes:
        facs_folder (str): Path to FACS folder in CK+ database.
        facsList (list): All action units in CK+ database. 
        facsTop (list): Target AUs in top part of face.
        facsBtm (list): Target AUs in bottom part of face.
        facsPaper (list): Target AUs by paper by Tian. 
        facs (list): All facs in database formated as 0s and 1s.
        facsT (list): Top facs in binary.
        facsB (list): Bottom facs in binary. 
        inten (int): Ground truth intensity from CK+ referring to how intense an AU is. 
        intenT (int): Ground truth intensity for top half of face.
        AU_idx (int): Action unit index.
    
    """
  
    def __init__(self, facs_folder, facsList, facsTop,facsBtm, facsPaper):
        
        self.facs_folder = facs_folder # path to folder with ground truth of facial action coding system
        self.facsList = facsList # all AUs in CK+ database
        self.facsTop = facsTop # Targeted AUs in top part of face
        self.facsBtm = facsBtm # Targeted AUs in bottom part of face
        self.facsPaper = facsPaper # Targeted AUs by a paper by Tian
        self.facsPath = [] # Path of facs
        self.facs = [] # list of all facs in database formatted as 0s and 1s
        self.facsT = [] # list of top facs formatted as 0s and 1s
        self.facsB = [] # list of bottom facs formatted as 0s and 1s
        self.inten =[] # Ground truth intensity from CK+ referring to how intense an AU is. 
        self.intenT = []
        self.AU_idx = [] # Action unit index from search

    
    def process(self):
        """Read action units from CK+ database and store them."""

        for f in glob.glob(os.path.join(self.facs_folder, "*.txt")):
            print("Processing file: {}".format(f))
            self.facsPath.append(f)
            with open(f) as d:
                facsFrame = np.zeros(len(self.facsList))
                intenFrame = np.zeros(len(self.facsList))
                facsTopFrame = np.zeros(len(self.facsTop))
                facsBtmFrame = np.zeros(len(self.facsBtm))
                
                for i in d:
                    i = i.strip()
                    # Sorting AUs 
                    if int(float(i[0:13])) < 8: # 8 refers to AUs for the top half of the faces
                        facsTopIdx = self.facsTop.index(int(float(i[0:13])))
                        facsTopFrame[facsTopIdx] = 1

                    if np.sum( np.array(self.facsBtm) == int( float( i[0:13] ) ) ): # Sorting based on Tian's paper
                        facsBtmIdx = self.facsBtm.index(int(float(i[0:13])))
                        facsBtmFrame[facsBtmIdx] = 1
    
                    facsIdx = self.facsList.index(int(float(i[0:13])))
                    
                    facsFrame[facsIdx] = 1

                    if int(float(i[16:29]))==0:
                        hotInten = 3
                    else:
                        hotInten = int(float(i[16:29]))
                    intenFrame[facsIdx] = hotInten
                    
                    
                if len(self.facs)!=0:
                    self.facs = np.vstack((self.facs,facsFrame)) 
                    self.facsT = np.vstack((self.facsT, facsTopFrame))
                    self.facsB = np.vstack((self.facsB, facsBtmFrame))
                    self.inten = np.vstack((self.inten,intenFrame))
                else:
                    self.facs = facsFrame
                    self.inten = intenFrame
                    self.facsT = facsTopFrame
                    self.facsB = facsBtmFrame
        self.intenT = self.inten[:,:6]
        
        npFacsList = np.array(self.facsList)
        npFacsBtm = np.array(self.facsBtm)
        idxArray = np.isin(npFacsList, npFacsBtm)
        idx = np.where(idxArray)
        self.intenB = self.inten[:,idx[0]]
                
#    def AU(self, paperBool,topBool, btmBool, AU_thresh):
#        if (paperBool and not topBool and not btmBool):
#            for j in range(0,len(self.facsPaper)):
#                idxP = self.facsList.index(self.facsPaper[j]);
#                self.AU_idx.append(idxP);
#            self.AU_num = len(self.facsPaper)
#        elif (topBool and not paperBool and not btmBool):
#            for j in range(0,len(self.facsTop)):
#                idxP = self.facsList.index(self.facsTop[j]);
#                self.AU_idx.append(idxP);
#            self.AU_num = len(self.facsTop)
#        elif (btmBool and not paperBool and not topBool):
#            for j in range(0,len(self.facsBtm)):
#                idxP = self.facsList.index(self.facsBtm[j]);
#                self.AU_idx.append(idxP);
#            self.AU_num = len(self.facsBtm)
#        else:
#            AU_count = np.sum(self.facs,axis=0) # only include sets with more than AU_thresh samples
#            AU_select = AU_count>AU_thresh
#            AU_idx_list= np.where(AU_count>AU_thresh) # get indices
#            self.AU_idx = AU_idx_list[0][:]
#            self.AU_num = np.sum(AU_select) #sums True entries (23 in this case)
            
class emoFolder:
    """Convert facial action units to basic emotions.
    
    Not all sequences from the CK+ dataset have a recorded ground truth emotion.
    
    Args:
        emotion_folder (str): Path to ground truth emotions in CK+ dataset.
        facs_dir (str): Path to facial action units in CK+ dataset.
        
    Attributes:
        emotion_folder (str): Path to ground truth emotions in CK+ dataset.
        facs_dir (str): Path to facial action units in CK+ dataset.
        emotionPath (str): Store path of subject with recorded ground truth emotion. 
        emotions (int): Basic emotion of subject. 
    """
  
    def __init__(self, emotion_folder, facs_dir):
        self.emotion_folder = emotion_folder
        self.facs_dir = facs_dir
        self.emotionPath = []
        self.emotions = []
        
    def process(self):
        self.emotionPath = []
        self.emotions = []
        iter = 0
        for g in glob.glob(os.path.join(self.facs_dir)):
            iter+=1
            f = glob.glob(os.path.join(self.emotion_folder+g[39:47]+"/*.txt"))
            print(f)
            if len(f) > 0:
          #print("Processing file: {}".format(f))
                self.emotionPath.append(f[0])
                with open(f[0]) as d:
                    for i in d:
                        print(i)
                        i = i.strip()
                    if len(self.emotions)==0:
                        self.emotions = [int(i[0])]
                    else:
                        self.emotions = np.hstack((self.emotions,int(i[0]))) 
            else:
                if len(self.emotions)==0:
                  self.emotions = [-1]
                else:
                  self.emotions = np.hstack((self.emotions,-1))   

#Compare predicted emotions based on ground truth AUs with ground truth emotions
class compare:
    """Compare basic emotions predicted by FACS with ground truth emotion to test validity of
    decision tree.
    
    Args:
        emotions (int): Basic emotion of subject.
        facsT (list): Top facs in binary.
        facsB (list): Bottom facs in binary. 
        
    Attributes:
        emotions (int): Basic emotion of subject.
        facsT (list): Top facs in binary.
        facsB (list): Bottom facs in binary. 
        flagSet (int): Record predicted emotions in binary. 
        emoMatch (int): Predicted emotions. Will be compared with 'emotions'.
        matchBool (bool): True if predicted emotion matches ground truth. False otherwise.
    """
  
    def __init__(self, emotions, facsT, facsB):
        self.emotions = emotions
        self.facsT = facsT
        self.facsB = facsB
        self.flagSet = []
        self.emoMatch = []
        self.matchBool = []

    def match(self):
        """Check how predicted emotions match with ground truth emotion in CK+ dataset.
        
        Returns:
            emoMatch (int): Predicted emotions. Will be compared with 'emotions'.
            matchBool (bool): True if predicted emotion matches ground truth. False otherwise.
        """
        self.flagSet = np.zeros(len(self.emotions))
        self.emoMatch = np.ones(len(self.emotions))*-2
        for i in range(0,len(self.emotions)):
            if self.emotions[i] != -1:
                T = np.array(self.facsT[i])
                B = np.array(self.facsB[i])
                idxT = np.where(T==1)
                idxB = np.where(B==1)
                idxT = np.array(idxT)
                idxB = np.array(idxB)
                if (4 in idxT and 2 in idxB):
                  self.emoMatch[i] = 5 # happiness
                  self.flagSet[i] = 1
                elif (0 in idxT and 2 in idxT and 3 in idxB): # AU 9 (idx 1) for AU 10 (idx 2)
                  self.emoMatch[i] = 6 # sadness
                  self.flagSet[i] = 1
                elif (0 in idxT and 1 in idxT and 3 in idxT and 8 in idxB ): #exchanging jaw drop AU 26 for lips part 25
                  self.emoMatch[i] = 7 # surprise
                  self.flagSet[i] = 1
                elif (0 in idxB and (2 in idxT or 5 in idxT or 4 in idxB ) ):
                  self.emoMatch[i] = 3 # disgust (missing FAU 16)
                  self.flagSet[i] = 1
                elif (5 in idxB and (0 in idxT or 2 in idxT or 3 in idxT) ):  #exchanging jaw drop AU 26 for lips part 25
                  self.emoMatch[i] = 4 # fear
                  self.flagSet[i] = 1
                elif (6 in idxB and (4 in idxB or 7 in idxB) and (2 in idxT or 5 in idxT)):
                  self.emoMatch[i] = 1 # anger
                  self.flagSet[i] = 1
                elif (2 in idxB and 4 in idxB):
                  self.emoMatch[i] = 2 # contempt (missing FAU 14)
                  self.flagSet[i] = 1
        self.matchBool = self.emoMatch==self.emotions
        return self.emoMatch, self.matchBool