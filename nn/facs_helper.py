# -*- coding: utf-8 -*-
"""
This helper module performs tasks to process data. 
"""

__name__= 'facs_helper'

import numpy as np
import os
import glob

class facialActions:
    def __init__(self,brow_r, brow_ell, eye_ell, eye_r,lip_tu,lip_bu,lip_tl,lip_bl, nose_line,nose_arc, furrow, wrinkle_ell, wrinkle_r, brow_ri, brow_li, brow_ro, brow_lo):

# Declare key facial distances, ell means left, r is for right, u is for upper, b is for bottom
# u is for upper, l is for lower, i is for inner, and o is for outer. 
        self.brow_ell = brow_ell
        self.eye_ell = eye_ell
        self.eye_r = eye_r
        self.brow_r = brow_r
        
        self.lip_tu = lip_tu
        self.lip_bu = lip_bu
        self.lip_tl = lip_tl
        self.lip_bl = lip_bl
        self.nose_line = nose_line
        self.nose_arc = nose_arc
        self.furrow = furrow
        self.wrinkle_ell = wrinkle_ell
        self.wrinkle_r = wrinkle_r
        
        self.brow_ri = brow_ri
        self.brow_li = brow_li
        self.brow_ro = brow_ro
        self.brow_lo = brow_lo
  
    def detectFeatures (self):
        # Calculate facial features
        # To orient which landmark refers to which part of the face
        # this is helpful: https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
       
        # If the face slightly rotates: 
        
        D = abs(self.brow_r[0,0] - self.brow_ell[4,0]) # distance between eyebrows
        blo = abs( (self.brow_ell[0,1] + self.brow_ell[1,1])/2- self.eye_ell[0,1]) # height between outer corner of left eye and outer left eyebrow
        bli = abs( (self.brow_ell[4,1] + self.brow_ell[3,1])/2 - self.eye_ell[3,1]) # height between inner corner of left eye and inner left eyebrow

        bri = abs( (self.brow_r[0,1] + self.brow_r[1,1])/2 - self.eye_r[0,1]) # height between outer corner of right eye and outer right eyebrow

        bro = abs( (self.brow_r[4,1] + self.brow_r[3,1])/2 - self.eye_r[3,1]) # height between inner corner of right eye and inner right eyebrow

        hl1 = (1E-5+ abs(self.eye_ell[0,1] - self.eye_ell[2,1]) +abs(self.eye_ell[1,1] - self.eye_ell[3,1])  ) / 2 # Height of top left eyelid from pupil
        hr1 = (1E-5+ abs(self.eye_r[0,1] - self.eye_r[2,1]) +abs(self.eye_r[1,1] - self.eye_r[3,1])  ) / 2 # Height of top right eyelid from pupil
        hl2 = (1E-5+ abs(self.eye_ell[0,1] - self.eye_ell[4,1]) + abs(self.eye_ell[3,1] - self.eye_ell[5,1]) ) / 2 # Height of bottom left eyelid from pupil
        hr2 = (1E-5+ abs(self.eye_r[0,1] - self.eye_r[4,1]) + abs(self.eye_r[3,1] - self.eye_r[5,1]) ) / 2 # Height of bottom right eyelid from pupil

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
        # Motion of facial features comparing new frame to old frame
        #[h1,h2,w,D_ell, D_r, D_top, D_b, ang]
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
        return D_brow, r_blo, r_bli, r_bro, r_bri, r_hl1, r_hr1, r_hl2, r_hr2, r_el,r_er,r_bl, r_br, r_n_arc, r_hl_0, r_hr_0,  r_furrow, r_wrinkle_ell, r_wrinkle_r, r_brow_ri, r_brow_li, r_brow_ro, r_brow_lo, r_hl3, r_hr3


    def LowerFaceFeatures (self, old, new):
        #[h1,h2,w,D_ell, D_r, D_top, D_b]
        r_h = ( (new[0] + new[1]) - (old[0] + old[1]) ) /  (old[0] + old[1]) # lip height
        r_w = (new[2] - old[2]) / old[2] # lip width
        r_ell = - (new[3] - old[3]) / old[3] # left lip corner height to nose
        r_r = - (new[4] - old[4]) / old[4] # right lip corner height to nose
        r_top = - (new[5] - old[5]) / old[5] # top lip height to nose 
        r_btm = - (new[6] - old[6]) / old[6] # bottom lip height to nose    
        
        return r_h, r_w, r_ell, r_r, r_top, r_btm

class facsFolder:
  
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
        self.AU_idx = [] # Action unit index from search

    
    def process(self):

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
  
    def __init__(self, emotion_folder, facs_dir):
        self.emotion_folder = emotion_folder
        self.facs_dir = facs_dir
        
    def process(self):
        self.emotionPath = []
        self.emotions = []
        print("test")
        iter = 0
        for g in glob.glob(os.path.join(self.facs_dir)):
            iter+=1
            print(iter)
            print(g)
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
  
    def __init__(self, emotions, facsT, facsB):
        self.emotions = emotions
        self.facsT = facsT
        self.facsB = facsB

    def match(self):
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