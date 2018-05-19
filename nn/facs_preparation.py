# -*- coding: utf-8 -*-
"""
This module uses the dlib library to detect 68 facial landmarks 
and calculate key distances between them. The data are image time series
and distances from the first frame are subtracted from the last frame. 
The motion is saved and returned. 

Because this module analyzes images to fit facial landmarks, this module 
takes the most time to complete compared to the others. 
"""
# Process Data
###################################################################

__name__='facs_preparation'

import numpy as np
import os
import dlib
import glob
from skimage import io
import facs_helper
import cv2

class features():
    
    def __init__(self,faces_folder_path, predictor_path):
        self.faces_folder_path = faces_folder_path #Path of facial image data
        self._imCount = 0 # Number of images processed
        self.vec = np.empty([68, 2], dtype = int) # Hot array for holding facial landmarks
        self.landmarks = [] # All processed facial landmarks
        self._hotLand = [] # Hot array for landmarks and reshaping
        self._hotFeatures = [] # Hot array for facial features (distances calculated from facial landmarks)
        self._hotFeaturesLower = [] # Treating upper and lower facial features separately, this is the array for lower facial features
        self.features = [] # Store all features
        self.breakFolder = [] # Catch outliers in upper face data
        self.breakFolderLower =[] # Catch outliers in lower face data
        self.facialMotion = [] # Motion of upper half of face
        self.facialMotionLower = [] # Motion in lower half of face
        self.firstFeatures = [] # First image in series, which is neutral expression
        self.firstFeatures_Array = [] # Array of first images in series, neutral expressions
        self._old_f = 'boom' # Hot string to check the pathname so we can tell when we the face is a new person. 
        self.holdFeatures = [] # Array for debugging
        self.detector = dlib.get_frontal_face_detector() # Frontal face detector used for facial landmarks
        self.predictor = dlib.shape_predictor(predictor_path) # Gets facial landmarks
        
    def get_features(self):
        _seq_count = 0
        for f in glob.glob(os.path.join(self.faces_folder_path, "*.png")):
        #    print("Processing file: {}".format(f))
        
            self._finalLen = len(glob.glob(os.path.join('/Users/joshualamstein/Desktop/CK+/cohn-kanade-images/', f[53:62],"*png")))
                
            self._imCount += 1
            img = io.imread(f)
        
            # Ask the detector to find the bounding boxes of each face. The 1 in the
            # second argument indicates that we should upsample the image 1 time. This
            # will make everything bigger and allow us to detect more faces.
            dets = self.detector(img, 1)
        #    print("Number of faces detected: {}".format(len(dets)))
            for k, d in enumerate(dets):
        #        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #            k, d.left(), d.top(), d.right(), d.bottom()))
                # Get the landmarks/parts for the face in box d.
                shape = self.predictor(img, d)
                for i in range(shape.num_parts):
                    self.vec[i][0] = shape.part(i).x
                    self.vec[i][1] = shape.part(i).y
        
                # Save landmarks
                if len(self.landmarks)==0:
                    self.landmarks = self.vec
#                    self.landmarksFlat = self.vec.flatten()
                else:
                    self.landmarks = np.hstack((self.landmarks,self.vec))
#                    self.landmarksFlat = np.vstack((self.landmarksFlat, vec.flatten()))
                if len(self._hotLand)==0:
                    self._hotLand = self.vec
                else:
                    self._hotLand = np.dstack((self._hotLand,self.vec))
            
            
            if f[53:62] != self._old_f:
                _seq_count += 1 # Next sequence, new emotion
                self.seqBool = True
            else:
                self.seqBool = False # Boolean for troubleshooting
            
            self._old_f = f[53:62]
            dist = 20
            dist_eye = 15
            dist_shift = 15
            dist_shift_brow = 15
                
            if(self._finalLen == self._imCount):
                self._imCount = 0
                # Reshape array
                self._hotLand = np.asarray(self._hotLand)
                self._hotLand2 = np.transpose(self._hotLand)
                self._hotLand = np.swapaxes(self._hotLand2, 1,2)
                # Get key facial distances
                for idx, land in enumerate(self._hotLand):
                    self.brow_ell = land[17:22,:]
                    self.brow_r = land[22:27,:]
                    self.eye_ell = land[36:42,:]
                    self.eye_r = land[42:48,:]  
                    self.nose_line= land[27:31,:]
                    self.nose_arc = land[31:36,:]
                    self.lip_tu = land[48:54,:]
                    self.lip_bl = land[54:60,:]
                    self.lip_tl = land[60:64,:]
                    self.lip_bu = land[64:68,:]
                    
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
                    canny = cv2.Canny(roi,100,200)
                    canny_eye_r = cv2.Canny(roi_r, 100,200)
                    canny_eye_ell = cv2.Canny(roi_ell, 100, 200)
                    canny_brow_ri = cv2.Canny(roi_brow_ri, 100, 200)
                    canny_brow_li = cv2.Canny(roi_brow_li, 100, 200)
                    canny_brow_ro = cv2.Canny(roi_brow_ro, 100, 200)
                    canny_brow_lo = cv2.Canny(roi_brow_lo, 100, 200)
                    self.furrow = np.sum(canny/255) / dist**2
                    self.wrinkle_ell = np.sum(canny_eye_ell/255) / dist_eye**2
                    self.wrinkle_r = np.sum(canny_eye_r/255) / dist_eye**2
                    self.brow_ri = np.sum(canny_brow_ri/255) / dist**2
                    self.brow_li = np.sum(canny_brow_li/255) / dist**2
                    self.brow_ro = np.sum(canny_brow_ro/255) / dist**2
                    self.brow_lo = np.sum(canny_brow_lo/255) / dist**2

# Used for visualizing Canny edge detection in ROI. 
#                    if self._imCount % 8 == 0:
#                        cv2.imshow('ro_%i' %_seq_count, roi_brow_ro)
#                        cv2.imshow('ri_%i' %_seq_count, roi_brow_ri)
#                        cv2.imshow('lo_%i' %_seq_count, roi_brow_lo)
#                        cv2.imshow('li_%i' %_seq_count, roi_brow_li)
#                        cv2.imshow('canny_ro_%i' %_seq_count, canny_brow_ro)
#                        cv2.imshow('canny_ri_%i' %_seq_count, canny_brow_ri)
#                        cv2.imshow('canny_lo_%i' %_seq_count, canny_brow_lo)
#                        cv2.imshow('canny_li_%i' %_seq_count, canny_brow_li)

#                        cv2.imshow('image_%i' %_seq_count,img)
                        # cv2.imshow('roi_%i' %_seq_count, roi)
#                        cv2.imshow('roi_ell_%i' %_seq_count, roi_ell)
#                        cv2.imshow('roi_r_%i' %_seq_count, roi_r)
#                        cv2.imshow('canny_%i' %_seq_count, canny)
#                        cv2.imshow('canny_eye_r_%i' %_seq_count, canny_eye_r)
#                        cv2.imshow('canny_eye_ell_%i' %_seq_count, canny_eye_ell)

                    feat = facs_helper.facialActions(self.brow_r, self.brow_ell, self.eye_ell, self.eye_r,self.lip_tu,self.lip_bu,self.lip_tl,self.lip_bl, self.nose_line,self.nose_arc, self.furrow, self.wrinkle_ell, self.wrinkle_r, self.brow_ri, self.brow_li, self.brow_ro, self.brow_lo)
                    self.newFeatures = feat.detectFeatures()
                    self.newFeaturesLower = feat.detectLowerFeatures()
                    self.newFeatures = np.array(self.newFeatures)
                    self.newFeaturesLower = np.array(self.newFeaturesLower)
#                    print('newFeaturesInit', self.newFeatures)
        
                    if idx==0:
                        self.firstFeatures = self.newFeatures
                        self.firstFeaturesLower = self.newFeaturesLower

                        if np.sum(abs(self.newFeatures[:len(self.newFeatures) - 3]) < 1E-3):
                            self.breakFolder.append(f)
                            if len(self.holdFeatures) == 0:
                                self.holdFeatures = self.newFeatures
                            else:
                                self.holdFeatures = np.vstack((self.holdFeatures, self.newFeatures))
                            self.newFeatures[self.newFeatures==0] = 1E-5 # to avoid dividing by 0. 
                        if np.sum(abs(self.newFeaturesLower) < 1E-3):
                            self.breakFolderLower.append(f)
                            self.newFeaturesLower[self.newFeaturesLower==0] = 1E-5
                    self._hotFeatures = np.concatenate([self._hotFeatures, self.newFeatures]) # just for debugging
                    self._hotFeaturesLower = np.concatenate([self._hotFeaturesLower, self.newFeaturesLower])
                    self.oldFeatures = self.newFeatures
                    self.oldFeaturesLower = self.newFeaturesLower
                self.lastFeatures = self.newFeatures
                self.lastFeaturesLower = self.newFeaturesLower
                # Find changes from motion in facial features. 
                if len(self.facialMotion)==0:
                    self.newFeatures_Array = self.newFeatures
                    self.facialMotion = feat.UpperFaceFeatures(self.firstFeatures, self.lastFeatures)
                    self.facialMotionLower = feat.LowerFaceFeatures(self.firstFeaturesLower, self.lastFeaturesLower)
                else:
                    self.newFeatures_Array = np.vstack((self.newFeatures_Array, self.newFeatures))
                    self.facialMotion = np.vstack((self.facialMotion, feat.UpperFaceFeatures(self.firstFeatures, self.lastFeatures)))
                    self.facialMotionLower = np.vstack((self.facialMotionLower, feat.LowerFaceFeatures(self.firstFeaturesLower, self.lastFeaturesLower)))

                if (len(self.features)==0):
                    self.features = self._hotFeatures
                else:
                    self.features = np.hstack((self.features, self._hotFeatures))
                self._hotLand = []
                self._hotFeatures = []
                
        print("Finished features")
#        cv2.waitKey() 




