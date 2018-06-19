"""Module for detecting faces, calculating facial landmarks, and setting the 
neutral expression."""

import numpy as np
import cv2
import dlib


class faceUtil:
    """Utility for facial analysis.
    
    Attributes:
        predictor_path (str): Path for dlib facial landmark predictor.
        detector: Dlib facial detector. Returns location of face.
        predictor: Gets 68 landmarks of detected face.
        vec (int): Holds landmarks of detected face. 
        neutralFeaturesUpper (float): Neutral facial features of upper face.
        neutralFeaturesLower (float): Neutral facial features of lower face.
    """

    
    def __init__(self):
        self.predictor_path = '/home/pi/pythonScripts/lib/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.vec = np.empty([68, 2], dtype = int)
        self.neutralFeaturesUpper = []
        self.neutralFeaturesLower = []
        
    def get_vec(self,image, centerFixed, face_bool):
        """Get facial landmarks of face.
        
        Returns:
            vec (int): 68 landmarks of face.
            center (int): Coordinates of center of face.
            face_bool (bool): True if face detected. Otherwise, false."""
        dets = self.detector(image, 1) # dets includes rectangle coordinates of face
        center = [] # Center coordinates of face.
        if np.any(dets):
            face_bool = True
            for k, d in enumerate(dets):
        ## For troubleshooting
        ##        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        ##            k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
                shape = self.predictor(image, d)
                # Populate facial landmarks.
                for i in range(shape.num_parts):
                    self.vec[i][0] = shape.part(i).x
                    self.vec[i][1] = shape.part(i).y
                    # Identify landmarks with filled-in green circles on image.
                    cv2.circle(image, (self.vec[i][0],self.vec[i][1]),1,(0,255,0))
                    
                x = d.left() # Column coordinate - Top left corner of detected face.
                y = d.top() # Row coordinate - Top left corner of detected face.
                w = d.right()   - x # Width.
                h = d.bottom() - y # Height. 
                
                center = np.array((x + int(w/2),y + int(h/2) ))

        else:
            face_bool = False
        return self.vec, center, face_bool
    
    def set_neutral(self, feat, newFeaturesUpper, newFeaturesLower, neutralBool, tol):
        """Set neutral expression of detected face.
        
        In this script, facial emotion is detected based on displacement from 
        a neutral facial position to an emotional position. The subject must 
        initialize the robot with their neutral or blank facial expression
        for the facial actions to be detected properly.
        
        Args:
            feat: Class for analyzing facial features. Used for checking face looks at camera.
            newFeaturesUpper (int): Facial features, candidates upper neutral expression. 
            newFeaturesLower (int): Facial features, candidates for lower neutral expression.
            tol (int): Tolerance for how much head may be turned from straight-on portrait.
        
        Returns:
            neutralBool: True if face is looking directly at the camera. False, otherwise.
            neutralFeaturesUpper (float): Neutral facial features of upper face.
            neutralFeaturesLower (float): Neutral facial features of lower face.
        """
        if (not neutralBool):
            jawBool, eyeBool = feat.checkProfile(tol) # Check if the face is looking directly at the camera. 
            if jawBool and eyeBool:
                self.neutralFeaturesUpper = newFeaturesUpper
                self.neutralFeaturesLower = newFeaturesLower
                neutralBool = True
        return neutralBool, self.neutralFeaturesUpper, self.neutralFeaturesLower
    
    def face_detect(self, image, face_bool):
        """Check if face is detected."""
        dets = self.detector(image, 1) # dets includes rectangle coordinates of face
        if np.any(dets):
            face_bool = True
        return face_bool

                
