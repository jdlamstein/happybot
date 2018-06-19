"""Robot maneuvers objects and reacts to happy facial expressions.

The facial sentiment recognition is from a neural network based on the paper,
"Recognizing Action Units for Facial Expression Analysis" by Tian, Kanade, and Cohn, which
was trained on the CK+ dataset. The facial recognition was executed by the dlib library through
Histogram of Oriented Gradients. The object detection based on masking a color and finding
contours with openCV. 

The robot behaves in the following way:

1) The robot looks for a landmark, which in this case is a blue cup. If no blue cup
is seen, the robot turns and looks again. 

2) Finding the blue cup, the robot moves towards it until the blue cup doubles in size
or fills up a large portion of the frame. 

3) The robot scans its surroundings for a face. If a face is not found past the time limit, 
return to step (1). 

4) If a face is detected, the robot turns toward the face and orients itself such that
the servo motor is point forward and the face is centered in the frame. If a face is not
detected, return to step (1).

5) When the face is centered, the neutral expression must be set. (The sentiment recognition 
is cooperative. The robot reads a person's neutral expression and predicts emotions
based on that.) To show the robot is setting the neutral expression, it moves back and forth
as a visual cue. When the person looks straight-on at the robot, the neutral position is set 
and the robot ceases to rock back and forth. 

6) The robot predicts the emotions of the face. If happiness is predicted, the robot 
responds by spinning quickly. After happiness is detected, return to step (1).

Notes:

After step (4), if the face vanishes from frame for sufficient time, the program returns
to step (1). 

After step (5), if the face vanishes from frame, the neutral facial expression must be reset.
"""

# For facial recognition dlib documentation refer to:
# dlib models from : https://github.com/davisking/dlib-models
# Face rec process : https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
import Servo
import tf_helper
import facs_helper
import motor
import blob_detector
import face_helper

# Label Facial Action Units (AUs) and Basic Emotions.
dict_upper = ['AU1: Inner Brow Raiser','AU2: Outer Brow Raiser','AU4: Brow Lowerer','AU5: Upper Lid Raiser','AU6: Cheek Raiser','AU7: Lid Tightener']
dict_lower = ['AU9: Nose Wrinkler', 'AU10: Upper Lip Raiser', 'AU12: Lip Corner Puller', 'AU15: Lip Corner Depressor',  'AU17: Chin Raiser',  'AU20: Lip Stretcher',  'AU23: Lip Tightener','AU24: Lip Pressor', 'AU25: Lips Part',  'AU26: Jaw Drop',  'AU27: Mouth Stretch']
dict_emotion = ['Thinking...', 'Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

# Font size for text on video.
font_size = .6
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize Dlib
face_op = face_helper.faceUtil()

# Facial landmarks
vec = np.empty([68, 2], dtype = int)

skip = 2 # Skip analysis for video speed
iter = 0 # Number of iterations
tol = 5 # Tolerance for setting neutral expression profile. Verifies eye and ear separation
# are approximately equal on the left and right side. Checks the person is looking straight at the camera. 

# Initialize the camera and grab a reference to the raw camera capture.
camera = PiCamera()
camera.resolution = (640, 480)
camera.vflip = True
camera.hflip = False
camera.framerate = 32
width = 640
height = 480
rawCapture = PiRGBArray(camera, size=(width, height))

# Allow the camera to warmup.
time.sleep(0.1)

# Reduces images size for image processing.
scaleFactor = 0.4
# Scale up at the end for viewing.
scaleUp = 3/4
# Get center of image.
centerFixed = np.array((int(width*scaleFactor / 2), int(height*scaleFactor /2) ))

# Boolean flags for workflow.
go_no_go = True # Control boolean for testing. 
centered_bool = False
oriented_bool = False
go_bool = True
face_seek_bool = False
face_bool = False
servoBool = True
second_seek_bool = False
# Count number of times face is detected for pacing. A face must not be detected by not_face_count_thresh threshold
# before the algorithm resets. 
face_bool_count = 0
not_face_count = 0
not_face_count_thresh = 30
# Speed of motor when happy face is detected. 
happy_speed = 20
# Driving speed.
speed = -10
# Adjust speed to stay on target if necessary. 
adjust_speed = 2
# Threshold for determining if a landmark is centered along the y-axis.
thresh = 15
# Move servo motor along the y-axis.
y_shift = np.array([0,20])
# Set x and y center of frame.
yCenter = centerFixed[1]
xCenter = centerFixed[0]
# Rotate servo motor when seeking a face by dx.
dx = 40
# Approximate conversion of PWM current divisions to angle. 
degrees_per_div = 50/140
# Height of robot camera in cm. 
robot_height = 13
# Camera Position.
prevPos = []
cameraPos = []

# Threshold for determining if a landmark is centered along the x-axis. The motor turns if not. 
motor_thresh = int(50 * scaleFactor)

div = 3 # integer division for scaling step in centering camera. Value is found by experiment. 

# Position of text on video when face is detected. 
pos_lower = (np.arange(175, 450, 25)*scaleUp).astype(int)
pos_upper = (np.arange(25,175,25)*scaleUp).astype(int)
pos_emotion = (np.arange(25,225,25)*scaleUp).astype(int)

# Counts iterations when face is not found. 
iter_count = 0
# Stores facial features of upper face. 
groundFeatUpper = []
# Stores facial features of lower face. 
groundFeatLower = []
# Stores change in facial features of upper face. 
facialMotionUpper = []
# Stores change in facial features of lower face. 
facialMotionLower = []
# Boolean flag for when neutral expresion is set. 
neutralBool = False

# Tensorflow model path.
load_file_low = '/home/pi/nn/bottom/model-3000'
load_file_up = '/home/pi/nn/top2/model-3000'

# Tensorflow model for lower and upper face. 
modelLow = tf_helper.ImportGraph(load_file_low)
modelUp = tf_helper.ImportGraph(load_file_up)

# For finding landmarks and orienting robot. 
blob = blob_detector.blobFinder(xCenter, yCenter, thresh, motor_thresh)

# Set servo motors to look straight ahead. 
cameraPos = Servo.move_center(2,3)

 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    if iter % 4 ==0:
        iter = 0
    iter += 1
    image = frame.array
    # Make frame smaller for image.
    small_frame = cv2.resize(image, (0, 0), fx=scaleFactor, fy=scaleFactor)
    if not oriented_bool and not face_seek_bool and not face_bool:
        # Detect landmark.
        mean_sq, mean_max, max_area, blob_bool = blob.find_blob(small_frame)
        if blob_bool: # Landmark found
            motor.brake()
            centered_bool, xShift, yShift = blob.check_centered(mean_max)
        else: # Landmark not found. 
            motor.seek(speed) # Turn robot.
        if not centered_bool and blob_bool: # Landmark is not centered, but is detected.
            # So we center it. 
            cameraPos = blob.orient_to_blob(xShift, yShift, speed, adjust_speed,\
                                            cameraPos)
        else:
            # Landmark is centered. 
            oriented_bool = True
    elif go_bool and not face_seek_bool and not face_bool:
        # Drive forward until landmark is twice as big or fills up a large fraction of the image. 
        go_bool = blob.advance(small_frame, speed, adjust_speed, cameraPos, centered_bool, blob_bool)
    elif (oriented_bool and not go_bool and not face_bool) or face_seek_bool:
        # Check if face is detected. 
        face_bool = face_op.face_detect(small_frame, face_bool)
        if face_bool:
            # Face is detected, no need to look for a face any longer. 
            face_seek_bool = False
            print('face detected')
        else:
            # Keep looking for a face.
            face_seek_bool = True
            # Turn servo laterally to point camera at different direction. 
            dx*= -1 if cameraPos[0] > 480 or cameraPos[0] < 280 else 1
            cameraPos[0]+=dx
            cameraPos = Servo.scan(2,3,cameraPos[0])
            time.sleep(.5)
            iter_count += 1
            if iter_count > 10 and go_no_go:
                # If the robot doesn't find a face for a while, it resets and will find a new landmark. 
                cameraPos = Servo.move_center(2,3)
                motor.seek(speed) # Turn so the robot doesn't find the current landmark. 
                time.sleep(1)
                motor.brake() # Stop
                # Reset sequence. 
                iter_count = 0
                oriented_bool = False
                centered_bool = False
                go_bool = True
                face_seek_bool = False
                second_seek_bool = False

    elif face_bool: # Face is detected.
        face_bool_count +=1
        # Get facial landmarks and position of face on image. 
        vec, point, face_bool = face_op.get_vec(small_frame, centerFixed, face_bool)
##        cv2.circle(small_frame, (point[0], point[1]),5,(0,255,0),-1)
##        time.sleep(5)
        
        # Get facial features. 
        feat = facs_helper.facialActions(vec,small_frame)
        newFeaturesUpper = feat.detectFeatures()
        newFeaturesLower = feat.detectLowerFeatures()
        if not go_no_go: # Boolean for testing. Robot persists in facial sentiment recognition. 
            face_bool = True
        if np.any(point) and face_bool_count > 5: # Facial location is detected and the face has been detected
            # for at least 5 frames. 
            face_bool_count = 6 # Don't let the variable blow up.
            # Check if the robot is pointing right at the person's face. 
            centered_face_bool, xShift_face, yShift_face = blob.check_centered(point)
            straighten_bool = blob.check_straighten(cameraPos)
            print('centered', centered_face_bool, 'straighten', straighten_bool)
            print('cameraPos', cameraPos)
            if not centered_face_bool or not straighten_bool: # If the robot is not pointing at the person's face,
                # straighten up. 
                cameraPos = blob.straighten_up(xShift_face, yShift_face, 0, adjust_speed * 4,\
                                                cameraPos)
            else:
                # Visual cue that the robot is about to set the face's neutral expression. 
                if not neutralBool:
                    motor.backward(speed)
                    time.sleep(.3)
                    motor.forward(speed)
                    time.sleep(.3)
                    motor.brake()
                    time.sleep(.5)
                # Set neutral expression if face is properly aligned. 
                neutralBool, neutralFeaturesUpper, neutralFeaturesLower \
                             =face_op.set_neutral(feat, newFeaturesUpper, newFeaturesLower, neutralBool,tol)

        # Increase size of frame for viewing. 
        big_frame = cv2.resize(small_frame, (0, 0), fx=scaleUp * 1/scaleFactor, fy=scaleUp *1/scaleFactor)

        # Show text on video. 
        for idxJ, dd in enumerate(dict_upper):
          cv2.putText(big_frame, dd,(10,pos_upper[idxJ]), font, font_size,(255,255,255),2,cv2.LINE_AA)
        for idxJ, dd in enumerate(dict_lower):
          cv2.putText(big_frame, dd,(10,pos_lower[idxJ]), font, font_size,(255,255,255),2,cv2.LINE_AA)
        for idxJ, dd in enumerate(dict_emotion):
          cv2.putText(big_frame, dd,(380,pos_emotion[idxJ]), font, font_size,(255,255,255),2,cv2.LINE_AA)

        # Neutral expression is set. 
        if neutralBool:
            # Just reshape variables. 
            facialMotionUp = np.reshape(feat.UpperFaceFeatures(neutralFeaturesUpper, newFeaturesUpper),(-1,11))
            facialMotionLow = np.reshape(feat.LowerFaceFeatures(neutralFeaturesLower, newFeaturesLower),(-1,6))
            
            # Predict AUs with TF model. 
            facsLow = modelLow.run(facialMotionLow)
            facsUp = modelUp.run(facialMotionUp)
            # Predict emotion based on AUs. 
            feel = tf_helper.facs2emotion(facsUp[0,:], facsLow[0,:])
            emotion = feel.declare()

            # Get index of AUs.
            idxFacsLow = np.where(facsLow[0,:]==1)
            idxFacsUp = np.where(facsUp[0,:]==1)
            if emotion == 5: # If emotion is happiness.
                # Robot spins. 
                motor.seek(happy_speed)
                time.sleep(3)
                # Robot stops. 
                motor.brake()
                # Trigger reset. 
                face_bool = False
                not_face_count= not_face_count_thresh+1
                
            # Write text on frame. 
            if len(idxFacsLow) >0:
                for ii in idxFacsLow[0]:
                    cv2.putText(big_frame, dict_lower[ii],(10,pos_lower[ii]), font, font_size,(255,0,0),2,cv2.LINE_AA)
            if len(idxFacsUp) >0:
                for jj in idxFacsUp[0]:
                    cv2.putText(big_frame, dict_upper[jj],(10,pos_upper[jj]), font, font_size,(255,0,0),2,cv2.LINE_AA)  
            cv2.putText(big_frame, dict_emotion[emotion],(380,pos_emotion[emotion]), font, font_size,(255,0,0),2,cv2.LINE_AA)  

    if not face_bool:
        # Resize frame for viewing. 
        big_frame = cv2.resize(small_frame, (0, 0), fx=scaleUp * 1/scaleFactor, fy=scaleUp *1/scaleFactor)
        if face_bool_count > 0 and go_no_go:
            not_face_count +=1
            if face_bool_count < 3 and not second_seek_bool:
                face_seek_bool = True
                face_bool = False
                second_seek_bool = True
        if face_bool_count > 0 and not_face_count > not_face_count_thresh and go_no_go:
            # Reset. Robot will find a new landmark. 
            oriented_bool = False
            centered_bool = False
            neutralBool = False
            go_bool = True
            second_seek_bool = False
            not_face_count = 0
            face_bool_count = 0
            cameraPos = Servo.move_center(2,3)
        elif face_bool_count > 0:
            # Robot will analyze the face. Perhaps the face was not detected in a frame as an error. 
            face_bool = True
    cv2.imshow("Frame", big_frame)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()
# Stop servo and motor. 
Servo.servo_stop(2,3)
motor.brake()



