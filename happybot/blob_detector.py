"""Module for analyzing landmarks, contours, and image data."""

import numpy as np
import vision_helper
import Servo
import motor
import time

class blobFinder:
    """Analyze landmarks, contours, and image data.
    
    The functions find blobs based on color masking and detecting contours. 
    The largest contour is targeted and the robot moves towards it until the area
    of the contour is too large. 
    
    Args:
        xCenter (int): Column center of image frame.
        yCenter (int): Row center of image frame.
        thresh (int): Threshold for determining if a landmark is centered along the y-axis.
        motor_thresh (int): Threshold for determining if a landmark is centered along the x-axis.
        servo_thresh (int): Threshold for centering servo on landmark. 
        dist_count (int): Count iterations of robot moving towards landmark.
    """
    def __init__(self, xCenter, yCenter, thresh, motor_thresh):
        self.skip = 2
        self.yCenter = yCenter
        self.xCenter = xCenter
        self.x_servo_center = 390
        self.servo_thresh = 12
        self.thresh = thresh
        self.dist_count = 0
        self.motor_thresh = motor_thresh
        
    def find_blob(self, img):
        """Detect blobs and get blob with largest area.
        
        Args:
            img: Image.
            
        Returns:
            mean_max (int): Center coordinates of largest contour.
            max_area (int): Area of largest contour.
            blob_bool (bool): True if blob detected. False otherwise.
        """
        mean_max = np.array([-1,-1])
        blob_bool = False
        vis = vision_helper.vision(img)
        vis.colorMask()
        max_cnt, max_area = vis.get_contours()
        if np.any(max_cnt):
            mean_max = np.mean(max_cnt,0)
        if np.all(mean_max > np.array([0,0])):
            blob_bool = True
            
        return mean_max, max_area, blob_bool
    
    def check_centered(self,point):
        """Check that landmark is in the center of the frame.
        
        Args:
            point (int): Coordinates of landmark.
        
        Returns:
            centered_bool (bool): True if centered. Otherwise, false.
            xShift (int): Displacement along x-axis of landmark from center of frame.
            yShift (int): Displacement along y-axis of landmark from center of frame.
        """
        centered_bool = False
        xShift = 0
        yShift = 0
        if np.all(point > np.array([0,0])):
            yShift = self.yCenter - point[1]
            xShift = self.xCenter - point[0]
            if abs(yShift) < self.thresh and abs(xShift) < self.motor_thresh:
                centered_bool = True
                Servo.servo_stop(2,3)
        return centered_bool, xShift, yShift
    
        
    def check_straighten(self,cameraPos):
        """Check if the servo is pointing straight along the x-axis.
        
        Args:
            cameraPos (int): Position of servos.
            
        Returns:
            straighten_bool (bool): True if servo is centered. Otherwise, false.
        """
        straighten_bool = False
        camShift = self.x_servo_center - cameraPos[0]
        if abs(camShift) < self.servo_thresh:
            straighten_bool = True
        return straighten_bool
        

    def orient_to_blob(self,xShift, yShift, current_speed, adjust_speed, cameraPos): 
        """Move servo along y-axis to center landmark in frame. Rotate robot with wheels
        to center landmark along x-axis in frame.
        
        Args:
            xShift (int): Displacement of landmark from center of frame along x-axis.
            yShift (int): Displacement of landmark from center of frame along y-axis.
            current_speed (int): Current speed of motors.
            adjust_speed (int): Small change of speed to turn while moving. 
            cameraPos (int): Position of servos holding camera.
        
        Returns:
            cameraPos (int): Position of servos holding camera.        
        """
        servo_shift = [0, int(yShift)]
        cameraPos = Servo.vert_center (servo_shift, cameraPos, 3)
        motor.turn(current_speed,adjust_speed,np.sign(xShift))
        print('yshift',yShift)
        print('xShift',xShift)
        return cameraPos
    
    def straighten_up(self, xShift, yShift, current_speed, adjust_speed, cameraPos):
        """If landmark or target is detected on the camera, but the servo is not aligned along
        the x-axis, align the servo motor while keeping the landmark centered.

        Args:
            xShift (int): Displacement of landmark from center of frame along x-axis.
            yShift (int): Displacement of landmark from center of frame along y-axis.
            current_speed (int): Current speed of motors.
            adjust_speed (int): Small change of speed to turn while moving. 
            cameraPos (int): Position of servos holding camera.
            
        Returns:
            cameraPos (int): Position of servos holding camera.     
        """
        camShift = self.x_servo_center - cameraPos[0]
        print('sign xShift', np.sign(xShift))
        print('sign servo center', np.sign(camShift))
        if np.sign(xShift) == -np.sign(camShift) \
           and abs(xShift) > self.thresh:
            servo_shift = [int(xShift), int(yShift)]
            cameraPos = Servo.moveToCenter(servo_shift, cameraPos, 2,3)
        else:
            servo_shift = [0, int(yShift)]
            cameraPos = Servo.vert_center (servo_shift, cameraPos, 3)
            if abs(camShift) > self.servo_thresh:
                motor.turn(current_speed,adjust_speed,np.sign(camShift))
            else:
                motor.turn(current_speed,adjust_speed,np.sign(xShift))
            time.sleep(0.2)
            motor.brake()
        print('yshift',yShift)
        print('xShift',xShift)
        return cameraPos
    
    def advance(self, img, speed, adjust_speed, cameraPos, centered_bool, blob_bool):
        """Drive the robot forward toward the detected object.
        
        Args:
            img: Image.
            speed (int): Speed of robot motors. 
            adjust_speed (int): Small change of speed to turn while moving. 
            cameraPos (int): Position of servos holding camera.
            centered_bool (bool): True if centered. Otherwise, false.
            blob_bool (bool): True if blob detected. False otherwise.
            
        Returns:
            False if object's area is large beyond threshold. Otherwise, true.   
        
        """
        mean_sq, mean_max, max_area, bool_x, bool_y, blob_bool = self.find_blob(img)
        if blob_bool:
            motor.forward(speed)
            centered_bool, xShift, yShift = self.check_centered(mean_max)
            if not centered_bool:
                self.orient_to_blob(xShift, yShift, speed, adjust_speed, \
                                                cameraPos)
            if self.dist_count == 0:
                self.init_area = max_area
            else:
                if max_area > 1.8 * self.init_area or max_area > (2*self.xCenter * 2*self.yCenter ) // 4:
                    motor.brake()
                    self.dist_count= 0
                    return False
            self.dist_count +=1
            if self.dist_count > 100:
                self.dist_count = 1
        else:
            motor.seek(speed)
        return True
        
        
