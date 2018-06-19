"""Module to control servo motors.

Attributes:
    pwm: Initialize PWM device.
    yServoCenter: Centered position of servo along y-axis.
    xServoCenter: Centered position of servo along x-axis.
    yLim: Limit of displacement from center along y-axis.
    xMax: Maximum pwm position of servo along x-axis.
    xMin: Minimum pwm position of servo along x-axis.
    yMax: Maximum pwm position of servo along y-axis.
    yMin: Minimum pwm position of servo along y-axis."""

from Raspi_PWM_Servo_Driver import PWM
import numpy as np

# Initialise the PWM device using the default address
# bmp = PWM(0x40, debug=True)
pwm = PWM(0x6F, debug=True)

yServoCenter = 400
xServoCenter = 400
yLim = 210
xMax = 600
xMin = 200
yMax = 590
yMin = 270

def setServoPulse(channel, pulse):
    """Set pulse width modulation of servo."""
    pulseLength = 1000000                   # 1,000,000 us per second
    pulseLength /= 60                       # 60 Hz
    ##  print ("%d us per period" % pulseLength)
    pulseLength /= 4096                     # 12 bits of resolution
    ##  print ("%d us per bit" % pulseLength)
    pulse *= 1000
    pulse /= pulseLength
    pwm.setPWM(channel, 0, pulse)

pwm.setPWMFreq(60)                        # Set frequency to 60 Hz

def moveToCenter (shift, prevPos, lateralServo, verticalServo):
    """Move the servo to center the image on the landmark.

    Args:
        shift (int): 2-D displacement of servo center from landmark.
        prevPos (int): Previous pwm position of lateral and vertical servos.
        lateralServo (int): Identification number of lateral servo.
        verticalServo (int): Identification number of vertical servo.
    
    Returns:
        currentPos (int): Current PWM position of servo.
    """
    currentPos = prevPos
    distX = 0
    distY = 0
    div = 4 #Constant to scale shift to translate servo pwm. Found by experiment. 
    limit = 3 #Threshold in pixels for moving servo to center landmark in image.
    print('shift', shift)
    if shift[0] != 0 and abs(shift[0]) > limit:
        distX = abs(shift[0]//div)
    if shift[1] != 0 and abs(shift[1]) > limit:
        distY = abs(shift[1]//div)
    if (prevPos[0] - distX * np.sign(shift[0]) ) < xMax and (prevPos[0] - distX * np.sign(shift[0]) ) > xMin: 
        currentPos[0] = currentPos[0] - distX * np.sign(shift[0])
    if  (prevPos[1] - distY * np.sign(shift[1]) ) < yMax and (prevPos[1] - distY * np.sign(shift[1]) ) > yMin:
        currentPos[1] = currentPos[1] - distY * np.sign(shift[1])
    
    # Set servos to zero PWM. They are supported only be friction. 
    pwm.setPWM(lateralServo, 0, currentPos[0])
    pwm.setPWM(verticalServo, 0, currentPos[1])
    return currentPos

def vert_center (shift, prevPos, verticalServo):
    """Move the servo to center the image on the landmark along vertical axis.
    Use when motor wheels will turn robot.
    
    Args:
        shift (int): 2-D displacement of servo center from landmark. Only shift[1]
            is used.
        prevPos (int): Previous pwm position of lateral and vertical servos.
        verticalServo (int): Identification number of vertical servo.
    
    Returns:
        currentPos (int): Current PWM position of servo.
    """ 
    currentPos = prevPos
    distY = 0
    div = 4
    limit = 3
    if shift[1] != 0 and abs(shift[1]) > limit:
        distY = abs(shift[1]//div)
    if abs( prevPos[1] - distY * np.sign(shift[1]) ) < yLim:
        currentPos[1] = currentPos[1] - distY * np.sign(shift[1])
        
    pwm.setPWM(verticalServo, 0, currentPos[1])
    return currentPos

def move_low(lateralServo, verticalServo):
    """Point servo down."""
    pwm.setPWM(verticalServo, 0 , 550)
    pwm.setPWM(lateralServo, 0, xServoCenter)
    return [xServoCenter, 550]

def servo_stop(lateralServo, verticalServo):
    """Turn off servo. Servo will only be supported by friction of gears."""
    pwm.setPWM(verticalServo, 0 , 0)
    pwm.setPWM(lateralServo, 0, 0)
    
def move_center(lateralServo, verticalServo):
    """Center servo to look straight ahead."""
    pwm.setPWM(verticalServo, 0 , yServoCenter)
    pwm.setPWM(lateralServo, 0, xServoCenter)
    return [xServoCenter, yServoCenter]

def scan(lateralServo, verticalServo, x):
    """Move servo along x-axis."""
    pwm.setPWM(verticalServo, 0 , yServoCenter - 60)
    pwm.setPWM(lateralServo, 0, x)
    return [x, yServoCenter - 60]
    
    
    
    



