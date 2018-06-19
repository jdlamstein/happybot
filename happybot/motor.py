"""Raspberry Pi motor driver handler controls."""

from Raspi_MiniMoto_Driver import MOTO

motor1 = MOTO(0x60)
motor2 = MOTO(0x61)
    
def brake():
    """Brake motor."""
    motor1.brake()
    motor2.brake()
    
def forward(speed):
    """Drive forward."""
    motor1.drive(speed)
    motor2.drive(speed)

def backward(speed):
    """Go backward."""
    motor1.drive(-speed)
    motor2.drive(-speed)
    
def turn(speed, adjust_speed, dir):
    """Turn while moving forward."""
    motor1.drive(int(speed + adjust_speed * dir))
    motor2.drive(int(speed + adjust_speed * (-dir)))

def seek(speed):
    """Turn while stopped."""
    motor1.drive(-speed)
    motor2.drive(speed)
    


    

##    
##    
##
##while (1):
##    print ("Forward Speed=40!");
##    motor1.drive(40);
##    motor2.drive(40);
##    time.sleep(3);
##    print ("Stop!");
##    motor1.stop();
##    motor2.stop();
##    time.sleep(2);
##    print ("Reverse Speed=-40!");
##    motor1.drive(-40);
##    motor2.drive(-40);
##    time.sleep(3);
##    print ("Brake!");
##    motor1.brake();
##    motor2.brake();
##    time.sleep(2);
