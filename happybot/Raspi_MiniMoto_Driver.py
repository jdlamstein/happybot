#!/usr/bin/python

# Copyright 2014 Herman Liu , Geekroo Technologies
#
#Example code demonstrating the use of the python Library for the MiniMoto Copper
#from Geekroo, which uses the DRV8830 IC for I2C low-voltage DC motor control.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software")
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from Raspi_I2C import Raspi_I2C
import time

# ===========================================================================
# DRV8830 Class
# ===========================================================================
class MOTO:
    i2c = None
    # DRV8830 Registers
    __DRV8830_CONTROL           = 0x00  
    __DRV8830_FAULT             = 0x01 

    # Constructor
    def __init__(self, address=0x60,debug=False):
        self.i2c = Raspi_I2C(address)

        self.address = address
        self.debug = debug
    #Send the drive command over I2C to the DRV8830 chip. Bits 7:2 are the speed
    #setting; range is 0-63. Bits 1:0 are the mode setting:
    #- 00 = Standby/coast(HI-Z)
    #- 01 = Reverse
    #- 10 = Forward
    #- 11 = brake(H-H)
    def drive(self, speed):
        self.i2c.write8(self.__DRV8830_FAULT, 0x80) #Clear the fault status.
        speedval = abs(speed)
        if speedval > 63:            #Cap the value at 63
            speedval = 63 
        speedval = speedval << 2  #Left shift to make room for bits 1:0
        if speed < 0:                     #Set bits 1:0 based on sign of input
            speedval |= 0x01 #Reverse
        else:
            speedval |= 0x02 #Forward
        self.i2c.write8(self.__DRV8830_CONTROL, speedval) #control the moto
        return 1
        
    #Coast to a stop by hi-z'ing the drivers.
    def stop(self):
        self.i2c.write8(self.__DRV8830_CONTROL, 0x00) #Standby
        return 1
        
    #Stop the motor by providing a heavy load on it.
    def brake(self):
        self.i2c.write8(self.__DRV8830_CONTROL, 0x03)#brake
        return 1
            
    #Return the fault status of the DRV8830 chip. Also clears any existing faults.
    def getFault(self):
        fault = self.i2c.readU8(self.__DRV8830_FAULT)
        self.i2c.write8(self.__DRV8830_FAULT, 0x80) #Clear the fault status.
        return fault
