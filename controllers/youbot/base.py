"""
Copyright 1996-2024 Cyberbotics Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Description: Python wrapper for YouBot base control
"""

from controller import Robot
import math

# Constants
SPEED = 4.0
MAX_SPEED = 0.3
SPEED_INCREMENT = 0.05
DISTANCE_TOLERANCE = 0.001
ANGLE_TOLERANCE = 0.001

# Robot geometry
WHEEL_RADIUS = 0.05
LX = 0.228  # longitudinal distance from robot's COM to wheel [m]
LY = 0.158  # lateral distance from robot's COM to wheel [m]

def bound(value, min_val, max_val):
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))


class Base:
    """Controls the YouBot mobile base with omnidirectional wheels"""
    
    def __init__(self, robot):
        """Initialize base motors and sensors
        
        Args:
            robot: Webots Robot instance
        """
        self.robot = robot
        self.time_step = int(robot.getBasicTimeStep())
        
        # Get wheel motors
        self.wheels = [
            robot.getDevice("wheel1"),
            robot.getDevice("wheel2"),
            robot.getDevice("wheel3"),
            robot.getDevice("wheel4")
        ]
        
        # Set wheels to velocity control mode
        for wheel in self.wheels:
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)
        
        # Movement state
        self.vx = 0.0 
        self.vy = 0.0  
        self.omega = 0.0 
        
    def _set_wheel_speeds_helper(self, speeds):
        """Set wheel velocities from a list of 4 speeds
        
        Args:
            speeds: list of 4 wheel speeds
        """
        for i in range(4):
            self.wheels[i].setVelocity(speeds[i])
    
    def move(self, vx, vy, omega):
        """Set wheel velocities for omnidirectional movement using proper kinematics"""
        speeds = [0.0] * 4
        speeds[0] = (1.0 / WHEEL_RADIUS) * (vx - vy - (LX + LY) * omega)  # front-left
        speeds[1] = (1.0 / WHEEL_RADIUS) * (vx + vy + (LX + LY) * omega)  # front-right
        speeds[2] = (1.0 / WHEEL_RADIUS) * (vx + vy - (LX + LY) * omega)  # rear-left
        speeds[3] = (1.0 / WHEEL_RADIUS) * (vx - vy + (LX + LY) * omega)  # rear-right
        
        self._set_wheel_speeds_helper(speeds)
        self.vx = vx
        self.vy = vy
        self.omega = omega

    
    def reset(self):
        """Stop all wheel movements"""
        speeds = [0.0, 0.0, 0.0, 0.0]
        self._set_wheel_speeds_helper(speeds)
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
    
    def forwards(self):
        """Move forward at SPEED"""
        speeds = [SPEED, SPEED, SPEED, SPEED]
        self._set_wheel_speeds_helper(speeds)
    
    def backwards(self):
        """Move backward at SPEED"""
        speeds = [-SPEED, -SPEED, -SPEED, -SPEED]
        self._set_wheel_speeds_helper(speeds)
    
    def turn_left(self):
        """Rotate counter-clockwise at SPEED"""
        speeds = [-SPEED, SPEED, -SPEED, SPEED]  
        self._set_wheel_speeds_helper(speeds)

    def turn_right(self):
        """Rotate clockwise at SPEED"""
        speeds = [SPEED, -SPEED, SPEED, -SPEED]  
        self._set_wheel_speeds_helper(speeds)

    def strafe_left(self):
        """Strafe left at SPEED"""
        speeds = [SPEED, -SPEED, -SPEED, SPEED]  
        self._set_wheel_speeds_helper(speeds)

    def strafe_right(self):
        """Strafe right at SPEED"""
        speeds = [-SPEED, SPEED, SPEED, -SPEED]
        self._set_wheel_speeds_helper(speeds)
 
    
    def forwards_increment(self):
        """Increment forward velocity"""
        self.vx += SPEED_INCREMENT
        self.vx = min(self.vx, MAX_SPEED)
        self.move(self.vx, self.vy, self.omega)
    
    def backwards_increment(self):
        """Increment backward velocity"""
        self.vx -= SPEED_INCREMENT
        self.vx = max(self.vx, -MAX_SPEED)
        self.move(self.vx, self.vy, self.omega)
    
    def turn_left_increment(self):
        """Increment left rotation velocity"""
        self.omega += SPEED_INCREMENT
        self.omega = min(self.omega, MAX_SPEED)
        self.move(self.vx, self.vy, self.omega)
    
    def turn_right_increment(self):
        """Increment right rotation velocity"""
        self.omega -= SPEED_INCREMENT
        self.omega = max(self.omega, -MAX_SPEED)
        self.move(self.vx, self.vy, self.omega)
    
    def strafe_left_increment(self):
        """Increment left strafe velocity"""
        self.vy += SPEED_INCREMENT
        self.vy = min(self.vy, MAX_SPEED)
        self.move(self.vx, self.vy, self.omega)
    
    def strafe_right_increment(self):
        """Increment right strafe velocity"""
        self.vy -= SPEED_INCREMENT
        self.vy = max(self.vy, -MAX_SPEED)
        self.move(self.vx, self.vy, self.omega)
