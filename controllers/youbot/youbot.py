from controller import Robot
from base import Base
from arm import Arm
from gripper import Gripper

class YouBotController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        
        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.time_step)

        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.time_step)
        
        
    def run(self):
        raise NotImplementedError("This method should be implemented")

if __name__ == "__main__":
    controller = YouBotController()
    #controller.run()