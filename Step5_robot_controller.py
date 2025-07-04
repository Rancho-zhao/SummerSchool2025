import logging
import time
import sys
import lcm
from lcm_msgs import RobotCommand, RobotStatus  # Generated by lcm-gen

from lerobot.common.robots.so101_follower.so101_follower import SO101Follower
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig

class RobotController:
    def __init__(self, robot_port="/dev/ttyACM0"):
        config = SO101FollowerConfig(
            port=robot_port,
            use_degrees=True,
        )

        time.sleep(2)
        self.robot = SO101Follower(config)
        self.robot.connect()
        self.lc = lcm.LCM()
        self.lc.subscribe("ROBOT_COMMAND", self.command_handler)
        print(f"✅ RobotController initialized on port '{robot_port}' and waiting for commands...")
    
    def command_handler(self, channel, data):
        msg = RobotCommand.decode(data)
        area = msg.area.lower()
        print(f"📥 Received command for area: {area}")
        if area == "grasp_area":
            self.do_grasp_area()
        elif area in ["region_a", "region_b", "region_c"]:
            self.do_region(area)
        else:
            print(f"⚠️ Unknown area: {area}")

    def do_grasp_area(self):
        open_gripper = 60
        closed_gripper = 0
        pre_grasp_pos = {
            "shoulder_pan.pos": 0,
            "shoulder_lift.pos": 0,
            "elbow_flex.pos": 0,
            "wrist_flex.pos": 0,
            "wrist_roll.pos": 0,
            "gripper.pos": open_gripper,
        }
        print("🔧 Moving to grasp area (open gripper)...")
        self.robot.send_action(pre_grasp_pos)
        time.sleep(2)

        grasp_pos = pre_grasp_pos.copy()
        grasp_pos["gripper.pos"] = closed_gripper
        print("✊ Closing gripper to grasp object...")
        self.robot.send_action(grasp_pos)
        time.sleep(2)

        self.publish_complete()

    def do_region(self, region):
        closed_gripper = 0
        open_gripper = 60

        waypoint_pos = {
            "shoulder_pan.pos": -60,
            "shoulder_lift.pos": 0,
            "elbow_flex.pos": 0,
            "wrist_flex.pos": 0,
            "wrist_roll.pos": 0,
            "gripper.pos": closed_gripper,
        }

        region_pos = {
            "shoulder_pan.pos": -60,
            "shoulder_lift.pos": 0,
            "elbow_flex.pos": 20,
            "wrist_flex.pos": 70,
            "wrist_roll.pos": 0,
            "gripper.pos": closed_gripper,
        }

        if region == "region_b":
            waypoint_pos["shoulder_pan.pos"] = -75
            region_pos["shoulder_pan.pos"] = -75
        elif region == "region_a":
            waypoint_pos["shoulder_pan.pos"] = -100
            region_pos["shoulder_pan.pos"] = -100

        print(f"🚚 Moving to {region} with closed gripper...")
        self.robot.send_action(waypoint_pos)
        time.sleep(2)
        self.robot.send_action(region_pos)
        time.sleep(2)

        region_pos["gripper.pos"] = open_gripper
        print("👐 Opening gripper to release object...")
        self.robot.send_action(region_pos)
        time.sleep(2)

    def publish_complete(self):
        status_msg = RobotStatus()
        status_msg.completed = True
        self.lc.publish("ROBOT_STATUS", status_msg.encode())
        print("📤 Published completion signal for grasp area.")

    def run(self):
        try:
            while True:
                self.lc.handle_timeout(100)
        except KeyboardInterrupt:
            print("🛑 Shutting down...")
            self.robot.disconnect()

if __name__ == "__main__":
    # Accept robot port as argument or fallback to input/default
    if len(sys.argv) > 1:
        port_input = sys.argv[1]
    else:
        port_input = input("Enter robot port (default: /dev/ttyACM0): ").strip() or "/dev/ttyACM0"

    controller = RobotController(port_input)
    controller.run()