import logging
import time
import cv2
import sys
from lerobot.common.robots.so101_follower.so101_follower import SO101Follower
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig

# Initialize robot configuration
config = SO101FollowerConfig(
    port="/dev/ttyACM0",  # Change to your actual port
    use_degrees=True,
)
time.sleep(2)
robot = SO101Follower(config)
robot.connect()

# Default joint angles (in degrees)
joint_positions = {
    "shoulder_pan.pos": 0,
    "shoulder_lift.pos": 0,
    "elbow_flex.pos": 0,
    "wrist_flex.pos": 0,
    "wrist_roll.pos": 0,
    "gripper.pos": 0,
}

# Parse camera ID from command-line argument or use default
camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(camera_id)
if not cap.isOpened():
    print(f"❌ Unable to open camera with ID {camera_id}")
    robot.disconnect()
    exit()

print("✅ Robot and camera initialized.")
print("🕹️  Use keys to control joints:")
print("    q: quit")
print("    a/z: shoulder_pan ++/--")
print("    s/x: shoulder_lift ++/--")
print("    d/c: elbow_flex ++/--")
print("    f/v: wrist_flex ++/--")
print("    g/b: wrist_roll ++/--")
print("    h/n: gripper ++/--")

step = 5  # Angle change step in degrees

# Send initial pose
robot.send_action(joint_positions)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame from camera")
        break

    # Show the live camera feed
    cv2.imshow("Robot Camera View", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("🛑 Exiting...")
        break
    elif key == ord('a'):
        joint_positions["shoulder_pan.pos"] += step
    elif key == ord('z'):
        joint_positions["shoulder_pan.pos"] -= step
    elif key == ord('s'):
        joint_positions["shoulder_lift.pos"] += step
    elif key == ord('x'):
        joint_positions["shoulder_lift.pos"] -= step
    elif key == ord('d'):
        joint_positions["elbow_flex.pos"] += step
    elif key == ord('c'):
        joint_positions["elbow_flex.pos"] -= step
    elif key == ord('f'):
        joint_positions["wrist_flex.pos"] += step
    elif key == ord('v'):
        joint_positions["wrist_flex.pos"] -= step
    elif key == ord('g'):
        joint_positions["wrist_roll.pos"] += step
    elif key == ord('b'):
        joint_positions["wrist_roll.pos"] -= step
    elif key == ord('h'):
        joint_positions["gripper.pos"] += step
    elif key == ord('n'):
        joint_positions["gripper.pos"] -= step

    # Clamp gripper values if necessary
    joint_positions["gripper.pos"] = max(0, min(100, joint_positions["gripper.pos"]))

    # Send updated joint positions to robot
    robot.send_action(joint_positions)
    # print("Sent joint positions:", joint_positions)

# Cleanup
cap.release()
cv2.destroyAllWindows()
robot.disconnect()
print("✅ Camera and robot connections closed.")