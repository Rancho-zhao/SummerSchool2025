import lcm
import cv2
import time
import sys
from lcm_msgs import RobotCommand, RobotStatus
import numpy as np
from PIL import Image
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from Step3_texture_classify import extract_feature, load_data_with_features

# TODO: Modify the function to with your real classifier
def classify_image(frame, model, transform, knn):
    img = frame.copy()
    if img.ndim == 3:
        img = rgb2gray(img)
    feature = extract_feature(img, model, transform)
    pred = knn.predict([feature])[0]
    if pred == "fabrics":
        sort_area = "region_a"
    elif pred == "metal":
        sort_area = "region_b"
    elif pred == "plastics":
        sort_area = "region_c"
    return sort_area, pred

class PolicyNode:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.lc = lcm.LCM()
        self.lc.subscribe("ROBOT_STATUS", self.status_handler)
        self.completed = False

        self.state = "START"
        self.current_region = None
        self.pred = ""

        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.model = None
        self.transform = None

        # Load training data features
        X_train, y_train = load_data_with_features("data/train", self.model, self.transform)
        self.knn.fit(X_train, y_train)

    def status_handler(self, channel, data):
        msg = RobotStatus.decode(data)
        print(f"[Policy] Got completion signal: {msg.completed}")
        if msg.completed:
            self.completed = True

    def send_command(self, area):
        print(f"[Policy] Sending command: {area}")
        msg = RobotCommand()
        msg.area = area
        self.lc.publish("ROBOT_COMMAND", msg.encode())

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"[Policy] Failed to open camera with ID {self.camera_id}!")
            return

        plt.ion()
        fig, ax = plt.subplots()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[Policy] Camera read failed")
                    break

                frame_rgb = np.array(frame)
                ax.clear()
                ax.imshow(frame_rgb)
                ax.axis('off')
                ax.text(30, 30, self.pred, color='lime', fontsize=16, weight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=4))
                plt.pause(0.001)

                if not plt.fignum_exists(fig.number):
                    break

                # --- State machine logic ---
                if self.state == "START":
                    print("[Policy] Sending grasp_area command...")
                    self.send_command("grasp_area")
                    self.state = "WAIT_GRASP_DONE"

                elif self.state == "WAIT_GRASP_DONE":
                    self.lc.handle_timeout(100)
                    if self.completed:
                        print("[Policy] Grasp completed! Classifying image...")
                        self.current_region, self.pred = classify_image(frame, self.model, self.transform, self.knn)
                        print(f"[Policy] Classified as: {self.current_region}")
                        self.state = "SEND_REGION_COMMAND"
                        self.completed = False

                #TODO: Modify next two states to achieve continuous object sorting
                elif self.state == "SEND_REGION_COMMAND":
                    self.send_command(self.current_region)
                    print(f"[Policy] Sent command to {self.current_region}.")
                    self.state = "DONE"

                elif self.state == "DONE":
                    pass

                time.sleep(0.1)
        finally:
            cap.release()
            plt.close(fig)

if __name__ == "__main__":
    # Get camera ID from command-line argument or user input
    if len(sys.argv) > 1:
        cam_id = int(sys.argv[1])
    else:
        cam_id_input = input("Enter camera ID (default: 0): ").strip()
        cam_id = int(cam_id_input) if cam_id_input else 0

    node = PolicyNode(camera_id=cam_id)
    node.run()