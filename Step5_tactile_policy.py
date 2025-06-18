import lcm
import cv2
import time
from lcm_msgs import RobotCommand, RobotStatus
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from Step3_texture_classify import extract_deep_feature

# === Placeholder: Replace with your real classifier ===
def classify_image(frame, model, transform, knn):
    # For demo: always return region_a
    # Replace with your model logic!
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    feature = extract_deep_feature(img, model, transform)
    pred = knn.predict([feature])[0]
    if pred=="nut":
        sort_area = "region_a"
    elif pred=="head":
        sort_area = "region_b"
    elif pred=="fabrics":
        sort_area = "region_c"
    return sort_area, pred
    # return "region_a"

class PolicyNode:
    def __init__(self):
        self.lc = lcm.LCM()
        self.lc.subscribe("ROBOT_STATUS", self.status_handler)
        self.completed = False

        self.state = "START"
        self.current_region = None
        self.pred = None

        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.model = None
        self.transform = None

        # Load training data features
        from Step3_texture_classify import load_data_with_features
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
        cap = cv2.VideoCapture(9)  # Adjust if needed
        if not cap.isOpened():
            print("[Policy] Failed to open camera!")
            return
        plt.ion()  # Interactive mode
        fig, ax = plt.subplots()
        img_disp = None
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[Policy] Camera read failed")
                    break

                # Show real-time camera image
                frame_rgb = np.array(frame)
                ax.clear()
                ax.imshow(frame_rgb)
                ax.axis('off')
                ax.text(30, 30, self.pred, color='lime', fontsize=16, weight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=4))
                plt.pause(0.001)

                # Optional: break if figure closed by user
                if not plt.fignum_exists(fig.number):
                    break

                # --- State machine logic ---
                if self.state == "START":
                    print("[Policy] Sending grasp_area command...")
                    self.send_command("grasp_area")
                    self.state = "WAIT_GRASP_DONE"

                elif self.state == "WAIT_GRASP_DONE":
                    # Wait for the completion signal
                    self.lc.handle_timeout(100)  # Check for LCM message every 100ms
                    if self.completed:
                        print("[Policy] Grasp completed! Classifying image...")
                        # Classify the current frame
                        self.current_region, self.pred = classify_image(frame, self.model, self.transform, self.knn)
                        print(f"[Policy] Classified as: {self.current_region}")
                        self.state = "SEND_REGION_COMMAND"
                        self.completed = False  # Reset flag

                elif self.state == "SEND_REGION_COMMAND":
                    self.send_command(self.current_region)
                    print(f"[Policy] Sent command to {self.current_region}.")
                    self.state = "DONE"

                elif self.state == "DONE":
                    # Keep showing camera, or break if you want to stop after one cycle
                    pass

                #TODO: Modify this part to handle multiple regions

                time.sleep(0.1)
        finally:
            cap.release()
            plt.close(fig)
            # cv2.destroyAllWindows()

if __name__ == "__main__":
    node = PolicyNode()
    node.run()