import cv2

def find_camera_id(max_id=15):
    print("Searching for available camera devices (0-{})...".format(max_id))
    for cam_id in range(max_id + 1):
        print(f"\nTrying camera ID {cam_id}...")
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"Camera ID {cam_id} not available.")
            continue

        print(f"Camera ID {cam_id} opened. Press 'y' if this is your camera, any other key to skip.")
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            cap.release()
            continue

        cv2.imshow(f"Camera ID {cam_id}", frame)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        cap.release()

        if key == ord('y'):
            print(f"Camera ID {cam_id} selected.")
            return cam_id
        else:
            print(f"Camera ID {cam_id} not selected, moving to next.")

    print("No camera selected or found in range.")
    return None

def show_camera_feed(cam_id):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera ID {cam_id}.")
        return
    print("Press 'q' to exit live camera feed.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        cv2.imshow('Tactile Image', frame)
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Find camera ID")
    print("2. Show camera feed")
    choice = input("Enter option (1/2): ").strip()
    if choice == '1':
        cam_id = find_camera_id(15)
        if cam_id is not None:
            print(f"\nYour camera device ID is: {cam_id}")
        else:
            print("\nNo suitable camera found. Try plugging in your camera or increasing the max_id range.")
    elif choice == '2':
        cam_id = input("Enter the camera ID to use (e.g., 0): ").strip()
        try:
            cam_id = int(cam_id)
            show_camera_feed(cam_id)
        except ValueError:
            print("Invalid camera ID.")
    else:
        print("Invalid choice.")