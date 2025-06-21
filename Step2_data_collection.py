import os
import cv2
import random
import shutil

DATA_ROOT = "data"
ALL_DIR = os.path.join(DATA_ROOT, "all")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
os.makedirs(ALL_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

def collect_images(class_name, num_samples=10, camera_id=0):
    """
    Collect images from camera and save to the ALL_DIR with class_name prefix.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Unable to open camera with ID {camera_id}")
        return
    count = 0
    print(f"Collecting {num_samples} images for class '{class_name}' using camera ID {camera_id}...")
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Camera error.")
            break
        cv2.imshow("Collecting - Press SPACE to capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space to capture
            filename = f"{class_name}_{count+1:02d}.jpg"
            save_path = os.path.join(ALL_DIR, filename)
            cv2.imwrite(save_path, frame)
            print(f"✅ Saved: {save_path}")
            count += 1
        elif key == ord('q'):
            print("❌ Collection cancelled.")
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"✔️ Collection complete for class '{class_name}'.\n")

def split_train_test(all_dir, train_dir, test_dir, train_ratio=0.8):
    """
    Automatically split images in all_dir into train_dir and test_dir.
    Keeps class distribution.
    """
    files = [f for f in os.listdir(all_dir) if f.endswith('.jpg')]
    class_files = {}
    for fname in files:
        label = fname.split('_')[0]
        class_files.setdefault(label, []).append(fname)
    for label, filelist in class_files.items():
        random.shuffle(filelist)
        n_train = int(len(filelist) * train_ratio)
        train_files = filelist[:n_train]
        test_files = filelist[n_train:]
        for fname in train_files:
            shutil.copy(os.path.join(all_dir, fname), os.path.join(train_dir, fname))
        for fname in test_files:
            shutil.copy(os.path.join(all_dir, fname), os.path.join(test_dir, fname))
        print(f"📂 Class '{label}': {len(train_files)} train, {len(test_files)} test images.")
    print("✅ Data splitting done.")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Collect images for a class")
    print("2. Split data into train/test")
    choice = input("Enter option (1-2): ").strip()
    if choice == "1":
        class_name = input("Enter class label (e.g. 'fabrics', 'metal', 'plastics'): ").strip()
        num_samples = int(input("How many samples to collect? "))
        camera_id_input = input("Enter camera ID (default is 0): ").strip()
        camera_id = int(camera_id_input) if camera_id_input else 0
        collect_images(class_name, num_samples, camera_id)
    elif choice == "2":
        train_ratio = float(input("Enter train split ratio (default 0.8): ") or "0.8")
        split_train_test(ALL_DIR, TRAIN_DIR, TEST_DIR, train_ratio)
    else:
        print("Invalid choice.")