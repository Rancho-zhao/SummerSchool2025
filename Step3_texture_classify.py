import os
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import canny
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
import cv2
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def extract_feature(img, model, transform):
    """
    TODO: Extracts a deep feature vector from a image using the given model and transform.
    """
    mean_val = np.mean(img)
    std_val = np.std(img)
    edge_count = np.sum(canny(img))
    return [mean_val, std_val, edge_count]

def load_data_with_features(folder, model, transform):
    """
    Loads image features and labels from a folder using the model for feature extraction.
    Assumes filenames are in format 'class_xx.jpg'.
    """
    files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    X_features = []
    y_labels = []
    for fname in files:
        label = fname.split('_')[0]
        img_path = os.path.join(folder, fname)
        # img = Image.open(img_path).convert('RGB')
        img = imread(img_path)
        if img.ndim == 3:
            img = rgb2gray(img)
            
        feature = extract_feature(img, model, transform)
        X_features.append(feature)
        y_labels.append(label)
    return np.array(X_features), np.array(y_labels)

def live_webcam_demo(knn, model, transform, class_names, cam_id=0):
    print("Starting live webcam prediction. Close the window to quit.")
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Failed to open camera with ID {cam_id}.")
        return

    plt.ion()  # Interactive mode
    fig, ax = plt.subplots()
    img_disp = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            #TODO: Convert to PIL Image and predict
            img = frame.copy()
            if img.ndim == 3:
                img = rgb2gray(img)

            feature = extract_feature(img, model, transform)
            pred = knn.predict([feature])[0]
            last_prediction = f"Predicted: {pred}"

            # Overlay text using matplotlib
            frame_rgb = np.array(frame)
            ax.clear()
            ax.imshow(frame_rgb)
            ax.axis('off')
            ax.text(30, 30, last_prediction, color='lime', fontsize=16, weight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=4))
            plt.pause(0.001)

            # Optional: break if figure closed by user
            if not plt.fignum_exists(fig.number):
                break
    finally:
        cap.release()
        plt.close(fig)


def main():
    #TODO: You need to try use a pre-trained model to extract features.
    pretrained_model = None
    transform = None

    # 3. Extract features for training data
    print("Extracting features for training set...")
    train_folder = "data/train"
    X_train, y_train = load_data_with_features(train_folder, pretrained_model, transform)
    print(f"Training set: {len(y_train)} images, feature shape: {X_train.shape}")

    # 4. Extract features for test data
    print("Extracting features for test set...")
    test_folder = "data/test"
    X_test, y_test = load_data_with_features(test_folder, pretrained_model, transform)
    print(f"Test set: {len(y_test)} images, feature shape: {X_test.shape}")

    # 5. Train kNN classifier
    print("Training kNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    # 6. Test and print accuracy
    predictions = knn.predict(X_test)
    acc = np.mean(predictions == y_test)
    print(f"Overall test set accuracy: {acc*100:.2f}%")

    # 7. Per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for pred, true in zip(predictions, y_test):
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1
    print("Per-class accuracy:")
    for cls in sorted(class_total):
        print(f"  Class '{cls}': {class_correct[cls]/class_total[cls]*100:.2f}%")

    # 8. Live webcam demo
    option = input("Do you want to start live webcam prediction? (y/n): ").strip().lower()
    if option == "y":
        cam_id = input("Camera ID to use (default 0): ").strip()
        cam_id = int(cam_id) if cam_id else 0
        class_names = sorted(set(y_train))
        live_webcam_demo(knn, pretrained_model, transform, class_names, cam_id)
    else:
        print("Webcam prediction skipped.")

if __name__ == '__main__':
    main()
