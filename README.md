# Exercise Classification using Pose Estimation and Custom CNN

## Objective
This project aims to classify exercises based on 3D pose estimation using a combination of **YOLOv5** for bounding box detection, **MediaPipe** for pose landmark detection, and a custom **Convolutional Neural Network (CNN)** for exercise classification. The system processes images of individuals performing exercises, detects their pose, and classifies the exercise based on the 3D landmarks.

---

## Approach

### Step 1: Image Acquisition
The process begins by acquiring an image using OpenCV's `imread()` function. The image is then passed through a series of steps for detection and pose estimation.

### Step 2: Bounding Box Detection Using YOLO
**YOLOv5** (You Only Look Once) is employed for detecting the person in the image. The model is pre-trained on a variety of objects, but here we focus on detecting human figures. Once a bounding box is identified around the person, we extract the coordinates of this box to crop the image to the region of interest (the person).

- **Model**: YOLOv5s (small version of YOLOv5) is used for real-time object detection.
- **Bounding Box Conversion**: YOLO outputs normalized bounding box coordinates (center_x, center_y, width, height). These are converted into pixel values to define the exact area for pose estimation.

### Step 3: 3D Pose Estimation Using MediaPipe
After detecting the bounding box, we crop the region containing the person and pass it to **MediaPipe**'s Pose Estimation model. This model estimates the 3D coordinates of key body landmarks (such as shoulders, hips, knees, etc.).

- **Landmarks**: MediaPipe detects 33 landmarks for each person, and we collect the x, y, and z coordinates of these points.
- **Pose Estimation**: The pose is processed from the cropped image to estimate the key body joints' positions in 3D space.

### Step 4: Visualizing the Predictions
After extracting the 3D landmarks, the landmarks are visualized on the original image by drawing circles at the detected positions. The bounding box is also drawn around the person, and the predicted exercise label is displayed on the image.

### Step 5: Exercise Classification using Custom CNN
A custom **CNN** is trained on 3D landmark data to classify the type of exercise being performed. The model is trained on preprocessed data, where each exercise pose is represented by the 3D coordinates of key body landmarks.

- **Architecture**: The CNN consists of multiple fully connected layers, with ReLU activations, followed by a final output layer that predicts the exercise class.
- **Data**: 3D distances and landmarks data are used as input features for the network.

### Step 6: Data Preparation and Model Training
The dataset consists of 3D distances between landmarks and labels for different exercises. The data is split into training and test sets using `train_test_split`. Label encoding is used to transform the categorical exercise labels into numerical values.

- **Training**: The model is trained using the **Cross-Entropy Loss** function and the **Adam optimizer**. The training runs for 100 epochs, and the model’s performance is evaluated based on the loss at each epoch.

### Step 7: Exercise Classification and Visualization
Once the model is trained, it can classify new images based on their pose landmarks. The landmarks from the pose estimation model are passed through the trained CNN, which predicts the exercise label. The predicted label is then visualized on the image along with the pose landmarks.

---

## Dependencies

- **OpenCV**: For image processing and bounding box detection.
- **Ultralytics YOLOv5**: For bounding box detection and object recognition.
- **MediaPipe**: For pose estimation and landmark detection.
- **PyTorch**: For building and training the custom CNN model.
- **scikit-learn**: For label encoding and train-test data split.
- **pandas**: For handling dataset operations and merging CSV data.

To install the required dependencies, use the following command:
```bash
pip install opencv-python torch torchvision ultralytics mediapipe scikit-learn pandas
