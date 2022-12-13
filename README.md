# Image-Video Processing Projects with OpenCV and MediaPipe
In this projects, I used MediaPipe and OpenCV python libraries to detect face and hand landmarks. I used a Holistic model from MediaPipe solutions to detect all the face and hand landmarks. Also saw how i can access different landmarks of the face and hands which can be used for different computer vision applications such as sign language detection, drowsiness detection, etc.
(Mediapipe is a cross-platform library developed by Google that provides amazing ready-to-use ML solutions for computer vision tasks. OpenCV library in python is a computer vision library that is widely used for image analysis, image processing, detection, recognition, etc.)

###### Projects:
   - Hand Tracking
   - Finger Counting
   - Pose Estimation
   - Personal Trainer
   - Face Detection
   - Face Mesh
   - Parking Space Counter
   - Road Line Detection
   - Sleep Detection

## 1) Hand Tracking
MediaPipe Hands utilizes an ML pipeline consisting of multiple models working together: A palm detection model that operates on the full image and returns an oriented hand bounding box. A hand landmark model that operates on the cropped image region defined by the palm detector and returns high-fidelity 3D hand keypoints. Providing the accurately cropped hand image to the hand landmark model drastically reduces the need for data augmentation (e.g. rotations, translation and scale) and instead allows the network to dedicate most of its capacity towards coordinate prediction accuracy. In addition, in our pipeline the crops can also be generated based on the hand landmarks identified in the previous frame, and only when the landmark model could no longer identify hand presence is palm detection invoked to relocalize the hand.
The pipeline is implemented as a MediaPipe graph that uses a hand landmark tracking subgraph from the hand landmark module, and renders using a dedicated hand renderer subgraph. The hand landmark tracking subgraph internally uses a hand landmark subgraph from the same module and a palm detection subgraph from the palm detection module. 

+ ###### Palm Detection Model
Detecting hands is a decidedly complex task: our lite model and full model have to work across a variety of hand sizes with a large scale span (~20x) relative to the image frame and be able to detect occluded and self-occluded hands. Whereas faces have high contrast patterns, e.g., in the eye and mouth region, the lack of such features in hands makes it comparatively difficult to detect them reliably from their visual features alone. Instead, providing additional context, like arm, body, or person features, aids accurate hand localization.

First, I trained a palm detector instead of a hand detector, since estimating bounding boxes of rigid objects like palms and fists is significantly simpler than detecting hands with articulated fingers. In addition, as palms are smaller objects, the non-maximum suppression algorithm works well even for two-hand self-occlusion cases, like handshakes. Moreover, palms can be modelled using square bounding boxes (anchors in ML terminology) ignoring other aspect ratios, and therefore reducing the number of anchors by a factor of 3-5. Second, an encoder-decoder feature extractor is used for bigger scene context awareness even for small objects (similar to the RetinaNet approach). Lastly, we minimize the focal loss during training to support a large amount of anchors resulting from the high scale variance.

* ###### Hand Landmark Model
After the palm detection over the whole image mine subsequent hand landmark model performs precise keypoint localization of 21 3D hand-knuckle coordinates inside the detected hand regions via regression, that is direct coordinate prediction. The model learns a consistent internal hand pose representation and is robust even to partially visible hands and self-occlusions.

To obtain ground truth data, i have manually annotated ~30K real-world images with 21 3D coordinates, as shown below (we take Z-value from image depth map, if it exists per corresponding coordinate). To better cover the possible hand poses and provide additional supervision on the nature of hand geometry, i also render a high-quality synthetic hand model over various backgrounds and map it to the corresponding 3D coordinates.

![hand_landmarks](https://user-images.githubusercontent.com/34747978/207119099-5a6a8326-ae53-46ed-95df-4bf936492926.png)
![hand_crops](https://user-images.githubusercontent.com/34747978/207119201-b938be04-00d5-4a5c-b21e-00f0f18ae0b3.png)

## 2) Finger Counting

## 3) Pose Estimation

Human pose estimation from video plays a critical role in various applications such as quantifying physical exercises, sign language recognition, and full-body gesture control. For example, it can form the basis for yoga, dance, and fitness applications. It can also enable the overlay of digital content and information on top of the physical world in augmented reality.

Using a detector, the pipeline first locates the person/pose region-of-interest (ROI) within the frame. The tracker subsequently predicts the pose landmarks and segmentation mask within the ROI using the ROI-cropped frame as input. Note that for video use cases the detector is invoked only as needed, i.e., for the very first frame and when the tracker could no longer identify body pose presence in the previous frame. For other frames the pipeline simply derives the ROI from the previous frame’s pose landmarks.

The pipeline is implemented as a MediaPipe graph that uses a pose landmark subgraph from the pose landmark module and renders using a dedicated pose renderer subgraph. The pose landmark subgraph internally uses a pose detection subgraph from the pose detection module.

* ###### Person/Pose Detection Model
The detector is inspired by our own lightweight BlazeFace model, used in MediaPipe Face Detection, as a proxy for a person detector. It explicitly predicts two additional virtual keypoints that firmly describe the human body center, rotation and scale as a circle. Inspired by Leonardo’s Vitruvian man, we predict the midpoint of a person’s hips, the radius of a circle circumscribing the whole person, and the incline angle of the line connecting the shoulder and hip midpoints.

![pose_tracking_detector_vitruvian_man](https://user-images.githubusercontent.com/34747978/207120482-41158835-4ed5-4f23-bc48-2d925f600bf5.png)

* ###### Pose Landmark Model
The landmark model in MediaPipe Pose predicts the location of 33 pose landmarks (see figure below).

![pose_tracking_full_body_landmarks](https://user-images.githubusercontent.com/34747978/207120489-523a0423-933e-4173-b8e2-b052d7142abd.png)

https://user-images.githubusercontent.com/34747978/207123661-c06aba37-4e9f-426a-8cd8-73512fbdf63c.mp4

<img width="350" alt="1" src="https://user-images.githubusercontent.com/34747978/207121587-456316c3-c853-4fb5-b386-b0e20ec7eb7e.png">

## 4) Personal Trainer

<img width="350" alt="2" src="https://user-images.githubusercontent.com/34747978/207122950-cc26f01f-9e33-431d-a578-771e3685f5da.png">

## 5) Face Detection

MediaPipe Face Detection is an ultrafast face detection solution that comes with 6 landmarks and multi-face support. It is based on BlazeFace, a lightweight and well-performing face detector tailored for mobile GPU inference. The detector’s super-realtime performance enables it to be applied to any live viewfinder experience that requires an accurate facial region of interest as an input for other task-specific models, such as 3D facial keypoint estimation (e.g., MediaPipe Face Mesh), facial features or expression classification, and face region segmentation. BlazeFace uses a lightweight feature extraction network inspired by, but distinct from MobileNetV1/V2, a GPU-friendly anchor scheme modified from Single Shot MultiBox Detector (SSD), and an improved tie resolution strategy alternative to non-maximum suppression.

![face_detection](https://user-images.githubusercontent.com/34747978/207132154-8db79f90-e812-4a6c-9fdf-16099ba1936d.gif)

## 6) Face Mesh

MediaPipe Face Mesh is a solution that estimates 468 3D face landmarks in real-time even on mobile devices. It employs machine learning (ML) to infer the 3D facial surface, requiring only a single camera input without the need for a dedicated depth sensor. Utilizing lightweight model architectures together with GPU acceleration throughout the pipeline, the solution delivers real-time performance critical for live experiences.

<img width="350" alt="1" src="https://user-images.githubusercontent.com/34747978/207133565-c9c62b63-fdf6-468a-b552-f44a117087ad.png">  <img width="350" alt="2" src="https://user-images.githubusercontent.com/34747978/207134291-b00e038d-c9ad-4e41-b195-880421bacea0.png"> <img width="350" alt="3" src="https://user-images.githubusercontent.com/34747978/207134297-af902df9-6524-4eb8-95d3-c3a0404cdbe6.png">

## 7) Parking Space Counter

![first_frame](https://user-images.githubusercontent.com/34747978/207135035-d017e4c3-23a8-432a-bae4-f6d50af107df.png)
<img width="217" alt="1" src="https://user-images.githubusercontent.com/34747978/207367695-1dd14cdf-3e57-4ff6-84f8-854d798606d9.png">

## 8) Road Line Detection

Lane Line detection is a critical component for self driving cars and also for computer vision in general. This concept is used to describe the path for self-driving cars and to avoid the risk of getting in another lane.

<img width="400" alt="1" src="https://user-images.githubusercontent.com/34747978/207369646-03e68f31-42af-45bd-ba64-749bfa8e508f.png">
<img width="400" alt="2" src="https://user-images.githubusercontent.com/34747978/207369684-29ec2abe-6203-4cc1-9cc2-a404c7186887.png">

## 9) Sleep Detection

Driver sleep detection is a car safety technology which helps to prevent accidents when the driver gets drowsy. Various studies have suggested that around 20% of road accidents are fatigue related. A sleep alarm is used in a vehicle for detecting the condition indicative of the onset of sleepiness of a driver and for alerting the driver. An eye blink sensor is used to keep track of the driver’s eyelid motion.

<img width="1176" alt="1" src="https://user-images.githubusercontent.com/34747978/207137259-69aad106-7e51-4115-b0cb-1ba994068be5.png">
<img width="1161" alt="2" src="https://user-images.githubusercontent.com/34747978/207137275-be169dd5-7cb1-4960-949c-3787037fc5b9.png">




