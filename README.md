# Top-Down Environment Reconstruction from a computer generated Dashcam Frame

**Author:** Robbe De Wit (20233818)  
**Course:** Digital Image Processing

### 1. Project Objective
The goal of this project is to create an automatic environment reconstruction from a single dashcam frame in a low-poly 3D style and visualize it in a top-down overview (bird's-eye view). While traditional computer vision projects often focus on photorealistic outdoor images, this project specifically focuses on stylized, virtual, or simulated environments (such as low-poly cities). I chose to use computer generated images so I don’t have to account for the imperfections that are present in real life. The system must be capable of identifying both the infrastructure (roadways) and surrounding objects (houses, buildings, vehicles) and geometrically translating them into a clear, structured map.

### 2. Methodology and Approach 
To achieve this goal, we combine two complementary approaches within digital image processing: 
1. **Classical Image Processing:**
    **For analyzing road structure and creating perspective, we use proven algorithms from OpenCV. This includes:**
   - Color segmentation and thresholding to isolate the road surface..
   - Edge detection (such as the Canny detector) for locating lane markings and contours. 
   - Perspective transformations (Inverse Perspective Mapping) to convert the angled dashcam frame into a flat top-down plane. 

2. **Simple Self-Trained Models (Machine Learning):**
    **Because standard, pre-trained AI models (such as YOLO on the COCO dataset) struggle with the unique, blocky aesthetics of low-poly objects, I (probably) need to train a custom model.**
   - Using online tools, such as Roboflow, a custom dataset can be compiled of specific low-poly elements (such as houses, buildings, and cars).
   - This lightweight, self-trained model is integrated into the Python pipeline to classify and locate objects within the frame with high accuracy.

### 3. Technical Pipeline of the System
The processing of a single frame follows sequential steps: 
1. Ingestion and Preprocessing: The dashcam frame is loaded and prepared for both AI detection and classical edge detection (including noise reduction via Gaussian blurring).
2. Object and Lane Detection: The custom YOLO model detects the positions of houses and vehicles, while classical algorithms trace the roadway and road markings.
3. Geometric Transformation: The detected elements and lane lines are projected via matrix operations into a shared top-down coordinate system. 
4. Reconstruction and Visualization: All components are merged into a unified top-down map where the position of the car, the road, and the surrounding low-poly buildings are clearly visible.

### 4. Expected results
The end product is a Python script that takes a dashcam frame as input and generates a visual overview reconstructing the environment from an aerial perspective. Obviously there are multiple flaws in this method, where car companies  use multiple sets of cameras, distance sensors or even lidar, I only have one viewpoint. This will significantly reduce the accuracy of the result. 
