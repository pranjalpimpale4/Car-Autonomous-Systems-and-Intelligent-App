# Advanced Driver Assistance System (ADAS)

## Project Overview
The **Advanced Driver Assistance System (ADAS)** is designed to enhance road safety and driver convenience by integrating real-time computer vision and machine learning algorithms. The system detects and interprets key driving environment elements, providing features such as:

- Lane Detection
- Traffic Light Detection
- Stop Sign Detection
- Pedestrian Detection
- Vehicle Detection

This repository contains the source code and documentation for the ADAS project, including technical details, implementation strategies, and evaluation metrics.

---

## Features

### 1. Lane Detection
- **Algorithm**: Hough Transform
- **Capabilities**: 
  - Detects lanes in various lighting/weather conditions.
  - Supports road types such as highways, urban roads, and intersections.
- **Performance**:
  - Target throughput: 30–35 fps
  - Achieved precision: 0.8+; recall: 0.8+

### 2. Traffic Light Detection
- **Algorithm**: Haar Cascades
- **Capabilities**:
  - Detects traffic light states (RED, YELLOW, GREEN).
- **Performance**:
  - Target throughput: 30–35 fps
  - Achieved precision: 0.8+; recall: 0.8+

### 3. Stop Sign Detection
- **Algorithm**: Haar Cascades
- **Capabilities**:
  - Detects stop signs and provides timely driver alerts.
- **Performance**:
  - Target throughput: 30–35 fps
  - Achieved precision: 0.09+; recall: 0.08+

### 4. Pedestrian Detection
- **Algorithm**: Haar Cascades
- **Capabilities**:
  - Detects pedestrians with pose and behavior recognition.
- **Performance**:
  - Target throughput: 30–35 fps
  - Achieved precision: 0.76+; recall: 0.74+

### 5. Vehicle Detection
- **Algorithm**: Haar Cascades
- **Capabilities**:
  - Detects vehicles, differentiates types.
- **Performance**:
  - Target throughput: 30–35 fps
  - Achieved precision: 0.82+; recall: 0.8+

---

## Implementation Details

### Technologies Used
- **Programming Language**: C++
- **Frameworks/Libraries**: OpenCV
- **Hardware**: NVIDIA Jetson Nano
- **Video Processing**:
  - Real-time processing with Canny Edge Detection and Hough Transform.
  - Object detection using Haar Cascade classifiers.

### Key Algorithms
1. **Lane Detection**: Combines edge detection, line detection, and image blending for precise lane identification.
2. **Object Detection**:
   - Traffic lights, stop signs, pedestrians, and vehicles using pre-trained Haar Cascade models.
3. **Integration**:
   - Frame-by-frame real-time processing with annotations (bounding boxes and overlays).

---

### Prerequisites
- Install OpenCV and necessary dependencies.
- Use an NVIDIA Jetson Nano or a similar embedded system for optimal performance.


### Conclusion
The ADAS project successfully demonstrates a proof-of-concept for enhancing road safety through real-time video processing. Future enhancements include incorporating predictive algorithms, deep learning models, and cooperative collision avoidance mechanisms.

