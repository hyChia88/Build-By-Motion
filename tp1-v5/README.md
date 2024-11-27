# 112 Term Project 2024 Fall
# 3D Pattern Generator and Drawing Tool
@Author: Chia Hui Yen, huiyenc@andrew.cmu.edu  

A Python application that combines 3D pattern generation, hand gesture controls, and interactive drawing features.

## Features

### Menu Page (MenuPage.py)
- Interactive start menu with animated title
- Multiple mode selection options:
  - Build Mode: Create 3D patterns
  - Draw Mode: Free-form drawing with hand gestures
- Help section with instructions
- Settings configuration

### Build Mode (Build.py)
- 3D cube pattern generation
- Interactive rotation controls
- Pattern subdivision and fractal generation
- Real-time 3D visualization
- Camera perspective adjustments
- Edge and vertex manipulation

### Draw Mode (Draw.py)
- Grid-based drawing interface
- Hand gesture detection for drawing
- Pattern subdivision capabilities
- Two drawing methods:
  - Mouse-based drawing
  - Hand gesture controls
- Real-time preview
- Pattern reset functionality
- For advanced development:
    will use space to view the drawing in 3d, getting the same projection algo from Build.py

## Controls:
### Build Mode
- Mouse drag: Rotate the 3D view
- hand gesture to move the cube in the 3D space, one finger tip to move in X Y plane, two finger tip to move in Z plane
- [/]: toggle subdivision view (not completed yet)
- Space: make cell fixed (valid)
- R: Reset view (curr cell back to 0,0,0)

### Draw Mode
- Mouse click/drag: Draw on grid
- H: Toggle hand gesture drawing
- S: Toggle subdivision view
- UP/DOWN: Adjust subdivision levels
- R: Reset grid

## Requirements
- Python 3.x
- OpenCV (cv2)
- MediaPipe
- CMU Graphics Library

## Installation
1. Install required packages:
opencv-python 
mediapipe 
cmu-graphics

2. Run the application:
run MenuPage.py , it will automatically run Build.py or Draw.py depending on the mode you choose. #(not done this part yet)

## Technical Details
- Uses MediaPipe for hand gesture detection
- Implements 3D projection and rotation matrices in Build.py
- Features pattern subdivision algorithms in Draw.py
- Real-time webcam integration
- Grid-based drawing system in Draw.py

## Notes
- Webcam access required for hand gesture features
- Recommended screen resolution: 1200x800
- Maximum subdivision level: 3 #(for now, not completed yet)

## Credits
- Hand gesture detection based on MediaPipe
- 3D visualization using custom projection system
- Pattern subdivision algorithm implementation