# 112 Term Project 2024 Fall
## Pattern Generator Tool based on Hand Gesture

@Author: Chia Hui Yen, huiyenc@andrew.cmu.edu  

This Python application is an interactive pattern generation tool that combines 3D modeling, hand gesture controls, and drawing capabilities. At its core, it offers two main modes: a Build mode for creating and manipulating 3D cube patterns in space, and a Draw mode for free-form 2D pattern creation that can be transformed into 3D structures. The application leverages computer vision through MediaPipe to enable intuitive hand gesture controls - users can manipulate objects in 3D space using simple finger movements, with one finger controlling X-Y plane movement and two fingers controlling Z-axis depth. The pattern generation system incorporates subdivision algorithms that can create complex fractal-like patterns from simple initial shapes, while maintaining interactive performance.

The tool features a grid-based interface with snap-to-grid functionality, multiple cell types through inheritance, and real-time 3D visualization with adjustable camera perspectives. Users can seamlessly switch between mouse/keyboard controls and hand gestures, resize cells from 1x1x1 up to 3x3x3, and apply subdivisions to create intricate geometric patterns. The application also supports importing reference images and includes comprehensive undo/redo capabilities to ensure a smooth creative workflow.

[Placeholder for sample 3D pattern image]
[Placeholder for hand gesture control demonstration image]

### Features:
#### Menu Page (MenuPage.py)
- Interactive start menu with animated title
- Multiple mode selection options:
  - Build Mode: Create 3D patterns 
  - Draw Mode: Free-form drawing with hand gestures
- Help section with instructions
- Settings configuration

#### Build Mode (Build.py)
- 3D cube pattern generation and manipulation
- Interactive rotation controls via mouse drag
- Pattern subdivision and fractal generation
- Real-time 3D visualization with perspective adjustments
- Edge and vertex manipulation
- Hand gesture controls:
  - One finger tip to move in X-Y plane
  - Two finger tips to move in Z plane
- Cell resizing capabilities (1x1x1 up to 3x3x3)
- Grid scaling
- Space key to fix cell position
- R key to reset view

#### Draw Mode (Draw.py)
- Grid-based drawing interface
- Hand gesture detection for drawing
- Pattern subdivision capabilities with adjustable levels
- Two drawing methods:
  - Mouse-based drawing
  - Hand gesture controls
- Real-time preview
- Pattern reset functionality
- 3D view toggle with space key
- Multiple cell types with inheritance
- Cell placement and removal
- Image import for reference
- Subdivision algorithm for both single and complex cells
- Snap-to-grid functionality

=============================================================================
## Similar Projects
- https://github.com/matt77harris/3d-drawing-app

=============================================================================
## Version Control / Backup Plan
The project uses with GitHub for backup and collaboration:
Remote: https://github.com/huiyenc/112-term-project-2024-fall
- Remote repository on GitHub serves as backup, regular commits tracking feature additions and bug fixes.
- Commit history provides rollback capability if needed
Local:
- Local backups saved in my own laptop, regularly pushed to remote

=============================================================================
## Tech List
### Libraries and Technologies Used
- Python 3.x
- CMU Graphics Library - For 2D graphics rendering and user interface
- OpenCV (cv2) - For webcam capture and image processing
- MediaPipe - For hand gesture detection and tracking
- Tkinter - For file dialog and basic GUI elements

### Installation
Install required packages: opencv-python mediapipe cmu-graphics

Run the application: run MenuPage.py , it will automatically run Build.py or Draw.py depending on the mode you choose. (not done this part yet)

=============================================================================
## Key Algorithms
### 3D Graphics
- Custom 3D projection system
- Rotation matrices for 3D transformations
- Perspective projection calculations
- Vertex and edge manipulation

### Pattern Generation
- Recursive subdivision algorithm
- Fractal pattern generation
- Grid-based pattern system
- Cell inheritance hierarchy

### Computer Vision
- Real-time hand tracking
- Gesture recognition
- Coordinate mapping
- Webcam integration

### User Interface
- Interactive grid system
- Mouse and keyboard controls
- Hand gesture controls
- Mode switching system

### Data Structures
- 2D and 3D arrays for grid representation
- Object-oriented cell system
- Inheritance-based cell types
- Matrix transformations

=============================================================================
## Notes
- Webcam access required for hand gesture features
- Recommended screen resolution: 1200x800
- Maximum subdivision level: 3 #(for now, not completed yet)

=============================================================================
## References & Acknowledgements

- https://skannai.medium.com/projecting-3d-points-into-a-2d-screen-58db65609f24
- https://www.youtube.com/watch?v=X8kC5p76y7s

=============================================================================
## Storyboard