# Build By Motion
Author: Chia Hui Yen  
Mentor: Professor Austin & Lucy   
Date: 2024 Fall 

## Overview 
This application explores the potential of using body gestures to build, draw, and create geometric patterns. It is an experiment—a brief demo or prototype—to test the experience of controlling building blocks with hand gestures. Currently, the gestures are limited to moving cells. In my ultimate vision, this project will operate within VR, XR, or MR environments, turning it into an immersive game or creative tool.
In the main build page, user can use hand gesture and shortcuts to control cell to build unique patterns, and use [/] to do subdivision on geometry.  
Besides, to simplify the creation of complex patterns, I’ve implemented features that allow users to use basic actions or methods to visualize intricate designs. For example, users can import images to generate patterns, jump to 2d draw page draw their own designed cell pattern.  
This project applied computer vision through OpenCV and MediaPipe to enable intuitive hand gesture controls - users can manipulate objects in 3D space using simple finger movements, with one finger controlling X-Y plane movement and two fingers controlling Z-axis depth. Besides, the pattern generation method included several cell types through inheritance of class. Another highlight of the mode is it also incorporates catmulk-clark subdivision algorithms that can create smoother surface from edgy initial cube, while maintaining interactive performance. In background of running the appearance, the cube could “merge” together to generate less shape.
