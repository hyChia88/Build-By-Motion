# Build By Motion
*Pattern Generator Tool based on Hand Gesture, term project for 15112 CMU   
Author: Chia Hui Yen  
Mentor: Professor Austin & Lucy   
Date: 2024 Fall* 
  
<div align="center"> <br><img src = "https://github.com/user-attachments/assets/10b006f0-b43d-4268-a5a3-8a1f17ce0497"></br></div>
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0968313a-1a19-47c5-b551-9442a7c90947"/></td>
    <td><img src="https://github.com/user-attachments/assets/4c09012a-c0d4-4754-ad7e-e91edf1dea1f"/></td>
    <td><img src="https://github.com/user-attachments/assets/85d89eea-a023-4e3b-8bd0-8b0ade7ca896"/></td>
  </tr>
</table>
  
## Overview 
This application explores the potential of using body gestures to build, draw, and create geometric patterns. It is an experiment—a brief demo or prototype—to test the experience of controlling building blocks with hand gestures.   
Currently, the gestures are limited to moving cells. In my ultimate vision, this project will operate within VR, XR, or MR environments, turning it into an immersive game or creative tool.  
  
![image](https://github.com/user-attachments/assets/2657c9b0-0a80-4105-a6cd-04c0dc5d2e50)
  
In the main build page, user can use hand gesture and shortcuts to control cell to build unique patterns, and use [/] to do subdivision on geometry. Besides, to simplify the creation of complex patterns, I’ve implemented features that allow users to use basic actions or methods to visualize intricate designs. For example, users can import images to generate patterns, jump to 2d draw page draw their own designed cell pattern.  

This project applied computer vision through **OpenCV** and **MediaPipe** to enable intuitive hand gesture controls. Besides, the pattern generation method included several cell types through inheritance of class. 

Another highlight of the mode is it also incorporates **catmulk-clark subdivision algorithms** that can create smoother surface from edgy initial cube, while maintaining interactive performance. In background of running the appearance, the cube could “merge” together to generate less shape.  

Demo link: https://drive.google.com/file/d/1qRN519mcnK0UUIrHIP6DIMGxLYHkUj4z/view?usp=sharing 
