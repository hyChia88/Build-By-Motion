# Build by Motion: 3D Pattern Generator and Drawing Tool

## Project Description
Build by Motion is an interactive pattern generation tool that leverages hand gesture controls to create patterns in both 2D and 3D. The application offers two main modes: 
- **Build Mode**: For creating and manipulating 3D cube patterns in space.
- **Draw Mode**: For free-form 2D pattern creation that can be transformed into 3D structures.

Users can switch between mouse/keyboard controls and hand gestures, resize cells, and apply subdivisions to create intricate geometric patterns. The project utilizes computer vision through OpenCV and MediaPipe for intuitive hand gesture controls.

## Video Demo:
https://drive.google.com/file/d/1qRN519mcnK0UUIrHIP6DIMGxLYHkUj4z/view?usp=sharing

## Run Instructions
1. **Install Required Libraries**:
   - Python 3.x
   - OpenCV: `pip install opencv-python`
   - MediaPipe: `pip install mediapipe`
   - CMU Graphics Library

2. **Running the Application**:
   - Run the `Menu.py` file to start the application. This will allow you to navigate to either the Build or Draw mode.

3. **Webcam Access**:
   - Ensure your webcam is enabled for hand gesture features.

## Shortcut Commands
### Build Mode
- **Mouse Drag**: Rotate the 3D view.
- **Up/Down**: Zoom in/out
- **Right/Left**: Change grid size
- **Hand Gesture/WASDQE**: Move the cube in the 3D space. Use one finger tip to move in the X-Y plane and two finger tips to move in the Z plane.
- **Space**: Place the cell in the grid.
- **x**: Remove cell
- **1~6**: Change cell type
- **7**: Place Image Cell
   - button to Import image for imageCell or Draw unique cell pattern
- **[ / ]**: Toggle subdivision view.
- **R**: Reset the view (current cell back to 0,0,0).
- **Escape**: Return to the start screen.

### Draw Mode
- **Mouse Click/Drag**: Draw on the grid.
- **H**: Toggle hand gesture drawing.
- **Right/Left**: Adjust the grid size.
- **R**: Reset the grid.
- **S**: Save the pattern.
- **E**: Export the pattern to Build mode.
- **Escape**: Return to the start screen.

## Notes
- Recommended screen resolution: 1200x750.
- Maximum subdivision level: 0-2.
- Ensure your webcam is accessible for gesture controls.

## Credits
- Hand gesture detection is based on MediaPipe.
- 3D visualization uses a custom projection system.
- Pattern subdivision algorithm is based on Catmull-Clark subdivision surfaces.

For more detailed information, please refer to the project documentation and source code comments.
