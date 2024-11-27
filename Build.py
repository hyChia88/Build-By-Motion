'''
This is the building page that allows the user to build on a grid with blocks (cells)
'''
from cmu_graphics import *
import math
import cv2
# import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog

# Take reference from Week7 assignment tetris3D
class Grid3D:
    def __init__(self, cellSize, gridSize):
        self.cellSize = cellSize
        self.gridSize = gridSize
        # it will be a cube
        self.board = [[[None for x in range(self.gridSize)]
                        for y in range(self.gridSize)]
                        for z in range(self.gridSize)]
        print(self.board) # for debug
    
    def __repr__(self):
        return f"Grid3D(cellSize={self.cellSize}, gridSize={self.gridSize})"
    
    # check if the position is valid, return True or False
    def isPosValid(self, cell):
        positionList = cell.getPlacementPos()
        if isinstance(cell, Cell):
            for pos in positionList:
                if (0<=pos[0]<self.gridSize and
                    0<=pos[1]<self.gridSize and
                    0<=pos[2]<self.gridSize):
                    return True
                if self.board[pos[2]][pos[1]][pos[0]] is not None:
                    return False
        return False
    
    # place the cube at the position, return True or False
    def placeCube(self,cell):
        if self.isPosValid(cell):
            # the order of x,y,z is different from the order of the board
            self.board[cell.z][cell.y][cell.x] = cell
            return True
        return False
    
    # get the cube at the position, return the cell or None
    def getCube(self, cell):
        if self.isPosValid(cell):
            # the order of x,y,z is different from the order of the board
            return self.board[cell.z][cell.y][cell.x]
        return None
        
'''
Cell class, frac level I left it for future use
'''
class Cell:
    def __init__(self, x, y, z, fracLevel):
        self.fracLevel = fracLevel
        #  x,y,z here is position
        self.x = x
        self.y = y
        self.z = z
        self.size = 1
        self.fracLevel = fracLevel
        self.resizable = True
        self.pattern = [[[True]]] # 1x1x1 cube
    
    def __repr__(self):
        return f"Cell({self.x}, {self.y}, {self.z}, {self.size})"
        
    def fracCell(self):
        self.x = self.x *2
        self.y = self.y *2
        self.z = self.z *2
        
    def getPattern(self):
        return self.pattern
    
    def isResizable(self):
        return self.resizable
    
    def resize(self, newSize):
        if not self.resizable:
            return False
        elif 1<= newSize <= 4:
            self.size = newSize
            self.pattern = [[[True for _ in range(newSize)] for _ in range(newSize)] for _ in range(newSize)]
            return True
        return False

    '''return a list of (x,y,z), get the exact pattern pos of cell, send to board and make it not None (occupied)'''
    def getPlacementPos(self):
        posList = []
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    posList.append((self.x+x, self.y+y, self.z+z))
        return posList
    
'''
Various types of cells, inherit from Cell class
'''
class LShapeCell(Cell):
    def __init__(self, x, y, z, fracLevel):
        super().__init__(x, y, z, fracLevel)
        self.resizable = False
        self.pattern = [[[True, False], [True, True]]]

class TShapeCell(Cell):
    def __init__(self, x, y, z, fracLevel):
        super().__init__(x, y, z, fracLevel)
        self.resizable = False
        self.pattern = [[[True, False, False], [True, True, True], [True, False, False]]]
        
class LineCell(Cell):
    def __init__(self, x, y, z, fracLevel):
        super().__init__(x, y, z, fracLevel)
        self.resizable = False
        self.pattern = [[[True, True, True]]]

class StairCell(Cell):
    def __init__(self, x, y, z, fracLevel=1):
        super().__init__(x, y, z, fracLevel)
        self.resizable = False
        self.pattern = [[[True, False], [False, False]],
                        [[True, True], [False, False]],
                        [[True, True], [True, False]]]

# Mainly reference from https://youtu.be/RRBXVu5UE-U?si=FTBWxNPHmmu-KmW6
class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.prevX = None
        self.prevY = None
        self.prevZ = None
        self.moveInZ = False
        self.counter = 0
        self.swipeThreshold = 0.05
        self.cap = cv2.VideoCapture(0)

    def detectGesture(self):
        ret, frame = self.cap.read()
        if not ret:
            return self.counter
            
        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frameRGB)
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            h, w = frame.shape[:2]
            
            currentHandX = hand.landmark[8].x
            currentHandY = hand.landmark[8].y
            # Draw red circle on index finger
            indexX = int(currentHandX * w)
            indexY = int(currentHandY * h)
            cv2.circle(frame, (indexX, indexY), 10, (0, 0, 255), -1)
            
            if hand.landmark[8] and hand.landmark[12]:
                midFingerX = hand.landmark[12].x
                midFingerY = hand.landmark[12].y
                midIndexX = int(midFingerX * w)
                midIndexY = int(midFingerY * h)
                currentHandZ = hand.landmark[12].y
                if abs(midIndexX - indexX) < 50 and abs(midIndexY - indexY) < 50:
                    self.moveInZ = True
                    self.prevZ = currentHandZ
                    cv2.circle(frame,(midIndexX, midIndexY), 10, (0, 255, 0), -1)
                else:
                    self.moveInZ = False

            self.prevX = currentHandX
            self.prevY = currentHandY
        else:
            self.prevX = None
            self.prevY = None
            self.prevZ = None
        cv2.imshow('Hand Gesture Counter', frame)
        # something like FPS 帧数
        cv2.waitKey(1)
        # Make sure the value is between 0 and 1
        if self.prevX is not None and self.prevY is not None:
            if self.prevX > 1:
                self.prevX = 1
            elif self.prevX < 0:
                self.prevX = 0
            elif self.prevY > 1:
                self.prevY = 1
            elif self.prevY < 0:
                self.prevY = 0
            if self.prevZ is not None:
                if self.prevZ > 1:
                    self.prevZ = 1
                elif self.prevZ < 0:
                    self.prevZ = 0
        return self.prevX, self.prevY, self.prevZ

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Reference Sudoku-3D\game3D.py, builder\CMU-15112-\isometric.py by Kui Yang Yang, https://skannai.medium.com/projecting-3d-points-into-a-2d-screen-58db65609f24 and modified it to fit my needs
class Projection3D:
    def __init__(self):
        pass

    def basicProj(self, app, x, y, z, rotationY,rotationX):
        '''
        Do Y-axis rotation first, then X-axis rotation
        scale and translate to screen coordinates
        '''
        rotX = x * math.cos(rotationY) - y * math.sin(rotationY)
        rotY = x * math.sin(rotationY) + y * math.cos(rotationY)
        
        finalY = rotY * math.cos(rotationX) - z * math.sin(rotationX)
        finalZ = rotY * math.sin(rotationX) + z * math.cos(rotationX)
        
        screenX = app.boardLeft + app.boardWidth/2 + rotX * app.cellSize * app.scale
        screenY = app.boardTop + app.boardHeight/2 + finalY * app.cellSize * app.scale
        
        return (screenX, screenY, finalZ)

# Main method:
def drawGridPlane(app, Projection3D):
    gridColor = rgb(200,200,200) #grey
    
    centerPt = Projection3D.basicProj(app, 0, 0, 0, app.rotationY, app.rotationX)
    endOfX = Projection3D.basicProj(app, app.gridSize, 0, 0, app.rotationY, app.rotationX)
    endOfY = Projection3D.basicProj(app, 0, app.gridSize, 0, app.rotationY, app.rotationX)
    endOfZ = Projection3D.basicProj(app, 0, 0, app.gridSize, app.rotationY, app.rotationX)
    
    drawLine(centerPt[0], centerPt[1], endOfX[0], endOfX[1], fill=gridColor, dashes=True)
    drawLine(centerPt[0], centerPt[1], endOfY[0], endOfY[1], fill=gridColor, dashes=True)
    drawLine(centerPt[0], centerPt[1], endOfZ[0], endOfZ[1], fill=gridColor, dashes=True)
    
    drawLabel(f'x', endOfX[0], endOfX[1], size=20)
    drawLabel(f'y', endOfY[0], endOfY[1], size=20)
    drawLabel(f'z', endOfZ[0], endOfZ[1], size=20)
    
    # Draw grid boundaries along X axis
    for i in range(app.gridSize + 1):
        startPt = Projection3D.basicProj(app, i, 0, 0, app.rotationY, app.rotationX)
        endPt = Projection3D.basicProj(app, i, app.gridSize, 0, app.rotationY, app.rotationX)
        drawLine(startPt[0], startPt[1], endPt[0], endPt[1], fill=gridColor, dashes=True)

    # Draw grid boundaries along Y axis
    for i in range(app.gridSize + 1):
        startPt = Projection3D.basicProj(app, 0, i, 0, app.rotationY, app.rotationX)
        endPt = Projection3D.basicProj(app, app.gridSize, i, 0, app.rotationY, app.rotationX)
        drawLine(startPt[0], startPt[1], endPt[0], endPt[1], fill=gridColor, dashes=True)

# Using the midpoint and direction vector to get the next center point. a 2 level recursion to achieve subdivision, not complete yet. Take reference from chatGpt, and method https://en.wikipedia.org/wiki/Catmull%E2%80%93Clark_subdivision_surface, and modified & rewrite it by myself.
def getNextCenters(cell, pt1, pt2, shift=0.1, currFracLevel=None):
    if currFracLevel is None:
        currFracLevel = cell.fracLevel
        
    # Calculate the midpoint & direction vector between two vertices
    centerX = (pt1[0] + pt2[0])/2
    centerY = (pt1[1] + pt2[1])/2
    centerZ = (pt1[2] + pt2[2])/2
    # Shift the center point outward from cube center
    centerX += shift * (centerX - (cell.x + 0.5))
    centerY += shift * (centerY - (cell.y + 0.5))
    centerZ += shift * (centerZ - (cell.z + 0.5))
    
    # Base case: if fracLevel == 1, return the center point
    if currFracLevel == 1:
        return (centerX, centerY, centerZ)
    
    # Recursive case: subdivide further
    # Get centers between pt1->center and center->pt2
    center1 = getNextCenters(cell, pt1, (centerX, centerY, centerZ), shift, currFracLevel-1)
    center2 = getNextCenters(cell, (centerX, centerY, centerZ), pt2, shift, currFracLevel-1)
    
    # Return the center point at this level
    return (centerX, centerY, centerZ)

def drawCell(app, cell, Projection3D, color='black', isSubd=False):
    # Original vertices
    pts = [
        (cell.x, cell.y, cell.z),         # 0: front bottom left, 0,0,0
        (cell.x+1, cell.y, cell.z),       # 1: front bottom right, 1,0,0
        (cell.x+1, cell.y+1, cell.z),     # 2: back bottom right, 1,1,0
        (cell.x, cell.y+1, cell.z),       # 3: back bottom left, 0,1,0
        (cell.x, cell.y, cell.z+1),       # 4: front top left, 0,0,1
        (cell.x+1, cell.y, cell.z+1),     # 5: front top right, 1,0,1
        (cell.x+1, cell.y+1, cell.z+1),   # 6: back top right, 1,1,1
        (cell.x, cell.y+1, cell.z+1)      # 7: back top left, 0,1,1
    ]
    
    # Calculate all edge centers
    shift = 0.1
    edgeCenters = [
        # Bottom face edges
        getNextCenters(cell, pts[0], pts[1], shift, app.fracLevel),  # Front edge
        getNextCenters(cell, pts[1], pts[2], shift, app.fracLevel),  # Right edge
        getNextCenters(cell, pts[2], pts[3], shift, app.fracLevel),  # Back edge
        getNextCenters(cell, pts[3], pts[0], shift, app.fracLevel),  # Left edge
        
        # Vertical edges
        getNextCenters(cell, pts[0], pts[4], shift, app.fracLevel),  # Front-left
        getNextCenters(cell, pts[1], pts[5], shift, app.fracLevel),  # Front-right
        getNextCenters(cell, pts[2], pts[6], shift, app.fracLevel),  # Back-right
        getNextCenters(cell, pts[3], pts[7], shift, app.fracLevel),  # Back-left
        
        # Top face edges
        getNextCenters(cell, pts[4], pts[5], shift, app.fracLevel),  # Front edge
        getNextCenters(cell, pts[5], pts[6], shift, app.fracLevel),  # Right edge
        getNextCenters(cell, pts[6], pts[7], shift, app.fracLevel),  # Back edge
        getNextCenters(cell, pts[7], pts[4], shift, app.fracLevel)   # Left edge
    ]
    
    # Shifted vertices for subdivision
    s = 0  # shift amount
    ptsShifted = [
        (cell.x+s, cell.y+s, cell.z+s),         # 0
        (cell.x+1-s, cell.y+s, cell.z+s),       # 1
        (cell.x+1-s, cell.y+1-s, cell.z+s),     # 2
        (cell.x+s, cell.y+1-s, cell.z+s),       # 3
        (cell.x+s, cell.y+s, cell.z+1-s),       # 4
        (cell.x+1-s, cell.y+s, cell.z+1-s),     # 5
        (cell.x+1-s, cell.y+1-s, cell.z+1-s),   # 6
        (cell.x+s, cell.y+1-s, cell.z+1-s)      # 7
    ]
    
    # Face centers
    faceCenters = {
        'front': (cell.x + 0.5, cell.y, cell.z + 0.5),
        'back': (cell.x + 0.5, cell.y + 1, cell.z + 0.5),
        'left': (cell.x, cell.y + 0.5, cell.z + 0.5),
        'right': (cell.x + 1, cell.y + 0.5, cell.z + 0.5),
        'top': (cell.x + 0.5, cell.y + 0.5, cell.z + 1),
        'bottom': (cell.x + 0.5, cell.y + 0.5, cell.z)
    }
    
    # Project all points
    projectedPts = [Projection3D.basicProj(app, px, py, pz, app.rotationY, app.rotationX) 
                   for px, py, pz in pts]
    projectedPtsShifted = [Projection3D.basicProj(app, px, py, pz, app.rotationY, app.rotationX) 
                          for px, py, pz in ptsShifted]
    projectedEdgeCenters = [Projection3D.basicProj(app, px, py, pz, app.rotationY, app.rotationX) 
                           for px, py, pz in edgeCenters]
    projectedFaceCenters = [Projection3D.basicProj(app, px, py, pz, app.rotationY, app.rotationX) 
                           for px, py, pz in faceCenters.values()]
    
    # Draw faces first
    faces = [
        ([0,1,5,4], 0),      # Front
        ([1,2,6,5], -20),    # Right
        ([2,3,7,6], -40),    # Back
        ([3,0,4,7], -20),    # Left
        ([4,5,6,7], -10),    # Top
        ([0,1,2,3], -30),    # Bottom
    ]
    
    # Sort faces by depth
    faces.sort(key=lambda f: sum(projectedPts[i][2] for i in f[0])/4)

    # Draw faces with transparency
    opacity = 30
    edgeWidth = 1
    baseColor = rgb(100,100,255) if color == 'blue' else rgb(200,200,200)

    # this part from Claude AI
    for faceIndices, colorAdj in faces:
        faceColor = rgb(max(0, baseColor.red + colorAdj),
                       max(0, baseColor.green + colorAdj),
                       max(0, baseColor.blue + colorAdj))
        
        facePoints = []
        for idx in faceIndices:
            facePoints.extend([projectedPts[idx][0], projectedPts[idx][1]])
        
        drawPolygon(*facePoints, fill=faceColor, opacity=opacity,
                    border='black', borderWidth=edgeWidth)

    # Define vertex to edge connections
    # Key is the vertex index, value is the edge indices that connect to the vertex
    vertexConnections = {
        0: [0, 3, 4],    # Front bottom left
        1: [0, 1, 5],    # Front bottom right
        2: [1, 2, 6],    # Back bottom right
        3: [2, 3, 7],    # Back bottom left
        4: [8, 11, 4],   # Front top left
        5: [8, 9, 5],    # Front top right
        6: [9, 10, 6],   # Back top right
        7: [10, 11, 7]   # Back top left
    }

    if app.showSubd:
        # Draw all connections
        for vertexIdx, edgeIndices in vertexConnections.items():
            for edgeIdx in edgeIndices:
                # Draw line from shifted vertex to edge center
                drawLine(projectedPtsShifted[vertexIdx][0], 
                        projectedPtsShifted[vertexIdx][1],
                        projectedEdgeCenters[edgeIdx][0], 
                        projectedEdgeCenters[edgeIdx][1], 
                        fill='red')
            
    # Draw the points last so they're on top
    for pt in projectedEdgeCenters:
        drawCircle(pt[0], pt[1], 2, fill='black')
    
    for pt in projectedPtsShifted:
        drawCircle(pt[0], pt[1], 2, fill='red')
    
    for pt in projectedFaceCenters:
        drawCircle(pt[0], pt[1], 2, fill='blue')
        
def drawGrid(app):
    drawGridPlane(app, app.projection)
    cubesToDraw = []
    
    for z in range(app.grid.gridSize):
        for y in range(app.grid.gridSize):
            for x in range(app.grid.gridSize):
                cell = Cell(x,y,z, app.fracLevel)
                if app.grid.getCube(cell) is not None:
                    # draw the cell at the center of the cube
                    depth = app.projection.basicProj(app, x+0.5, y+0.5, z+0.5, app.rotationY, app.rotationX)[2]
                    cubesToDraw.append((depth, cell, False))

    # draw the current cell, just for init & indication
    currCell = Cell(app.currentX, app.currentY, app.currentZ, app.fracLevel)
    depth = app.projection.basicProj(app, app.currentX+0.5, app.currentY+0.5, app.currentZ+0.5, app.rotationY, app.rotationX)[2]
    cubesToDraw.append((depth, currCell, True))
    
    # Sort the cubesToDraw by depth
    cubesToDraw.sort(key=lambda x: x[0])
    for _, cell, isSubd in cubesToDraw:
        drawCell(app, cell, app.projection)

# use backtracking to check if the cell is alone or not
def isSubdCell(app):
    # use the face centers, if two face centers are close, show the ori cell without subdivision, turn the isSubd to False
    # if the face centers is alone, show the subdivision cell, turn the isSubd to True
    pass

def importImage(app):
    root = tk.Tk()
    root.withdraw()
    try:
        filePaths = filedialog.askopenfilenames(title='Select Image', 
                                                filetypes=[('Image Files', '*.jpg *.jpeg *.png *.bmp *.gif *.tiff')])
        if filePaths:
            app.image = filePaths[0]
    except Exception as e:
        print(f"Error importing image: {e}")
    finally:
        root.destroy()

def init(app):
    app.projection = Projection3D()
    
    #board szie
    app.boardLeft = 300
    app.boardTop = 100
    app.boardWidth = 800
    app.boardHeight = 600

    # init the current cell
    app.currentX = 0
    app.currentY = 0
    app.currentZ = 0
    app.lastValidX = 0
    app.lastValidY = 0
    app.gridScale = 1
    app.gridSize = 5 * app.gridScale
    app.cellSize = 50
    app.angle = 30
    app.fracLevel = 1
    
    app.rotationY = math.pi/4 # 45 degree
    app.rotationX = math.pi/6 # 30 degree
    app.dragging = False
    app.lastMouseX = 0
    app.lastMouseY = 0
    app.scale = 1
    app.cell = Cell(app.currentX, app.currentY, app.currentZ, app.fracLevel)
    app.grid = Grid3D(app.cellSize, app.gridSize)
    
    #hand gesture starting 
    app.detector = HandGestureDetector()
    app.handCountX = 0
    app.handCountY = 0
    app.handCountZ = 0
    # show centers
    app.showSubd = False
    
    # import the image
    app.image = None
    app.frameImgX = 80
    app.frameImgSize = 250
    app.buttonSize = 30

def onAppStart(app):
    init(app)

def onMousePress(app, mouseX, mouseY):
    app.dragging = True
    app.lastMouseX = mouseX
    app.lastMouseY = mouseY
    if app.frameImgX + app.frameImgSize/2 - app.buttonSize/2 < mouseX < app.frameImgX + app.frameImgSize/2 + app.buttonSize/2 \
    and app.height/2 + app.frameImgSize/2 + app.buttonSize/2 < mouseY < app.height/2 + app.frameImgSize/2 + app.buttonSize*(3/2):
        importImage(app)

def onMouseDrag(app, mouseX, mouseY):
    if app.dragging:
        dx = mouseX - app.lastMouseX
        dy = mouseY - app.lastMouseY
        app.rotationY += dx * 0.01
        app.rotationX += dy * 0.01
        # app.rotationX = app.rotationX % (2 * math.pi)
        if app.rotationX > math.pi/2:
            app.rotationX = math.pi/2
        elif app.rotationX < -math.pi/2:
            app.rotationX = -math.pi/2
        app.lastMouseX = mouseX
        app.lastMouseY = mouseY

def onMouseRelease(app, mouseX, mouseY):
    app.dragging = False

def onKeyPress(app, key):
    # Cube related functions 
    if key == 'space':
        # Create a Cell object for position checking
        currentCell = Cell(app.currentX, app.currentY, app.currentZ, app.fracLevel)
        # Only place cube if position is empty and valid
        if (app.grid.isPosValid(currentCell) and
            app.grid.getCube(currentCell) is None):
            # place the cube
            app.grid.placeCube(currentCell)
            # Wrap around to bottom when reaching top
            if app.currentZ >= app.grid.gridSize - 1:
                app.currentZ = 0
            else:
                app.currentZ += 1
            print("placed cube at:", app.currentX, app.currentY, app.currentZ)
            print(app.grid. board)
    elif key == 'd':
        if (app.grid.isPosValid(currentCell) and
            app.grid.getCube(currentCell) is not None):
            app.grid.board[app.currentZ][app.currentY][app.currentX] = None
            print("removed cube at:", app.currentX, app.currentY, app.currentZ)
    elif key == 'q':
        # app.cellSize = app.cellSize *2
        pass
    elif key == 'e':
        # app.cellSize = app.cellSize //2
        pass

    # Change the grid size
    elif key == 'up':
        if app.gridSize < 32:
            app.gridSize = app.gridSize *2
            print(f"gridSize to: {app.gridSize}")
    elif key == 'down':
        if app.gridSize > 1:
            app.gridSize = app.gridSize //2
            print(f"gridSize to: {app.gridSize}")

    # Reset the game
    elif key == 'r':
        onAppStart(app)

    # Show subdivision
    elif key == 's':
        app.showSubd = not app.showSubd
    elif key == ']':
        if app.fracLevel == 3:
            app.fracLevel = 3
        else:
            app.fracLevel += 1
        print(f"fracLevel to: {app.fracLevel}") 
    elif key == '[':
        if app.fracLevel == 1:
            app.fracLevel = 1
        else:
            app.fracLevel -= 1
        print(f"fracLevel to: {app.fracLevel}")  
    elif key == 'left':
        app.scale -= 0.1
    elif key == 'right':
        app.scale += 0.1

def onStep(app):
    # it will be x & y
    app.handCountX, app.handCountY, app.handCountZ = app.detector.detectGesture()
    if app.handCountZ:
        # Use the last valid X/Y positions when moving in Z
        app.currentX = app.lastValidX
        app.currentY = app.lastValidY
        mappedZ = int(app.handCountZ * (app.gridSize))
        app.currentZ = app.grid.gridSize - mappedZ
    elif app.handCountX is not None and app.handCountY is not None:
        # Update X/Y positions and store them as last valid positions
        mappedX = int(app.handCountX * (app.gridSize-1))
        mappedY = int(app.handCountY * (app.gridSize-1))
        app.currentX = mappedX
        app.currentY = mappedY
        app.lastValidX = mappedX
        app.lastValidY = mappedY

def redrawAll(app):
    drawLabel('Build Game',app.width/2, 20, size=24)
    if app.handCountX or app.handCountY:
        drawLabel(f'count: {app.handCountX}, {app.handCountY}', app.width/2, app.height - 20, size = 20)
    else:
        drawLabel('Hand not detected, Move your hand to move the cube', app.width/2, app.height - 20, size = 20)
    drawLabel('SPACE to place cube, R to reset, C to show subdivision', app.width/2, 60, size=20)
    drawGrid(app)
    drawLabel(f'current: {app.currentX}, {app.currentY}, {app.currentZ}', app.width/2, 100, size=20)
    drawLabel(f'fracLevel: {app.fracLevel}, use [ ] to change', app.width/2, 140, size=20)
    # import the image 
    if app.image is not None:
        drawImage(app.image, app.frameImgX, app.height/2-app.frameImgSize/2, width=app.frameImgSize, height=app.frameImgSize)
    drawRect(app.frameImgX, app.height/2-app.frameImgSize/2, app.frameImgSize, app.frameImgSize, fill=None, border='black')
    drawImage('import.png', app.frameImgX + app.frameImgSize/2 - app.buttonSize/2, app.height/2 + app.frameImgSize/2 + app.buttonSize, width=app.buttonSize, height=app.buttonSize)

def main():
    runApp(width=1200, height=800)

main()