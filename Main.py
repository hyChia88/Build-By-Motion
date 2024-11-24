# Demo
# from matplotlib.pyplot import grid
from cmu_graphics import *
import math
import cv2
# import numpy as np
import mediapipe as mp

class Grid3D:
    def __init__(self, cellSize, gridSize):
        self.cellSize = cellSize
        self.gridSize = gridSize
        # it will be a cube
        self.board = [[[None for x in range(self.gridSize)]
                       for y in range(self.gridSize)]
                       for z in range(self.gridSize)]
    
    def isPosValid(self, cell):
        if isinstance(cell, Cell):
            if (0<=cell.x<self.gridSize and
                0<=cell.y<self.gridSize and
                0<=cell.z<self.gridSize):
                return True
        return False
    
    def placeCube(self,cell):
        if self.isPosValid(cell):
            # the order of x,y,z is different from the order of the board
            self.board[cell.z][cell.y][cell.x] = cell
            return True
        return False
    
    def getCube(self, cell):
        if self.isPosValid(self, cell):
            # the order of x,y,z is different from the order of the board
            return self.board[cell.z][cell.y][cell.x]
        return None
        
class Cell:
    def __init__(self,x,y,z,fracLevel):
        self.cellSize = 10
        self.fracLevel = fracLevel
        #  x,y,z here is position
        self.x = x
        self.y = y
        self.z = z
        
    def fracCell(self):
        self.x = self.x *2
        self.y = self.y *2
        self.z = self.z *2

class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.prevX = None
        self.prevY = None
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
            print("finger place:",indexX, indexY)
            cv2.circle(frame, (indexX, indexY), 10, (0, 0, 255), -1)

            if self.prevX is not None and self.prevY is not None:
                movementX = currentHandX - self.prevX
                movementY = currentHandY - self.prevY
                if abs(movementX) > self.swipeThreshold:
                    self.counter += -1 if movementX > 0 else 1
                if abs(movementY) > self.swipeThreshold:
                    self.counter += -1 if movementY > 0 else 1
                
            self.prevX = currentHandX
            self.prevY = currentHandY
        else:
            self.prevX = None
            self.prevY = None
            
        # cv2.imshow('Hand Gesture Counter', frame)
        # something like FPS 帧数
        cv2.waitKey(1)
        return self.counter

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

class Projection3D:
    def __init__(self):
        self.near = 0.1
        self.far = 1000.0
        self.fov = 60
        self.aspect = 16/9

    def basicProj(self, app, x,y,z,rotationY,rotationX):
        '''
        Do Y-axis rotation first, then X-axis rotation
        scale and translate to screen coordinates
        '''
        rotX = x * math.cos(rotationY) - y * math.sin(rotationY)
        rotY = x * math.sin(rotationY) + y * math.cos(rotationY)
        
        finalY = rotY * math.cos(rotationX) - z * math.sin(rotationX)
        finalZ = rotY * math.sin(rotationX) + z * math.cos(rotationX)
        
        scale = 1.0 / (finalZ + 5)
        screenX = app.boardLeft + app.boardWidth/2 + rotX * app.cellSize
        screenY = app.boardTop + app.boardHeight/2 + finalY * app.cellSize
        
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
    
    drawLabel(f'({centerPt[0]:.1f}, {centerPt[1]:.1f})', centerPt[0], centerPt[1], size=20)
    
def drawCell(app, cell, proj):
    vertices = [
        (cell.x, cell.y, cell.z),         # 0: front bottom left
        (cell.x+1, cell.y, cell.z),       # 1: front bottom right
        (cell.x+1, cell.y+1, cell.z),     # 2: back bottom right
        (cell.x, cell.y+1, cell.z),       # 3: back bottom left
        (cell.x, cell.y, cell.z+1),       # 4: front top left
        (cell.x+1, cell.y, cell.z+1),     # 5: front top right
        (cell.x+1, cell.y+1, cell.z+1),   # 6: back top right
        (cell.x, cell.y+1, cell.z+1)      # 7: back top left
    ]
    
    projectedVertices = [proj.basicProj(app, *v, app.rotationY, app.rotationX) for v in vertices]
    
    faces = [
        ([0,1,5,4], 0),      # Front
        ([1,2,6,5], -20),    # Right
        ([2,3,7,6], -40),    # Back
        ([3,0,4,7], -20),    # Left
        ([4,5,6,7], -10),    # Top
        ([0,1,2,3], -30),    # Bottom
    ]
    
    faces.sort(key=lambda f: sum(projectedVertices[i][2] for i in f[0])/4)
    
    opacity = 70
    edgeWidth = 4
    baseColor = rgb(100,100,255)
    
    for faceIndices, colorAdj in faces:
        faceColor = rgb(max(0, baseColor.red + colorAdj),
                       max(0, baseColor.green + colorAdj),
                       max(0, baseColor.blue + colorAdj))
        
        points = []
        for idx in faceIndices:
            points.extend([projectedVertices[idx][0], projectedVertices[idx][1]])
        
        drawPolygon(*points, fill=faceColor, opacity=opacity,
                    border='black', borderWidth=edgeWidth)
    
def drawGrid(app):
    drawGridPlane(app, app.projection)
    cubesToDraw = []
    
    for z in range(app.grid.gridSize):
        for y in range(app.grid.gridSize):
            for x in range(app.grid.gridSize):
                cell = Cell(x,y,z, app.currFracLevel)
                if app.grid.getCube(cell) is not None:
                    # draw the cell at the center of the cube
                    depth = app.projection.basicProj(app, x+0.5, y+0.5, z+0.5, app.rotationY, app.rotationX)[2]
                    cubesToDraw.append((depth, cell))

    # draw the current cell, just for init & indication
    currCell = Cell(app.currentX, app.currentY, app.currentZ, app.currFracLevel)
    depth = app.projection.basicProj(app, app.currentX+0.5, app.currentY+0.5, app.currentZ+0.5, app.rotationY, app.rotationX)[2]
    cubesToDraw.append((depth, currCell))
    
    # Sort the cubesToDraw by depth
    cubesToDraw.sort(key=lambda x: x[0])
    for _, cell, isPreview in cubesToDraw:
        drawCell(app, cell, app.projection)

def init(app):
    app.projection = Projection3D()
    app.cx = app.width/2
    app.cy = app.height/2
    
    app.gridSize = 4
    app.cellSize = 80
    app.angle = 30
    app.currFracLevel = 1
    
    #board szie
    app.boardLeft = 200
    app.boardTop = 100
    app.boardWidth = 600
    app.boardHeight = 600
    
    app.grid = Grid3D(app.cellSize, app.gridSize)
    
    app.rotationY = math.pi/4 # 45 degree
    app.rotationX = math.pi/6 # 30 degree
    
    # init the current cell
    app.currentX = 0
    app.currentY = 0
    app.currentZ = 0
    
    #hand gesture starting 
    app.detector = HandGestureDetector()
    app.handCount = 0
            
def onAppStart(app):
    init(app)
    
def getViewBasedMovement(app, key):
    # get the rotation of the camera 0-360 degree
    rotation = app.rotationY % (2 * math.pi)
    
    # movement in dict:
    movement = {
        'up': (0, 1),
        'down': (0, -1),
        'left': (-1, 0),
        'right': (1, 0)
    }
    return movement.get(key, (0,0))

def onKeyPress(app, key):
    if key in ['up', 'down', 'left', 'right']:
        dx, dy = getViewBasedMovement(app, key)
        newX = app.currentX + dx
        newY = app.currentY + dy
        
        if (0 <= newX < app.grid.xSize and 
            0 <= newY < app.grid.ySize):
            app.currentX = newX
            app.currentY = newY
    
    elif key == 'space':
        # Only place cube if position is empty and valid
        if (app.grid.isPosValid(app.currentX, app.currentY, app.currentZ) and
            app.grid.getCube(app.currentX, app.currentY, app.currentZ) is None):
            # place the cube
            app.grid.placeCube(app.currentX, app.currentY, app.currentZ, True)
            # move to the next level
            app.currentZ += 1
            # Wrap around to bottom when reaching top
            if app.currentZ >= app.grid.zSize:
                app.currentZ = 0
            print("placed cube at:", app.currentX, app.currentY, app.currentZ)
    elif key == 'r':
        onAppStart(app)

def onStep(app):
    app.handCount = app.detector.detectGesture()
    # newCx = app.cx + app.handCount*2
    # newCy = app.cy + app.handCount*2
    # # check if the new position is within the screen boundaries
    # if app.cellSize/2 <= newCx <= app.width-app.cellSize:
    #     app.cx = newCx
    # if app.cellSize/2 <= newCy <= app.height-app.cellSize:
    #     app.cy = newCy

def redrawAll(app):
    drawLabel(f'count: {app.handCount}', app.width/2, app.height/2, size = 30)
    drawLabel('Use arrow keys to move, SPACE to place cube, R to reset', 
             app.width/2, 60, size=20)
    drawGrid(app)
    
    # Instructions
    drawLabel('3D Grid Game', app.width/2, 30, size=20)

def main():
    runApp(width=1200, height=800)

main()