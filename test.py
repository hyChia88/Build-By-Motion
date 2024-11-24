# Demo
from matplotlib.pyplot import grid
from cmu_graphics import *
import math
import cv2
import mediapipe as mp

class Grid3D:
    def __init__(self, cellSize, gridSize):
        self.cellSize = cellSize
        self.gridSize = gridSize
        # Initialize size attributes for easier access
        self.xSize = self.ySize = self.zSize = gridSize
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
    
    def placeCube(self, cell):
        if self.isPosValid(cell):
            self.board[cell.z][cell.y][cell.x] = cell  # Fixed index order
            return True
        return False

    def get_cube(self, x, y, z):
        if (0 <= x < self.gridSize and 
            0 <= y < self.gridSize and 
            0 <= z < self.gridSize):
            return self.board[z][y][x]  # Fixed index order
        return None

    def drawGridPlane(self, app, Projection3D):
        gridColor = rgb(200,200,200) #grey
        
        centerPt = Projection3D.basicProj(app, 0, 0, 0, app.rotationY, app.rotationX)
        endOfX = Projection3D.basicProj(app, app.gridSize+1, 0, 0, app.rotationY, app.rotationX)
        endOfY = Projection3D.basicProj(app, 0, app.gridSize+1, 0, app.rotationY, app.rotationX)
        
        drawLine(centerPt[0], centerPt[1], endOfX[0], endOfX[1], fill=gridColor, dashes=True)
        drawLine(centerPt[0], centerPt[1], endOfY[0], endOfY[1], fill=gridColor, dashes=True)
        
    def drawCell(self, app, cell, proj):
        center = proj.basicProj(app, cell.x+0.5, cell.y+0.5, cell.z+0.5, 
                              app.rotationY, app.rotationX)
        drawCircle(center[0], center[1], cell.cellSize/2, fill='black')

    def moveCell(self, cell, direction):
        newX, newY, newZ = cell.x, cell.y, cell.z
        
        if direction == 'up':
            newZ += 1
        elif direction == 'down':
            newZ -= 1
        elif direction == 'left':
            newX -= 1
        elif direction == 'right':
            newX += 1
            
        newCell = Cell(newX, newY, newZ, cell.fracLevel)
        if self.isPosValid(newCell):
            cell.x, cell.y, cell.z = newX, newY, newZ
            return True
        return False

class Cell:
    def __init__(self, x, y, z, fracLevel):
        self.cellSize = 10
        self.fracLevel = fracLevel
        self.x = x
        self.y = y
        self.z = z
        
    def fracCell(self):
        self.x = self.x * 2
        self.y = self.y * 2
        self.z = self.z * 2

class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.prevX = None
        self.prevY = None  # Added missing prevY initialization
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
            
            # Use index finger tip (landmark 8) instead of landmark 0
            index_x = int(hand.landmark[8].x * w)
            index_y = int(hand.landmark[8].y * h)
            cv2.circle(frame, (index_x, index_y), 10, (0, 0, 255), -1)
            
            currentHandX = hand.landmark[8].x
            currentHandY = hand.landmark[8].y
            if self.prevX is not None:
                movementX = currentHandX - self.prevX
                if abs(movementX) > self.swipeThreshold:
                    self.counter += -1 if movementX > 0 else 1
                
            self.prevX = currentHandX
            self.prevY = currentHandY
        else:
            self.prevX = None
            self.prevY = None
            
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

    def basicProj(self, app, x, y, z, rotationY, rotationX):
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

def drawGrid(app):
    cubesToDraw = []
    
    for z in range(app.grid.zSize):
        for y in range(app.grid.ySize):
            for x in range(app.grid.xSize):
                cube = app.grid.get_cube(x, y, z)
                if cube is not None:
                    depth = app.projection.basicProj(app, x+0.5, y+0.5, z+0.5, 
                                                  app.rotationY, app.rotationX)[2]
                    cubesToDraw.append((depth, cube))
                    
    cubesToDraw.sort(key=lambda x: x[0])
    for _, cube in cubesToDraw:
        app.grid.drawCell(app, cube, app.projection)

def init(app):
    app.projection = Projection3D()
    app.cx = 200
    app.cy = 200
    
    app.gridSize = 4
    app.cellSize = 80
    app.angle = 30
    
    app.boardLeft = 200
    app.boardTop = 100
    app.boardWidth = 600
    app.boardHeight = 600
    
    app.grid = Grid3D(app.cellSize, app.gridSize)
    
    app.currX = 0
    app.currY = 0
    app.currZ = 0
    app.currFracLevel = 1
    app.currCell = Cell(app.currX, app.currY, app.currZ, app.currFracLevel)
    
    app.rotationY = math.pi/4  # 45 degree
    app.rotationX = math.pi/6  # 30 degree
    
    app.detector = HandGestureDetector()
    app.handCount = 0
            
def onAppStart(app):
    init(app)

def onKeyPress(app, key):
    if key == 'r':
        init(app)
    elif key == 'q':
        app.detector.cleanup()
    elif key == 'space':
        app.grid.placeCube(app.currCell)
    elif key == 'c':
        app.currCell.fracCell()
    elif key in ['up', 'down', 'left', 'right']:
        app.grid.moveCell(app.currCell, key)
    
def onStep(app):
    app.handCount = app.detector.detectGesture()
    newCx = app.cx + app.handCount*5
    newCy = app.cy + app.handCount*5
    
    if app.cellSize/2 <= newCx <= app.width-app.cellSize:
        app.cx = newCx
    if app.cellSize/2 <= newCy <= app.height-app.cellSize:
        app.cy = newCy

def redrawAll(app):
    app.grid.drawGridPlane(app, app.projection)
    drawLabel(f'count: {app.handCount}', app.width/2, app.height/2, size=30)
    drawGrid(app)
    
    drawLabel('3D Grid Game', app.width/2, 30, size=20)
    
    # Draw test cube
    vertices = [
        (1,1,1), (2,1,1), (1,2,1), (2,2,1),  # bottom face
        (1,1,2), (2,1,2), (1,2,2), (2,2,2)   # top face
    ]
    
    projectedPoints = []
    for x,y,z in vertices:
        px, py, _ = app.projection.basicProj(app, x, y, z, app.rotationY, app.rotationX)
        projectedPoints.append((px, py))
        drawCircle(px, py, 5, fill='red')
    
    # Draw cube faces
    drawPolygon(projectedPoints[0][0], projectedPoints[0][1],
                projectedPoints[1][0], projectedPoints[1][1],
                projectedPoints[3][0], projectedPoints[3][1],
                projectedPoints[2][0], projectedPoints[2][1],
                fill=None, border='blue')
    
    drawPolygon(projectedPoints[4][0], projectedPoints[4][1],
                projectedPoints[5][0], projectedPoints[5][1],
                projectedPoints[7][0], projectedPoints[7][1],
                projectedPoints[6][0], projectedPoints[6][1],
                fill=None, border='blue')
    
    for i in range(4):
        drawLine(projectedPoints[i][0], projectedPoints[i][1],
                projectedPoints[i+4][0], projectedPoints[i+4][1],
                fill='blue')

def main():
    runApp()

main()