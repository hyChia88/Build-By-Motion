# Demo
from cmu_graphics import *
import math
import cv2
import numpy as np
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
            # for now assume cell is 1,1,1, just check center of cell the exact only
            if (0<=cell.x<self.girdSize and
                0<=cell.y<self.girdSize and
                0<=cell.z<self.girdSize):
                return True
        return False
    
    def placeCube(self,cell):
        if self.isPosValid(self,cell):
            # fill the board will cell Data
            self.board[cell.x][cell.y][cell.z] = True
            return True
        return False

class Cell:
    def __init__(self,x,y,z,fracLevel):
        self.cellSize = 10
        self.fracLevel = fracLevel
        self.x = x
        self.y = y
        self.z = z
        
    def fracCell(self):
        self.x = self.x *2
        self.y = self.y *2
        self.z = self.z *2
        
# def drawCell(app, x,y,z, isPreview = True):
def drawCell(app):
    # Hardcode for now
    #offsets
    dx = app.cellSize * math.cos(math.radians(app.angle))
    dy = app.cellSize * math.sin(math.radians(app.angle))
    
    # Front face (pink)
    drawPolygon(
        app.cx, app.cy,
        app.cx + dx, app.cy - dy,
        app.cx + dx, app.cy - dy - app.cellSize,
        app.cx, app.cy - app.cellSize,
        fill='pink'
    )
    
    # Right face (lightBlue)
    drawPolygon(
        app.cx + dx, app.cy - dy,
        app.cx + 2*dx, app.cy,
        app.cx + 2*dx, app.cy - app.cellSize,
        app.cx + dx, app.cy - dy - app.cellSize,
        fill='lightBlue'
    )
    
    # Top face (lightGreen)
    drawPolygon(
        app.cx, app.cy - app.cellSize,
        app.cx + dx, app.cy - dy - app.cellSize,
        app.cx + 2*dx, app.cy - app.cellSize,
        app.cx + dx, app.cy - app.cellSize + dy,
        fill='lightGreen'
    )
    
    # Front face (boarder)
    drawPolygon(
        app.cx, app.cy, 
        app.cx, app.cy - app.cellSize,
        app.cx + dx, app.cy - app.cellSize + dy,
        app.cx + dx, app.cy + app.cellSize - dy, 
        fill=None, border = 'black'
    )
    
    drawPolygon(
        app.cx + dx, app.cy - app.cellSize + dy,
        app.cx + dx, app.cy + app.cellSize - dy, 
        app.cx + dx*2, app.cy,
        app.cx + dx*2, app.cy - app.cellSize,
        fill=None, border = 'black'
    )

class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.prev_x = None
        self.counter = 0
        self.swipe_threshold = 0.05
        self.cap = cv2.VideoCapture(0)

    def detect_gesture(self):
        ret, frame = self.cap.read()
        if not ret:
            return self.counter
            
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            h, w = frame.shape[:2]
            
            # Draw red circle on index finger
            index_x = int(hand.landmark[8].x * w)
            index_y = int(hand.landmark[8].y * h)
            cv2.circle(frame, (index_x, index_y), 10, (0, 0, 255), -1)
            
            current_x = hand.landmark[8].x
            if self.prev_x is not None:
                movement = current_x - self.prev_x
                if abs(movement) > self.swipe_threshold:
                    self.counter += -1 if movement > 0 else 1
            self.prev_x = current_x
        else:
            self.prev_x = None
            
        cv2.imshow('Hand Gesture Counter', frame)
        cv2.waitKey(1)
        return self.counter

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

def onAppStart(app):
    #hardcode
    app.cx = 200
    app.cy = 200
    
    app.gridSize = 4
    app.cellSize = 80
    app.angle = 30
    app.grid = Grid3D(app.cellSize, app.gridSize)
    
    app.currX = 0
    app.currY = 0
    app.currZ = 0
    app.currFracLevel = 1
    app.currCell = Cell(app.currX, app.currY, app.currZ, app.currFracLevel)
    
    #hand gesture starting 
    app.detector = HandGestureDetector()
    app.handCount = 0
    
def onStep(app):
    app.handCount = app.detector.detect_gesture()
    newCx = app.cx + app.handCount*5
    newCy = app.cy + app.handCount*5
    
    if app.cellSize/2 <= newCx <= app.width-app.cellSize:
        app.cx = newCx
    if app.cellSize/2 <= newCy <= app.height-app.cellSize:
        app.cy = newCy

def redrawAll(app):
    drawCell(app)
    drawLabel(f'count: {app.handCount}', app.width/2, app.height/2, size = 30)

def main():
    runApp()

main()