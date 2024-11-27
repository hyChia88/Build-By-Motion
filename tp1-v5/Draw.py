'''
This is the drawing page that allows the user to draw on a grid
'''
from cmu_graphics import *
import cv2
import mediapipe as mp

class PatternSubdivision:
    def __init__(self, initial_pattern):
        self.fractalLevel = 0
        self.pattern = initial_pattern
        self.maxLevel = 3
        
    def subdivide(self):
        if self.fractalLevel >= self.maxLevel:  # Limit max subdivision level
            self.fractalLevel = self.maxLevel
            return
        
        oldSize = len(self.pattern)
        newSize = oldSize * 2
        # create a new pattern (2D list) with the new size 
        newPattern = [[0 for _ in range(newSize)] for _ in range(newSize)]
        
        for y in range(newSize):
            for x in range(newSize):
                origX = x // 2
                origY = y // 2
                
                if x % 2 == 0 and y % 2 == 0:
                    newPattern[y][x] = self.pattern[origY][origX]
                else:
                    isEdge = x % 2 != 0 or y % 2 != 0
                    if isEdge:
                        newPattern[y][x] = int(self.getNeighborAverage(origX, origY))
                    else:
                        newPattern[y][x] = self.pattern[origY][origX]
        
        self.pattern = newPattern
        self.fractalLevel += 1
        
    def getNeighborAverage(self, x, y):
        count = 0
        total = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                newX, newY = x + dx, y + dy
                if 0 <= newX < len(self.pattern) and 0 <= newY < len(self.pattern):
                    total += self.pattern[newY][newX]
                    count += 1
        return total / count > 0.5

# Mainly reference from https://youtu.be/RRBXVu5UE-U?si=FTBWxNPHmmu-KmW6 (same as Build.py)
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

            self.prevX = currentHandX
            self.prevY = currentHandY
        else:
            self.prevX = None
            self.prevY = None

        cv2.imshow('Hand Gesture Counter', frame)
        # something like FPS 帧数
        cv2.waitKey(1)
        if self.prevX is not None and self.prevY is not None:
            if self.prevX > 1:
                self.prevX = 1
            elif self.prevX < 0:
                self.prevX = 0
            elif self.prevY > 1:
                self.prevY = 1
            elif self.prevY < 0:
                self.prevY = 0
        return self.prevX, self.prevY

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

def onAppStart(app):
    # Grid drawing settings
    app.gridSize = 8
    app.margin = 50
    app.cellSize = (min(app.width, app.height) - 2*app.margin) / app.gridSize
    app.grid = [[0 for _ in range(app.gridSize)] for _ in range(app.gridSize)]
    
    # Mode settings
    app.isDrawMode = True
    app.isShowSubd = False
    
    # Subdivision instance (will be created when needed)
    app.subdivision = None
    
    app.detector = HandGestureDetector()
    app.handCountX = app.detector.prevX if app.detector.prevX is not None else 0
    app.handCountY = app.detector.prevY if app.detector.prevY is not None else 0
    app.isHandDraw = False
    
def onStep(app):
    handX, handY = app.detector.detectGesture()
    onHandDraw(app, app.detector, handX, handY)

def getCell(app, mouseX, mouseY):
    gridLeft = app.margin
    gridTop = app.margin
    gridRight = gridLeft + app.gridSize * app.cellSize
    gridBottom = gridTop + app.gridSize * app.cellSize
    
    if not (gridLeft <= mouseX <= gridRight and
            gridTop <= mouseY <= gridBottom):
        return None
    
    col = int((mouseX - app.margin) // app.cellSize)
    row = int((mouseY - app.margin) // app.cellSize)
    
    if 0 <= row < app.gridSize and 0 <= col < app.gridSize:
        return (row, col)
    return None

def drawGrid(app):
    # Draw background
    drawRect(0, 0, app.width, app.height, fill='white')
    
    # Draw grid lines
    for i in range(app.gridSize + 1):
        lineX = app.margin + i * app.cellSize
        drawLine(lineX, app.margin, 
                lineX, app.margin + app.gridSize * app.cellSize,
                lineWidth=2)
        
        lineY = app.margin + i * app.cellSize
        drawLine(app.margin, lineY,
                app.margin + app.gridSize * app.cellSize, lineY,
                lineWidth=2)
    
    # Draw filled cells
    for row in range(app.gridSize):
        for col in range(app.gridSize):
            if app.grid[row][col] == 1:
                cellLeft = app.margin + col * app.cellSize
                cellTop = app.margin + row * app.cellSize
                drawRect(cellLeft, cellTop, 
                        app.cellSize, app.cellSize,
                        fill='black')

def drawSubdivision(app):
    if app.subdivision:
        # Draw background
        # drawRect(0, 0, app.width, app.height, fill='grey')
        
        # Calculate cell size for subdivided pattern
        cell_size = (min(app.width, app.height) - 2*app.margin) / len(app.subdivision.pattern)
        
        # Draw subdivided pattern
        for y in range(len(app.subdivision.pattern)):
            for x in range(len(app.subdivision.pattern)):
                if app.subdivision.pattern[y][x]:
                    drawRect(app.margin + x * cell_size, 
                           app.margin + y * cell_size,
                           cell_size, cell_size,
                           fill='black')

def redrawAll(app):
    if app.isShowSubd:
        drawSubdivision(app)
    else:
        drawGrid(app)
        if not app.isShowSubd and app.detector.prevX is not None and app.detector.prevY is not None:
            # Map normalized coordinates (0-1) to screen coordinates with grid offset
            screenX = app.margin + (app.detector.prevX * app.gridSize * app.cellSize)
            screenY = app.margin + (app.detector.prevY * app.gridSize * app.cellSize)
            if not app.isShowSubd and app.isHandDraw:
                cell = getCell(app, screenX, screenY)
                if cell:
                    row, col = cell
                    app.grid[row][col] = 1
            drawCircle(screenX, screenY, 10, fill='red', opacity=50)
            
    # Draw mode and level info
    drawLabel(f'Draw Game, mode: {"Drawing" if not app.isShowSubd else "Subdivision"}',
             app.width/2, 20, size=24)
    if app.isShowSubd and app.subdivision:
        drawLabel(f'Fractal Level: {app.subdivision.fractalLevel}',
                 app.width/2, 40, size=16)
    
    # Draw instructions
    if app.isShowSubd:
        drawLabel('Press UP/DOWN for subdivision levels, S to return to drawing',
                 app.width/2, app.height - 20, size=14)
    else:
        drawLabel('Click and drag to fill cells, press S to show subdivision',
                 app.width/2, app.height - 20, size=14)
        drawLabel('Press H to toggle hand drawing', app.width/2, app.height - 40, size=14)

def onMouseDrag(app, mouseX, mouseY):
    if not app.isShowSubd:
        cell = getCell(app, mouseX, mouseY)
        if cell:
            row, col = cell
            app.grid[row][col] = 1

def onHandDraw(app, detector, handX, handY):
    if not app.isShowSubd and detector.prevX is not None and detector.prevY is not None:
        
        # map the hand gesture to the grid
        mappedX = int(handX * app.gridSize)
        mappedY = int(handY * app.gridSize)
        print(mappedX, mappedY, app.gridSize)
        
        cell = getCell(app, mappedX, mappedY)
        if cell:
            row, col = cell
            app.grid[row][col] = 1

def onKeyPress(app, key):
    if key == 's':  # Toggle between draw and subdivision modes
        app.isShowSubd = not app.isShowSubd
        if app.isShowSubd and app.subdivision is None:
            # Initialize subdivision with current grid
            app.subdivision = PatternSubdivision(app.grid)
    
    elif app.isShowSubd:
        if key == 'up':
            app.subdivision.subdivide()
        elif key == 'down' and app.subdivision.fractalLevel > 0:
            # Reset subdivision but keep initial pattern
            app.subdivision = PatternSubdivision(app.grid)
    
    #reset the grid
    elif key == 'r':
        app.grid = [[0 for _ in range(app.gridSize)] for _ in range(app.gridSize)]
        if app.subdivision:
            app.subdivision.fractalLevel = 0
            app.subdivision.pattern = app.grid
    
    elif key == 'h':
        app.isHandDraw = not app.isHandDraw

def main():
    runApp(width=1200, height=800)

main()