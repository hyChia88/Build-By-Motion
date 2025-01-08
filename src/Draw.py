'''
draw page
This is the drawing page that allows the user to draw on a grid
'''
from cmu_graphics import *
import cv2
import mediapipe as mp
from Build import *

# For future use, wont implement this time.
class PatternSubdivision:
    def __init__(self, initial_pattern):
        self.fractalLevel = 0
        self.pattern = initial_pattern
        self.maxLevel = 1
    
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
                    isEdge = x % 2 != 0 or y % 2 != 0 # Checks if either x or y is odd
                    '''
                    for example:
                    (0,0) (0,1) (0,2) (0,3)  # x=0, y=0,1,2,3
                    (1,0) (1,1) (1,2) (1,3)  # x=1, y=0,1,2,3
                    (2,0) (2,1) (2,2) (2,3)  # x=2, y=0,1,2,3
                    (3,0) (3,1) (3,2) (3,3)  # x=3, y=0,1,2,3
                    
                    RESULT:
                    O E O E
                    E E E E
                    O E O E
                    E E E E
                    '''
                    if isEdge:
                        # Case 1: For edge positions (where x or y is odd), Calculates average of neighboring values and converts to 0 or 1
                        newPattern[y][x] = int(self.getNeighborAverage(origX, origY))
                    else:
                        # Case 2: For original positions (where both x and y are even),Directly copies the value from original pattern
                        newPattern[y][x] = self.pattern[origY][origX]
        
        self.pattern = newPattern
        self.fractalLevel += 1
    '''
    Calculate the average of the 8 neighbors, with code inspiration from ChatGPT. If the threshold is 0.5, then it is 1, otherwise 0
    '''
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

'''
Mainly reference from https://youtu.be/RRBXVu5UE-U?si=FTBWxNPHmmu-KmW6 (same as Build.py)
'''
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

def resizeGrid(app, newSize):
    if newSize < 2:
        newSize = 2
    elif newSize > 6:
        newSize = 6
    app.drawDGridSize = newSize
    app.drawCellSize = app.gridActualSize / app.drawDGridSize
    # create a new pattern (2D list) with the new size, reset the grid when resizing
    app.drawDGrid = [[0 for _ in range(newSize)] for _ in range(newSize)]

def draw_onScreenActivate(app):
    drawInit(app)
    
def drawInit(app):
    # Grid drawing settings
    app.gridActualSize = 400
    app.drawDGridSize = 4
    app.gridLeft = app.width/2 - app.gridActualSize/2
    app.gridTop = app.height/2 - app.gridActualSize/2
    app.drawCellSize = app.gridActualSize / app.drawDGridSize
    app.cellSize = app.gridActualSize / app.drawDGridSize
    app.drawDGrid = [[0 for _ in range(app.drawDGridSize)] for _ in range(app.drawDGridSize)]
    
    # Mode settings
    app.isDrawMode = True
    app.isShowSubd = False
    
    # Subdivision instance (will be created when needed)
    app.subdivision = None
    
    app.drawDetector = HandGestureDetector()
    app.isHandDraw = False
    app.drawHint = None

def draw_onStep(app):
    handX, handY = app.drawDetector.detectGesture()
    onHandDraw(app, app.drawDetector, handX, handY)

def getDrawnCell(app, mouseX, mouseY):
    gridRight = app.gridLeft + app.drawDGridSize * app.drawCellSize
    gridBottom = app.gridTop + app.drawDGridSize * app.drawCellSize
    
    if not (app.gridLeft <= mouseX <= gridRight and
            app.gridTop <= mouseY <= gridBottom):
        return None
    
    col = int((mouseX - app.gridLeft) // app.drawCellSize)
    row = int((mouseY - app.gridTop) // app.drawCellSize)
    
    if 0 <= row < app.drawDGridSize and 0 <= col < app.drawDGridSize:
        return (row, col)
    return None

def drawDGridMethod(app):
    # Draw background
    drawRect(0, 0, app.width, app.height, fill='white')

    # Draw grid lines
    for i in range(app.drawDGridSize + 1):
        lineX = app.gridLeft + i * app.drawCellSize
        drawLine(lineX, app.gridTop, lineX, app.gridTop + app.drawDGridSize * app.drawCellSize,
                lineWidth=2)
        
        lineY = app.gridTop + i * app.drawCellSize
        drawLine(app.gridLeft, lineY, app.gridLeft + app.drawDGridSize * app.drawCellSize, lineY,
                lineWidth=2)

    for row in range(app.drawDGridSize):
        for col in range(app.drawDGridSize):
            if app.drawDGrid[row][col] == 1:
                cellLeft = app.gridLeft + col * app.drawCellSize
                cellTop = app.gridTop + row * app.drawCellSize
                drawRect(cellLeft, cellTop, app.drawCellSize, app.drawCellSize, fill='black')

def drawSubdivision(app, pattern):
    if app.subdivision:
        # Calculate cell size for subdivided pattern
        cell_size = (min(app.width, app.height) - 2*app.gridLeft) / len(pattern)
        
        # Draw subdivided pattern
        for y in range(len(pattern)):
            for x in range(len(pattern)):
                if pattern[y][x]:
                    drawRect(app.gridLeft + x * cell_size, app.gridTop + y * cell_size, cell_size, cell_size, fill='black')

def draw_redrawAll(app):
    initialY = 20
    if app.isShowSubd:
        drawSubdivision(app, app.subdivision.pattern)
    else:
        drawDGridMethod(app)
        if not app.isShowSubd and app.drawDetector.prevX is not None and app.drawDetector.prevY is not None:
            # Map normalized coordinates (0-1) to screen coordinates with grid offset
            screenX = app.gridLeft + (app.drawDetector.prevX * app.drawDGridSize * app.drawCellSize)
            screenY = app.gridTop + (app.drawDetector.prevY * app.drawDGridSize * app.drawCellSize)
            drawCircle(screenX, screenY, 10, fill='red', opacity=50)
            
    # Draw mode and level info
    drawLabel(f'Draw pattern to build!',
             app.width/2, initialY, size=app.titleFS)
    initialY = 20
    spacing = 15
    # if app.isShowSubd and app.subdivision:
    #     drawLabel(f'Fractal Level: {app.subdivision.fractalLevel}',
    #              app.width/2, initialY + spacing, size=app.subtitleFS)
    drawLabel(f'Build GridSize: {app.gridSize}', app.width/2, initialY + spacing*2, size=app.normalFS)
    drawLabel(f'Hand Drawing: {"ON" if app.isHandDraw else "OFF"}', app.width/2, initialY + spacing*3, fill = 'red' if app.isHandDraw else 'black', size=app.normalFS)
    
    # # Draw instructions
    # if app.isShowSubd:
    #     drawLabel('Press UP/DOWN for subdivision levels, T to return to drawing',
    #              app.width/2, app.height - 20, size=app.normalFS)
    drawLabel('Click/drag to fill cells, T for subdivision, ESC for start screen',
                app.width/2, app.height - 20, size=app.normalFS)
    drawLabel('H to toggle hand drawing, LEFT/RIGHT to resize grid (2-8)',
                app.width/2, app.height - 40, size=app.normalFS)
    drawLabel('S to save pattern, E to export pattern, R to redraw',
                 app.width/2, app.height - 60, size=app.normalFS)
    
    if app.drawHint:
        drawLabel(app.drawHint, app.width/2, 100, fill='red', size=app.subtitleFS)

def draw_onMouseDrag(app, mouseX, mouseY):
    if not app.isShowSubd:
        cell = getDrawnCell(app, mouseX, mouseY)
        if cell:
            row, col = cell
            app.drawDGrid[row][col] = 1

def onHandDraw(app, detector, handX, handY):
    if detector.prevX is not None and detector.prevY is not None and app.isHandDraw:
        screenX = app.gridLeft + (handX * app.drawDGridSize * app.drawCellSize)
        screenY = app.gridTop + (handY * app.drawDGridSize * app.drawCellSize)
        cell = getDrawnCell(app, screenX, screenY)
        print("cell@onHandDraw:", cell)
        if cell:
            row, col = cell
            app.drawDGrid[row][col] = 1
            print("done")

def changeToBool(pattern):
    return [[True if cell == 1 else False for cell in row] for row in pattern]

def draw_onKeyPress(app, key):
    if key == 'right':
        if not app.isShowSubd:
            if app.drawDGridSize < 8:
                resizeGrid(app, app.drawDGridSize + 1)
    elif key == 'left':
        if not app.isShowSubd:
            if app.drawDGridSize > 2:
                resizeGrid(app, app.drawDGridSize - 1)

    # A function for future.
    # if key == 't':  # Toggle between draw and subdivision modes
    #     app.isShowSubd = not app.isShowSubd
    #     if app.isShowSubd and app.subdivision is None:
    #         # Initialize subdivision with current grid
    #         app.subdivision = PatternSubdivision(app.drawDGrid)
    
    elif app.isShowSubd:
        if key == 'up':
            app.subdivision.subdivide()
        elif key == 'down' and app.subdivision.fractalLevel > 0:
            app.subdivision.fractalLevel -= 1  # Decrement fractal level
            app.subdivision = PatternSubdivision(app.drawDGrid)
    
    elif key == 'h':
        app.isHandDraw = not app.isHandDraw
    
    elif key == 'escape':
        setActiveScreen('start')

    elif key == 's':
        # save and print the pattern in true/false
        for row in app.drawDGrid:
            for col in row:
                if col == 1: # checked the pattern is not empty
                    app.importPattern = changeToBool(app.drawDGrid)
        if app.importPattern is None:
            print('No pattern to save')
        else:
            print(app.importPattern)
            app.drawHint = "Pattern saved! Now hit E to export."
    
    elif key == 'e':
        if app.importPattern is not None:
            print("Export clicked! return to build screen, importPattern:")
            app.importPattern = [app.importPattern]
            print(app.importPattern)
            app.cell.getImportPattern(app.importPattern)
        setActiveScreen('build')

    #reset the grid
    elif key == 'r':
        app.drawDGrid = [[0 for _ in range(app.drawDGridSize)] for _ in range(app.drawDGridSize)]
        if app.subdivision:
            app.subdivision.fractalLevel = 0
            app.subdivision.pattern = app.drawDGrid