from cmu_graphics import *
import math

class Grid3D:
    def __init__(self):
        self.xSize = 4
        self.ySize = 4
        self.zSize = 4
        self.board = [[[None for x in range(self.xSize)] 
                      for y in range(self.ySize)] 
                      for z in range(self.zSize)]
        
    def is_position_valid(self, x, y, z):
        return (0 <= x < self.xSize and 
                0 <= y < self.ySize and 
                0 <= z < self.zSize)
    
    def place_cube(self, x, y, z, value):
        if self.is_position_valid(x, y, z):
            self.board[z][y][x] = value
            return True
        return False
    
    def get_cube(self, x, y, z):
        if self.is_position_valid(x, y, z):
            return self.board[z][y][x]
        return None

def onAppStart(app):
    app.grid = Grid3D()
    app.cubeSize = 80
    app.currentX = 0
    app.currentY = 0
    app.currentZ = 0
    app.rotationY = math.pi/4
    app.rotationX = math.pi/6
    app.showPreview = True
    app.boardLeft = 200
    app.boardTop = 100
    app.boardWidth = 600
    app.boardHeight = 600
    app.dragging = False
    app.lastMouseX = 0
    app.lastMouseY = 0
    app.score = 0

def projectPoint(app, x, y, z):
    # Apply Y rotation first
    rotX = x * math.cos(app.rotationY) - y * math.sin(app.rotationY)
    rotY = x * math.sin(app.rotationY) + y * math.cos(app.rotationY)
    
    # Then apply X rotation (tilt)
    finalY = rotY * math.cos(app.rotationX) - z * math.sin(app.rotationX)
    finalZ = rotY * math.sin(app.rotationX) + z * math.cos(app.rotationX)
    
    # Scale and translate to screen coordinates
    screenX = app.boardLeft + app.boardWidth/2 + rotX * app.cubeSize
    screenY = app.boardTop + app.boardHeight/2 + finalY * app.cubeSize
    
    return (screenX, screenY, finalZ)

def drawGridPlanes(app):
    gridColor = rgb(200, 200, 200)
    
    # Draw bottom grid lines
    for i in range(app.grid.xSize + 1):
        start = projectPoint(app, i, 0, 0)
        end = projectPoint(app, i, app.grid.ySize, 0)
        drawLine(start[0], start[1], end[0], end[1], 
                fill=gridColor, dashes=True)
    
    for i in range(app.grid.ySize + 1):
        start = projectPoint(app, 0, i, 0)
        end = projectPoint(app, app.grid.xSize, i, 0)
        drawLine(start[0], start[1], end[0], end[1], 
                fill=gridColor, dashes=True)
    
    # Draw vertical lines
    for i in range(app.grid.xSize + 1):
        for j in range(app.grid.ySize + 1):
            start = projectPoint(app, i, j, 0)
            end = projectPoint(app, i, j, app.grid.zSize)
            drawLine(start[0], start[1], end[0], end[1], 
                    fill=gridColor, dashes=True)
    
    # Draw coordinate axes
    origin = projectPoint(app, 0, 0, 0)
    xAxis = projectPoint(app, app.grid.xSize, 0, 0)
    yAxis = projectPoint(app, 0, app.grid.ySize, 0)
    zAxis = projectPoint(app, 0, 0, app.grid.zSize)
    
    drawLine(origin[0], origin[1], xAxis[0], xAxis[1], 
            fill='red', lineWidth=2)
    drawLine(origin[0], origin[1], yAxis[0], yAxis[1], 
            fill='green', lineWidth=2)
    drawLine(origin[0], origin[1], zAxis[0], zAxis[1], 
            fill='blue', lineWidth=2)

def drawCube(app, x, y, z, color='blue', isPreview=False):
    vertices = [
        (x, y, z),         # 0: front bottom left
        (x+1, y, z),       # 1: front bottom right
        (x+1, y+1, z),     # 2: back bottom right
        (x, y+1, z),       # 3: back bottom left
        (x, y, z+1),       # 4: front top left
        (x+1, y, z+1),     # 5: front top right
        (x+1, y+1, z+1),   # 6: back top right
        (x, y+1, z+1)      # 7: back top left
    ]
    
    projectedVertices = [projectPoint(app, *v) for v in vertices]
    
    faces = [
        ([0,1,5,4], 0),      # Front
        ([1,2,6,5], -20),    # Right
        ([2,3,7,6], -40),    # Back
        ([3,0,4,7], -20),    # Left
        ([4,5,6,7], -10),    # Top
        ([0,1,2,3], -30),    # Bottom
    ]
    
    faces.sort(key=lambda f: sum(projectedVertices[i][2] for i in f[0])/4)
    
    opacity = 70 if isPreview else 100
    edgeWidth = 4 if isPreview else 2
    baseColor = rgb(100,100,255) if color == 'blue' else rgb(200,200,200)
    
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
    drawGridPlanes(app)
    
    # Sort and draw cubes
    cubes_to_draw = []
    
    for z in range(app.grid.zSize):
        for y in range(app.grid.ySize):
            for x in range(app.grid.xSize):
                if app.grid.get_cube(x, y, z) is not None:
                    depth = projectPoint(app, x+0.5, y+0.5, z+0.5)[2]
                    cubes_to_draw.append((depth, (x, y, z), False))
    
    if app.showPreview:
        depth = projectPoint(app, app.currentX+0.5, 
                           app.currentY+0.5, app.currentZ+0.5)[2]
        cubes_to_draw.append((depth, (app.currentX, app.currentY, app.currentZ), True))
    
    cubes_to_draw.sort(key=lambda x: x[0])
    for _, (x, y, z), isPreview in cubes_to_draw:
        drawCube(app, x, y, z, color='grey' if isPreview else 'blue', 
                 isPreview=isPreview)

def drawArrow(centerX, centerY, angle, length, color='black'):
    # Draw line
    endX = centerX + math.cos(angle) * length
    endY = centerY + math.sin(angle) * length
    drawLine(centerX, centerY, endX, endY, fill=color)
    
    # Draw arrow head
    arrowSize = 8
    angle1 = angle - math.pi/6
    angle2 = angle + math.pi/6
    
    point1X = endX - math.cos(angle1) * arrowSize
    point1Y = endY - math.sin(angle1) * arrowSize
    point2X = endX - math.cos(angle2) * arrowSize
    point2Y = endY - math.sin(angle2) * arrowSize
    
    drawPolygon(endX, endY, point1X, point1Y, point2X, point2Y, fill=color)

def drawMovemenyoutGuide(app):
    centerX, centerY = 100, 700
    radius = 30
    
    drawCircle(centerX, centerY, radius, fill=None, border='gray')
    
    rotation = app.rotationY
    arrowLength = radius * 0.8
    
    directions = {
        'up': (-rotation),
        'right': (-rotation + math.pi/2),
        'down': (-rotation + math.pi),
        'left': (-rotation - math.pi/2)
    }
    
    for direction, angle in directions.items():
        drawArrow(centerX, centerY, angle, arrowLength)
        labelX = centerX + math.cos(angle) * (radius + 15)
        labelY = centerY + math.sin(angle) * (radius + 15)
        drawLabel(direction, labelX, labelY, size=12)

def getViewBasedMovement(app, key):
    rotation = app.rotationY % (2 * math.pi)
    
    movements = {
        'up':    (0, -1),
        'down':  (0, 1),
        'left':  (-1, 0),
        'right': (1, 0)
    }
    
    if key not in movements:
        return (0, 0)
    
    dx, dy = movements[key]
    quadrant = int((rotation + math.pi/4) / (math.pi/2)) % 4
    
    if quadrant == 0:
        return (dx, dy)
    elif quadrant == 1:
        return (-dy, dx)
    elif quadrant == 2:
        return (-dx, -dy)
    else:
        return (dy, -dx)

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
        if app.grid.get_cube(app.currentX, app.currentY, app.currentZ) is None:
            app.grid.place_cube(app.currentX, app.currentY, app.currentZ, True)
            app.currentZ += 1
            app.score += 1
            if app.currentZ >= app.grid.zSize:
                app.currentZ = 0
    elif key == 'r':
        onAppStart(app)

def onMousePress(app, mouseX, mouseY):
    app.dragging = True
    app.lastMouseX = mouseX
    app.lastMouseY = mouseY

def onMouseDrag(app, mouseX, mouseY):
    if app.dragging:
        dx = mouseX - app.lastMouseX
        dy = mouseY - app.lastMouseY
        
        app.rotationY += dx * 0.01
        app.rotationX += dy * 0.01
        
        app.rotationX = min(max(app.rotationX, -math.pi/2), math.pi/2)
        
        app.lastMouseX = mouseX
        app.lastMouseY = mouseY

def onMouseRelease(app, mouseX, mouseY):
    app.dragging = False

def redrawAll(app):
    drawRect(0, 0, 800, 800, fill='white')
    drawGrid(app)
    
    # Instructions
    drawLabel('3D Grid Game', 400, 30, size=20)
    drawLabel('Arrow keys: Move relative to view', 400, 60, size=16)
    drawLabel('Space: Place cube and move up', 400, 90, size=16)
    drawLabel('Drag mouse to rotate view', 400, 120, size=16)
    drawLabel(f'Position: ({app.currentX}, {app.currentY}, {app.currentZ})', 
              400, 150, size=16)
    drawLabel(f'Score: {app.score}', 400, 180, size=16)
    drawLabel('R: Reset game', 400, 210, size=16)
    
    drawMovementGuide(app)
    
    # Axis labels
    drawLabel('Red: X-axis', 700, 700, fill='red', size=14)
    drawLabel('Green: Y-axis', 700, 720, fill='green', size=14)
    drawLabel('Blue: Z-axis', 700, 740, fill='blue', size=14)

def main():
    runApp(width=800, height=800)

main()