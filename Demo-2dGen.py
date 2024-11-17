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
    app.smoothLevel = 0
    app.maxSmoothLevel = 5
    app.showCenters = False  # Toggle for debugging

def projectPoint(app, x, y, z):
    rotX = x * math.cos(app.rotationY) - y * math.sin(app.rotationY)
    rotY = x * math.sin(app.rotationY) + y * math.cos(app.rotationY)
    finalY = rotY * math.cos(app.rotationX) - z * math.sin(app.rotationX)
    finalZ = rotY * math.sin(app.rotationX) + z * math.cos(app.rotationX)
    screenX = app.boardLeft + app.boardWidth/2 + rotX * app.cubeSize
    screenY = app.boardTop + app.boardHeight/2 + finalY * app.cubeSize
    return (screenX, screenY, finalZ)

def drawGridPlanes(app):
    gridColor = rgb(200, 200, 200)
    
    for i in range(app.grid.xSize + 1):
        start = projectPoint(app, i, 0, 0)
        end = projectPoint(app, i, app.grid.ySize, 0)
        drawLine(start[0], start[1], end[0], end[1], fill=gridColor, dashes=True)
    
    for i in range(app.grid.ySize + 1):
        start = projectPoint(app, 0, i, 0)
        end = projectPoint(app, app.grid.xSize, i, 0)
        drawLine(start[0], start[1], end[0], end[1], fill=gridColor, dashes=True)
    
    for i in range(app.grid.xSize + 1):
        for j in range(app.grid.ySize + 1):
            start = projectPoint(app, i, j, 0)
            end = projectPoint(app, i, j, app.grid.zSize)
            drawLine(start[0], start[1], end[0], end[1], fill=gridColor, dashes=True)
    
    origin = projectPoint(app, 0, 0, 0)
    xAxis = projectPoint(app, app.grid.xSize, 0, 0)
    yAxis = projectPoint(app, 0, app.grid.ySize, 0)
    zAxis = projectPoint(app, 0, 0, app.grid.zSize)
    
    drawLine(origin[0], origin[1], xAxis[0], xAxis[1], fill='red', lineWidth=2)
    drawLine(origin[0], origin[1], yAxis[0], yAxis[1], fill='green', lineWidth=2)
    drawLine(origin[0], origin[1], zAxis[0], zAxis[1], fill='blue', lineWidth=2)

def interpolate(p1, p2, factor):
    return (p1[0] + (p2[0] - p1[0]) * factor,
            p1[1] + (p2[1] - p1[1]) * factor,
            p1[2] + (p2[2] - p1[2]) * factor)

def drawCube(app, x, y, z, color='blue', isPreview=False):
    vertices = [
        (x, y, z),                     # 0: front bottom left
        (x + 1, y, z),                 # 1: front bottom right
        (x + 1, y + 1, z),             # 2: back bottom right
        (x, y + 1, z),                 # 3: back bottom left
        (x, y, z + 1),                 # 4: front top left
        (x + 1, y, z + 1),             # 5: front top right
        (x + 1, y + 1, z + 1),         # 6: back top right
        (x, y + 1, z + 1)              # 7: back top left
    ]

    centers = {
        'front': (x + 0.5, y, z + 0.5),
        'back': (x + 0.5, y + 1, z + 0.5),
        'left': (x, y + 0.5, z + 0.5),
        'right': (x + 1, y + 0.5, z + 0.5),
        'top': (x + 0.5, y + 0.5, z + 1),
        'bottom': (x + 0.5, y + 0.5, z)
    }

    points = []
    if app.smoothLevel > 0:
        smooth_factor = app.smoothLevel / app.maxSmoothLevel
        for i, vertex in enumerate(vertices):
            connected_centers = []
            if i in [0, 1, 4, 5]: connected_centers.append(centers['front'])
            if i in [2, 3, 6, 7]: connected_centers.append(centers['back'])
            if i in [0, 3, 4, 7]: connected_centers.append(centers['left'])
            if i in [1, 2, 5, 6]: connected_centers.append(centers['right'])
            if i in [4, 5, 6, 7]: connected_centers.append(centers['top'])
            if i in [0, 1, 2, 3]: connected_centers.append(centers['bottom'])

            avg_x = sum(c[0] for c in connected_centers) / len(connected_centers)
            avg_y = sum(c[1] for c in connected_centers) / len(connected_centers)
            avg_z = sum(c[2] for c in connected_centers) / len(connected_centers)
            target_point = (avg_x, avg_y, avg_z)

            smooth_point = interpolate(vertex, target_point, smooth_factor)
            points.append(smooth_point)
    else:
        points = vertices

    projectedPoints = [projectPoint(app, px, py, pz) for px, py, pz in points]

    faces = [
        ([0,1,5,4], 0),      # Front
        ([1,2,6,5], -20),    # Right
        ([2,3,7,6], -40),    # Back
        ([3,0,4,7], -20),    # Left
        ([4,5,6,7], -10),    # Top
        ([0,1,2,3], -30),    # Bottom
    ]

    faces.sort(key=lambda f: sum(projectedPoints[i][2] for i in f[0])/4)

    opacity = 70 if isPreview else 100
    edgeWidth = 4 if isPreview else 2
    baseColor = rgb(100,100,255) if color == 'blue' else rgb(200,200,200)

    for faceIndices, colorAdj in faces:
        faceColor = rgb(max(0, baseColor.red + colorAdj),
                       max(0, baseColor.green + colorAdj),
                       max(0, baseColor.blue + colorAdj))
        
        facePoints = []
        for idx in faceIndices:
            facePoints.extend([projectedPoints[idx][0], projectedPoints[idx][1]])
        
        drawPolygon(*facePoints, fill=faceColor, opacity=opacity,
                    border='black', borderWidth=edgeWidth)

    if app.showCenters:
        for center in centers.values():
            cp = projectPoint(app, center[0], center[1], center[2])
            drawCircle(cp[0], cp[1], 2, fill='red')

def drawGrid(app):
    drawGridPlanes(app)
    
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

def drawMovementGuide(app):
    centerX, centerY = 100, 700
    radius = 30
    drawCircle(centerX, centerY, radius, fill=None, border='gray')
    
    for direction, angle in [('up', -app.rotationY),
                           ('right', -app.rotationY + math.pi/2),
                           ('down', -app.rotationY + math.pi),
                           ('left', -app.rotationY - math.pi/2)]:
        endX = centerX + math.cos(angle) * radius
        endY = centerY + math.sin(angle) * radius
        drawLine(centerX, centerY, endX, endY, fill='black')
        
        labelX = centerX + math.cos(angle) * (radius + 15)
        labelY = centerY + math.sin(angle) * (radius + 15)
        drawLabel(direction, labelX, labelY, size=12)

def onKeyPress(app, key):
    if key == 'w' and app.smoothLevel < app.maxSmoothLevel:
        app.smoothLevel += 1
    elif key == 's' and app.smoothLevel > 0:
        app.smoothLevel -= 1
    elif key == 'left':
        if app.currentX > 0:
            app.currentX -= 1
    elif key == 'right':
        if app.currentX < app.grid.xSize - 1:
            app.currentX += 1
    elif key == 'up':
        if app.currentY > 0:
            app.currentY -= 1
    elif key == 'down':
        if app.currentY < app.grid.ySize - 1:
            app.currentY += 1
    elif key == '+':
        if app.currentZ >= app.grid.zSize:
                app.currentZ +=1
    elif key == 'space':
        if app.grid.get_cube(app.currentX, app.currentY, app.currentZ) is None:
            app.grid.place_cube(app.currentX, app.currentY, app.currentZ, True)
            app.currentZ += 1
            app.score += 1
            if app.currentZ >= app.grid.zSize:
                app.currentZ = 0
    elif key == 'r':
        onAppStart(app)
    elif key == 'c':  # Toggle center points visibility
        app.showCenters = not app.showCenters
    elif key == 'p':  # 'p' for position input
        inputPosition(app)

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

def inputPosition(app):
    try:
        # Get input from user
        xStr = app.getTextInput('Enter X position (0-3):')
        if xStr is None: return
        yStr = app.getTextInput('Enter Y position (0-3):')
        if yStr is None: return
        zStr = app.getTextInput('Enter Z position (0-3):')
        if zStr is None: return
        
        # Convert to integers
        x = int(xStr)
        y = int(yStr)
        z = int(zStr)
        
        # Validate positions
        if (0 <= x < app.grid.xSize and 
            0 <= y < app.grid.ySize and 
            0 <= z < app.grid.zSize):
            app.currentX = x
            app.currentY = y
            app.currentZ = z
        else:
            app.showMessage('Invalid position! Use values between 0 and 3.')
    except ValueError:
        app.showMessage('Please enter valid numbers!')

def onMouseRelease(app, mouseX, mouseY):
    app.dragging = False

def redrawAll(app):
    drawRect(0, 0, 800, 800, fill='white')
    drawGrid(app)
    
    drawLabel('3D Grid Game', 400, 30, size=20)
    drawLabel('Left/Right: Move horizontally', 400, 60, size=16)
    drawLabel('W/S: Move forward/backward', 400, 90, size=16)
    drawLabel('Up/Down: Adjust smoothness', 400, 120, size=16)
    drawLabel('Space: Place cube', 400, 150, size=16)
    drawLabel('C: Toggle center points', 400, 180, size=16)
    drawLabel('Drag mouse to rotate view', 400, 210, size=16)
    drawLabel(f'Position: ({app.currentX}, {app.currentY}, {app.currentZ})', 
              400, 240, size=16)
    drawLabel(f'Score: {app.score}', 400, 270, size=16)
    drawLabel(f'Smoothness Level: {app.smoothLevel}', 400, 300, size=16)
    drawLabel('R: Reset game', 400, 330, size=16)
    drawLabel('P: Input position directly', 400, 360, size=16)
    
    drawMovementGuide(app)
    
    drawLabel('Red: X-axis', 700, 700, fill='red', size=14)
    drawLabel('Green: Y-axis', 700, 720, fill='green', size=14)
    drawLabel('Blue: Z-axis', 700, 740, fill='blue', size=14)



def main():
    runApp(width=800, height=800)

main()