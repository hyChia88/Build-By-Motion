from cmu_graphics import *
import math

def drawCube(app):
    # Center point
    cx, cy = 200, 200
    
    # Calculate offsets
    dx = app.cubeSize * math.cos(math.radians(app.angle))
    dy = app.cubeSize * math.sin(math.radians(app.angle))
    
    # Front face (pink)
    drawPolygon(
        cx, cy,
        cx + dx, cy - dy,
        cx + dx, cy - dy - app.cubeSize,
        cx, cy - app.cubeSize,
        fill='pink'
    )
    
    # Right face (lightBlue)
    drawPolygon(
        cx + dx, cy - dy,
        cx + 2*dx, cy,
        cx + 2*dx, cy - app.cubeSize,
        cx + dx, cy - dy - app.cubeSize,
        fill='lightBlue'
    )
    
    # Top face (lightGreen)
    drawPolygon(
        cx, cy - app.cubeSize,
        cx + dx, cy - dy - app.cubeSize,
        cx + 2*dx, cy - app.cubeSize,
        cx + dx, cy - app.cubeSize + dy,
        fill='lightGreen'
    )
    
    # Front face (boarder)
    drawPolygon(
        cx, cy, 
        cx, cy - app.cubeSize,
        cx + dx, cy - app.cubeSize + dy,
        cx + dx, cy + app.cubeSize - dy, 
        fill=None, border = 'black'
    )
    
    drawPolygon(
        cx + dx, cy - app.cubeSize + dy,
        cx + dx, cy + app.cubeSize - dy, 
        cx + dx*2, cy,
        cx + dx*2, cy - app.cubeSize,
        fill=None, border = 'black'
    )

def onAppStart(app):
    app.cubeSize = 80
    app.angle = 30

def redrawAll(app):
    drawCube(app)

runApp()