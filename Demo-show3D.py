from cmu_graphics import *
import math

def onAppStart(app):
    # Define cube vertices in 3D space (x, y, z)
    app.vertices = [
        (-1, -1, -1),  # 0: back-bottom-left
        (1, -1, -1),   # 1: back-bottom-right
        (1, 1, -1),    # 2: back-top-right
        (-1, 1, -1),   # 3: back-top-left
        (-1, -1, 1),   # 4: front-bottom-left
        (1, -1, 1),    # 5: front-bottom-right
        (1, 1, 1),     # 6: front-top-right
        (-1, 1, 1),    # 7: front-top-left
    ]
    
    # Define edges as pairs of vertex indices
    app.edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # back face
        (4, 5), (5, 6), (6, 7), (7, 4),  # front face
        (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
    ]
    
    # Initial rotation angles
    app.angleX = 30
    app.angleY = 30
    app.angleZ = 0
    
    # Scale factor for the cube
    app.scale = 50

def rotatePoint(x, y, z, angleX, angleY, angleZ):
    # Convert angles to radians
    angleX = math.radians(angleX)
    angleY = math.radians(angleY)
    angleZ = math.radians(angleZ)
    
    # Rotate around X axis
    y2 = y * math.cos(angleX) - z * math.sin(angleX)
    z2 = y * math.sin(angleX) + z * math.cos(angleX)
    y, z = y2, z2
    
    # Rotate around Y axis
    x2 = x * math.cos(angleY) + z * math.sin(angleY)
    z2 = -x * math.sin(angleY) + z * math.cos(angleY)
    x, z = x2, z2
    
    # Rotate around Z axis
    x2 = x * math.cos(angleZ) - y * math.sin(angleZ)
    y2 = x * math.sin(angleZ) + y * math.cos(angleZ)
    x, y = x2, y2
    
    return x, y, z

def project(x, y, z):
    # Simple perspective projection
    factor = 4 / (4 + z)
    x2 = x * factor
    y2 = y * factor
    return x2, y2

def onKeyPress(app, key):
    # Rotate cube with arrow keys
    if key == 'left':
        app.angleY -= 5
    elif key == 'right':
        app.angleY += 5
    elif key == 'up':
        app.angleX -= 5
    elif key == 'down':
        app.angleX += 5

def redrawAll(app):
    # Center of the screen
    centerX = app.width / 2
    centerY = app.height / 2
    
    # Draw each edge
    for edge in app.edges:
        # Get the two vertices for this edge
        v1 = app.vertices[edge[0]]
        v2 = app.vertices[edge[1]]
        
        # Rotate both vertices
        x1, y1, z1 = rotatePoint(v1[0], v1[1], v1[2], 
                                app.angleX, app.angleY, app.angleZ)
        x2, y2, z2 = rotatePoint(v2[0], v2[1], v2[2], 
                                app.angleX, app.angleY, app.angleZ)
        
        # Project to 2D
        px1, py1 = project(x1, y1, z1)
        px2, py2 = project(x2, y2, z2)
        
        # Scale and translate to center
        sx1 = centerX + px1 * app.scale
        sy1 = centerY + py1 * app.scale
        sx2 = centerX + px2 * app.scale
        sy2 = centerY + py2 * app.scale
        
        # Draw the edge
        drawLine(sx1, sy1, sx2, sy2)

def main():
    runApp(width=400, height=400)

main()