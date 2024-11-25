from cmu_graphics import *
import math

def getMidpoint(p1, p2):
    return ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)

def getNormalVector(p1, p2):
    # Get direction vector
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # Calculate normal (perpendicular) vector
    length = math.sqrt(dx*dx + dy*dy)
    if length == 0: return (0, 0)
    # Return normalized perpendicular vector
    return (-dy/length, dx/length)

def subdivideEdge(app, p1, p2, fracLevel, shiftDist=20):
    # Base case
    if fracLevel <= 1:
        drawLine(p1[0], p1[1], p2[0], p2[1], fill='black')
        return
    
    # Get midpoint
    mid = getMidpoint(p1, p2)
    
    # Calculate shifted midpoint
    normal = getNormalVector(p1, p2)
    shiftedMid = (mid[0] + normal[0]*shiftDist, 
                 mid[1] + normal[1]*shiftDist)
    
    # Draw points for visualization
    drawCircle(p1[0], p1[1], 3, fill='blue')
    drawCircle(p2[0], p2[1], 3, fill='blue')
    drawCircle(shiftedMid[0], shiftedMid[1], 3, fill='red')
    
    # Recursive calls with shifted midpoint
    subdivideEdge(app, p1, shiftedMid, fracLevel-1, shiftDist*0.5)
    subdivideEdge(app, shiftedMid, p2, fracLevel-1, shiftDist*0.5)

def onAppStart(app):
    app.fracLevel = 5
    app.shiftDist = 40
    
def redrawAll(app):
    # Clear background
    drawRect(0, 0, app.width, app.height, fill='white')
    
    # Draw a subdivided line from left to right
    startPoint = (100, app.height/2)
    endPoint = (app.width-100, app.height/2)
    subdivideEdge(app, startPoint, endPoint, app.fracLevel, app.shiftDist)
    
    # Draw instructions
    drawLabel('Press up/down to change fractal level', 
             app.width/2, 30, size=16)
    drawLabel(f'Current Level: {app.fracLevel}', 
             app.width/2, 50, size=16)

def onKeyPress(app, key):
    if key == 'up' and app.fracLevel < 8:
        app.fracLevel += 1
    elif key == 'down' and app.fracLevel > 1:
        app.fracLevel -= 1

def main():
    runApp(width=800, height=400)

main()