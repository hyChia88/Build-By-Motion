from cmu_graphics import *
import math

def drawBezierCurve(points, steps=50, color='blue'):
    if len(points) < 2:
        return
        
    # De Casteljau's algorithm for any degree Bézier curve
    def getBezierPoint(t, points):
        temp = points.copy()
        for r in range(len(points) - 1):
            for i in range(len(temp) - 1 - r):
                temp[i] = (
                    temp[i][0] * (1-t) + temp[i+1][0] * t,
                    temp[i][1] * (1-t) + temp[i+1][1] * t
                )
        return temp[0]
    
    # Draw the curve
    prev_point = None
    for t in range(steps + 1):
        t = t / steps
        current_point = getBezierPoint(t, points)
        
        if prev_point:
            drawLine(prev_point[0], prev_point[1], 
                    current_point[0], current_point[1], 
                    fill=color, lineWidth=2)
        prev_point = current_point

def onAppStart(app):
    app.points = []  # List to store control points
    app.maxPoints = 4  # Initial max points (cubic Bézier)
    app.minPoints = 2  # Minimum points needed
    app.isDragging = False
    app.dragPointIndex = None
    app.dragRadius = 10  # Radius for detecting mouse over points

def redrawAll(app):
    # Draw helper grid
    drawGrid()
    
    # Draw control polygon (lines between control points)
    for i in range(len(app.points) - 1):
        drawLine(app.points[i][0], app.points[i][1],
                app.points[i+1][0], app.points[i+1][1],
                fill='lightgray', dashes=True)
    
    # Draw Bézier curve if we have enough points
    if len(app.points) >= 2:
        drawBezierCurve(app.points)
    
    # Draw control points
    for i, (x, y) in enumerate(app.points):
        # Different colors for start/end points and control points
        color = 'green' if i == 0 or i == len(app.points)-1 else 'red'
        drawCircle(x, y, 5, fill=color)
        drawLabel(f'P{i}', x, y-15, size=12)
    
    # Draw instructions
    drawLabel('Click to add points (max ' + str(app.maxPoints) + ' points)', 
              200, 20, size=14)
    drawLabel('UP/DOWN arrows to change curve complexity', 
              200, 40, size=14)
    drawLabel('Current points: ' + str(len(app.points)) + 
              ' / Max points: ' + str(app.maxPoints), 
              200, 60, size=14)
    drawLabel('Drag points to modify curve', 
              200, 80, size=14)

def drawGrid():
    # Draw light grid for reference
    gridColor = rgb(240, 240, 240)
    gridSpacing = 20
    for x in range(0, 400, gridSpacing):
        drawLine(x, 0, x, 400, fill=gridColor)
    for y in range(0, 400, gridSpacing):
        drawLine(0, y, 400, y, fill=gridColor)

def onMousePress(app, mouseX, mouseY):
    # Check if clicking near existing point for dragging
    for i, (x, y) in enumerate(app.points):
        if math.dist((x, y), (mouseX, mouseY)) < app.dragRadius:
            app.isDragging = True
            app.dragPointIndex = i
            return
    
    # Add new point if not at max and not dragging
    if len(app.points) < app.maxPoints and not app.isDragging:
        app.points.append((mouseX, mouseY))

def onMouseDrag(app, mouseX, mouseY):
    if app.isDragging and app.dragPointIndex is not None:
        # Update point position
        app.points[app.dragPointIndex] = (mouseX, mouseY)

def onMouseRelease(app, mouseX, mouseY):
    app.isDragging = False
    app.dragPointIndex = None

def onKeyPress(app, key):
    if key == 'up' and app.maxPoints < 8:  # Limit maximum complexity
        app.maxPoints += 1
    elif key == 'down' and app.maxPoints > app.minPoints:
        app.maxPoints = max(app.maxPoints - 1, len(app.points))
    elif key == 'space':  # Clear all points
        app.points = []

def main():
    runApp(width=1000, height=600)

main()