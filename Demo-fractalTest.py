from cmu_graphics import *

def drawSmoothCorner(x, y, cellSize, depth, orientation):
    if depth == 0:
        return
    
    # Calculate the size of smaller rectangles based on depth
    size = cellSize * (2 ** (3-depth))  # Size decreases as depth increases
    
    # Draw smaller rectangles at corners based on orientation
    if orientation == 1:  # top-left
        drawRect(x + size/2, y + size/2, size, size, fill='blue')
        drawSmoothCorner(x + size/2, y + size/2, cellSize, depth-1, 1)
    elif orientation == 2:  # top-right
        drawRect(x - size*1.5, y + size/2, size, size, fill='blue')
        drawSmoothCorner(x - size*1.5, y + size/2, cellSize, depth-1, 2)
    elif orientation == 3:  # bottom-right
        drawRect(x - size*1.5, y - size*1.5, size, size, fill='blue')
        drawSmoothCorner(x - size*1.5, y - size*1.5, cellSize, depth-1, 3)
    elif orientation == 4:  # bottom-left
        drawRect(x + size/2, y - size*1.5, size, size, fill='blue')
        drawSmoothCorner(x + size/2, y - size*1.5, cellSize, depth-1, 4)

def drawSmoothRectangle(x, y, width, height, cellSize, depth):
    if depth == 0:
        # Draw original rectangle without smooth corners
        drawRect(x, y, width, height, fill='blue')
        return
    
    cornerSize = cellSize * (2 ** (3-depth))  # Adjust corner size based on depth
    
    # Main rectangle body
    drawRect(x + cornerSize, y, width - 2*cornerSize, height, fill='blue')
    drawRect(x, y + cornerSize, width, height - 2*cornerSize, fill='blue')
    
    # Draw smooth corners using fractals
    drawSmoothCorner(x, y, cellSize, depth, 1)  # top-left
    drawSmoothCorner(x + width, y, cellSize, depth, 2)  # top-right
    drawSmoothCorner(x + width, y + height, cellSize, depth, 3)  # bottom-right
    drawSmoothCorner(x, y + height, cellSize, depth, 4)  # bottom-left

def onAppStart(app):
    app.baseSize = 10  # Base cell size
    app.rectX = 100
    app.rectY = 50
    app.rectWidth = 200
    app.rectHeight = 150
    app.fractalDepth = 0
    app.maxDepth = 3

def redrawAll(app):
    # Calculate current cell size based on depth
    currentCellSize = app.baseSize / (app.fractalDepth + 1) if app.fractalDepth > 0 else app.baseSize
    
    # Draw grid (optional, for visualization)
    gridColor = rgb(200, 200, 200)
    for x in range(0, 400, int(currentCellSize)):
        drawLine(x, 0, x, 300, fill=gridColor)
    for y in range(0, 300, int(currentCellSize)):
        drawLine(0, y, 400, y, fill=gridColor)
    
    # Draw the rectangle
    drawSmoothRectangle(app.rectX, app.rectY, app.rectWidth, app.rectHeight, 
                        currentCellSize, app.fractalDepth)
    
    # Display current depth level and instructions
    drawLabel(f'Depth Level: {app.fractalDepth}', 200, 20, size=16)
    drawLabel(f'Cell Size: {pythonRound(currentCellSize, 2)}px', 200, 40, size=14)
    drawLabel('Use UP/DOWN arrows to adjust smoothness', 200, 60, size=14)

def onKeyPress(app, key):
    if key == 'up' and app.fractalDepth < app.maxDepth:
        app.fractalDepth += 1
    elif key == 'down' and app.fractalDepth > 0:
        app.fractalDepth -= 1

def main():
    runApp(width=400, height=300)

main()