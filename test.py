from cmu_graphics import *

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
        drawRect(0, 0, app.width, app.height, fill='black')
        
        # Calculate cell size for subdivided pattern
        cell_size = (min(app.width, app.height) - 2*app.margin) / len(app.subdivision.pattern)
        
        # Draw subdivided pattern
        for y in range(len(app.subdivision.pattern)):
            for x in range(len(app.subdivision.pattern)):
                if app.subdivision.pattern[y][x]:
                    drawRect(app.margin + x * cell_size, 
                           app.margin + y * cell_size,
                           cell_size, cell_size,
                           fill='white')

def redrawAll(app):
    if app.isShowSubd:
        drawSubdivision(app)
    else:
        drawGrid(app)
    
    # Draw mode and level info
    drawLabel(f'Mode: {"Drawing" if not app.isShowSubd else "Subdivision"}',
             app.width/2, 20, size=16)
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

def onMouseDrag(app, mouseX, mouseY):
    if not app.isShowSubd:
        cell = getCell(app, mouseX, mouseY)
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
def main():
    runApp(width=800, height=600)

main()