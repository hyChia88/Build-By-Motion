'''
This is the menu page that allows the user to choose between drawing and building
'''
from cmu_graphics import *
# from Draw import *
# from Build import *

def onAppStart(app):    
    app.subdIcon = 'draw.png'
    app.drawIcon = 'build.png'
    
    app.buttonWidth = 200
    app.buttonHeight = 100
    app.iconSize = 30

def onMousePress(app, mouseX, mouseY):
    if app.width/2 - app.buttonWidth/2 <= mouseX <= app.width/2 + app.buttonWidth/2:
        if app.height/2 - app.buttonHeight - app.buttonHeight/2 <= mouseY <= app.height/2 - app.buttonHeight + app.buttonHeight/2:
            print('draw')

        elif app.height/2 + app.buttonHeight - app.buttonHeight/2 <= mouseY <= app.height/2 + app.buttonHeight + app.buttonHeight/2:
            print('build')
            
        else:
            return None

def redrawAll(app):
    drawLabel('Menu', app.width/2, 50, size=24)
    
    drawRect(app.width/2, app.height/2 - app.buttonHeight, app.buttonWidth, app.buttonHeight, fill=None, border='black', align='center')
    drawLabel('Draw', app.width/2, app.height/2 - app.buttonHeight, size=24)
    drawImage(app.drawIcon, app.width/2-app.buttonWidth/2-app.iconSize, app.height/2 - app.buttonHeight, align='center', width=app.iconSize, height=app.iconSize)
    
    drawRect(app.width/2, app.height/2 + app.buttonHeight, app.buttonWidth, app.buttonHeight, fill=None, border='black', align='center')
    drawLabel('Build', app.width/2, app.height/2 + app.buttonHeight, size=24)
    drawImage(app.subdIcon, app.width/2-app.buttonWidth/2-app.iconSize, app.height/2 + app.buttonHeight, align='center', width=app.iconSize, height=app.iconSize)
    
def main():
    runApp(width=1200, height=600)

main()