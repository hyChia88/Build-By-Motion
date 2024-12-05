from cv2 import AlignExposures
from cmu_graphics import *
from Build import *
from Draw import *
'''
https://academy.cs.cmu.edu/cpcs-docs/screens
'''
def onAppStart(app):
    # The model is shared between all screens
    app.setMaxShapeCount(5000)
    app.spacingMenu = 30
    app.logoHeight = 200
    app.drawnPattern = None
    app.importPattern = None
    app.titleFS = 24
    app.subtitleFS = 16
    app.normalFS = 12
    
    # Import the init status of Build and Draw
    app.posListAll = []
    buildInit(app)
    drawInit(app)

############################################################
# Start Screen
############################################################

def start_redrawAll(app):
    drawLabel('Welcome!', app.width/2, app.height/2 - app.spacingMenu*4, size=app.titleFS, bold=True)
    # Note: we can access app.highScore (and all app variables) from any screen
    drawLabel(f'This is the start screen!', app.width/2, app.height/2 - app.spacingMenu*2, size=app.titleFS)
    drawLabel('Press space or click logo to build!', app.width/2, app.height/2 - app.spacingMenu, size=app.subtitleFS)
    
    drawImage('logo.png', app.width/2, app.height/2 + app.spacingMenu*2 + app.logoHeight/2, width=app.logoHeight, height = app.logoHeight, align= "center")
    
def start_onKeyPress(app, key):
    if key == 'space':
        setActiveScreen('build')

def start_onMousePress(app, mouseX, mouseY):
    if app.width/2 - app.logoHeight/2 < mouseX < app.width/2 + app.logoHeight/2 and \
       app.height/2 + app.spacingMenu*2 < mouseY < app.height/2 + app.spacingMenu*2 + app.logoHeight:
        setActiveScreen('build')

############################################################
# Main
############################################################

def main():
    runAppWithScreens(initialScreen='start', width=1200, height=750)

main()