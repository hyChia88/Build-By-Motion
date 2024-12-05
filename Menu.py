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
    app.spacing = 30
    app.logoHeight = 200
    app.drawnPattern = None
    app.importPattern = None
    
    # Import the init status of Build and Draw
    buildInit(app)
    drawInit(app)

############################################################
# Start Screen
############################################################

def start_redrawAll(app):
    drawLabel('Welcome!', app.width/2, app.height/2 - app.spacing*4, size=24, bold=True)
    # Note: we can access app.highScore (and all app variables) from any screen
    drawLabel(f'This is the start screen!', app.width/2, app.height/2 - app.spacing*2, size=24)
    drawLabel('Press space or click logo to build!', app.width/2, app.height/2 - app.spacing, size=16)
    
    drawImage('logo.png', app.width/2, app.height/2 + app.spacing*2 + app.logoHeight/2, width=app.logoHeight, height = app.logoHeight, align= "center")
    
def start_onKeyPress(app, key):
    if key == 'space':
        setActiveScreen('build')

def start_onMousePress(app, mouseX, mouseY):
    if app.width/2 - app.logoHeight/2 < mouseX < app.width/2 + app.logoHeight/2 and \
       app.height/2 + app.spacing*2 < mouseY < app.height/2 + app.spacing*2 + app.logoHeight:
        setActiveScreen('build')

############################################################
# Main
############################################################

def main():
    runAppWithScreens(initialScreen='start', width=1200, height=750)

main()