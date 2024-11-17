3D sculpture generate app
==========================
It will be smtg like tetris3D (framework), and advanced with adding pattern rules btwn block, and block itself can be upgraded with fractal rules.
* creating board in 3D, def cell size.
* def a few pieces by 0,1 (possible to load stl and "translate it to my board?")
* def generate rules: r/s btwn pieces, adv pieces


##==============================
Main.py
##==============================
    -UIUX
    -read input keyPress
    -cread app.board, board is possible to rotate
    -load pattern, show stl/3D**
    -call pattern generator algo**

##==============================
PatternGenerator.py --> can be cool algo to generate blocks, generate connector btwn blocks, make it printable
##==============================
    def
    -keyPress == 'A'
    -keyPress == 'S'
    -keyPress == 'D'

    -generate pattern rules
        -if 'A' & 'S': ## do smtg

##==============================
FractalLevel.py --> subd the corner
##==============================


* Additional function to add if I have spare time:
##==============================
Parse.py
##==============================
#if the boundary of model > (printer size), parse it to be a new blocks in a "best way".
#Use the maze concept (back tracking) to write it.


##==============================
Slicing.py
##==============================
    -optimize 3D
    -slicing
    -generate Gcode
