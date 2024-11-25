3D sculpture generate app
==========================
1. Project Title and Description
It will be smtg like tetris3D (framework), and advanced with adding pattern rules btwn block, and block itself can be upgraded with fractal rules. (using fractal to subdivide the corner)
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
fractal level 0, cell size = 1/1
[F,T,T,F]
[T,T,T,T]
[T,T,T,T]
[F,T,T,F]

fractal level 1, cell size = 1/2
[F,F,T,T,T,T,F,F]
[F,T,T,T,T,T,T,F]
[T,T,T,T,T,T,T,T]
[T,T,T,T,T,T,T,T]
[T,T,T,T,T,T,T,T]
[T,T,T,T,T,T,T,T]
[F,T,T,T,T,T,T,F]
[F,F,T,T,T,T,F,F]

fractal level 2, cell size = 1/4
[F,F,F,F,T,T,T,T,T,T,T,T,F,F,F,F]
[F,F,T,T,T,T,T,T,T,T,T,T,T,T,F,F]
[F,T,T,T,T,T,T,T,T,T,T,T,T,T,T,F]
[F,T,T,T,T,T,T,T,T,T,T,T,T,T,T,F]
[T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T]
[T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T]
[T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T]
[T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T]
[F,T,T,T,T,T,T,T,T,T,T,T,T,T,T,F]
[F,F,T,T,T,T,T,T,T,T,T,T,T,T,F,F]
[F,F,F,F,T,T,T,T,T,T,T,T,F,F,F,F]

Another example:
fractal level 0, cell size = 1/1
[1,1,1,1]
[1,1,1,1]
[0,0,1,1]
[0,0,1,1]

fractal level 1, cell size = 1/2
[0,1,1,1,1,1,1,0]
[1,1,1,1,1,1,1,1]
[1,1,1,1,1,1,1,1]
[0,1,1,1,1,1,1,1]
[0,0,0,1,1,1,1,1]
[0,0,0,0,1,1,1,1]
[0,0,0,0,1,1,1,1]
[0,0,0,0,0,1,1,0]

fractal level 2, cell size = 1/4
[0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0]
[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
[0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
[0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1]
[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0]
[0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0]


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
