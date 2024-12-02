def resize(newSize):
    if not resizable:
        return False
    elif 1 <= newSize <= 3:
        size = newSize
        pattern = [[[True for _ in range(size)] 
                        for _ in range(size)]
                        for _ in range(size-1)]
        return pattern
    return pattern

def getPlacementPos(pattern):
    cellPosList = []
    # get the exact pattern pos of cell, send to board and make it not None (occupied)
    for x in range(len(pattern)):
        for y in range(len(pattern[x])):
            for z in range(len(pattern[x][y])):
                if pattern[x][y][z] is True:
                    pos = [x, y, z]
                    cellPosList.append(pos)
    return cellPosList
'''
class StairCell(Cell):
    def __init__(self, x, y, z, fracLevel=1):
        super().__init__(x, y, z, fracLevel)
        self.resizable = False
        self.pattern = [
                            [[True, True], [True, False]],
                            [[True, True], [False, False]],
                            [[True, False], [False, False]]
                        ]
                        
                        [
                            [[True, True, True], [True, True, True], [True, True, True]],
                            [[True, True, True], [True, True, True], [True, True, True]],
                            [[True, True, True], [True, True, True], [True, True, True]]
                        ]
'''

size = 3
resizable = True

print(resize(size))
print(getPlacementPos(resize(size)))
