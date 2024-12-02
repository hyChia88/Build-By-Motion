# import math
# import numpy as np

# rotationY = 45
# rotationX = 60

# def threeDtotwoD(treeDPoints,rotationY,rotationX):
#     #inspired by 15-112 3D graphics mini lecture
#     transformationMatrix=[[ math.cos(rotationY),math.sin(rotationY)],
#                           [ math.cos(rotationX),math.sin(rotationX)],
#                         [ 0, 1]]
#     twoDPoints=np.matmul(treeDPoints,transformationMatrix)
#     return twoDPoints

# print(threeDtotwoD([[1,2,3]],rotationY,rotationX))

# def testProj(x, y, z, rotationY,rotationX):
#     # 3d matrix rotation
#     matrix = [[ math.cos(rotationY),math.sin(rotationY)],
#               [ math.cos(rotationX),math.sin(rotationX)],
#               [ 0, 1]]
#     # matrix multiplication with 3d point
#     xcood = x * matrix[0][0] + y * matrix[1][0] + z * matrix[2][0]
#     ycood = x * matrix[0][1] + y * matrix[1][1] + z * matrix[2][1]
    
#     '''
#     xcood = x * math.cos(rotationY) + y * math.cos(rotationX) + z * 0
#     ycood = x * math.sin(rotationY) + y * math.sin(rotationX) + z * 1
#     '''
    
#     return (xcood,ycood)

# print(testProj(1,2,3,45,60))

# def basicProj(x, y, z, rotationY,rotationX):
#     # rotationY = math.radians(rotationY)
#     # rotationX = math.radians(rotationX)
#     '''
#     Do Y-axis rotation first, then X-axis rotation
#     scale and translate to screen coordinates
#     matrix = [[ math.cos(rotationY),math.sin(rotationY)],
#             [ math.cos(rotationX),math.sin(rotationX)],
#             [ 0, 1]]
#     '''
    
#     finalX = x * math.cos(rotationY) - y * math.sin(rotationY)
#     finalY = x * math.sin(rotationY)* math.cos(rotationX) + \
#              y * math.cos(rotationY) * math.cos(rotationX) -\
#              z * math.sin(rotationX)
    
#     screenX = finalX
#     screenY = finalY
    
#     return (screenX, screenY)

# print(basicProj(1,2,3,45,60))

a = lambda x,y,z : print(range(x+y+z))
print(a(1,2,3))