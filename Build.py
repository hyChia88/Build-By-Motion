'''
This is the building page that allows the user to build on a grid with blocks (cells)
'''
from cmu_graphics import *
import math
import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog as fd

# Take reference from Week7 assignment tetris3D
class Grid3D:
    def __init__(self, cellSize, gridSize):
        self.cellSize = cellSize
        self.gridSize = gridSize
        # it will be a cube
        self.board = [[[None for x in range(self.gridSize)]
                        for y in range(self.gridSize)]
                        for z in range(self.gridSize)]
        print(self.board) # for debug
    
    def __repr__(self):
        return f"Grid3D(cellSize={self.cellSize}, gridSize={self.gridSize})"
    
    # check if the position is valid, return True or False
    def isPosValid(self, cell):
        positionList = cell.getPlacementPos()
        if isinstance(cell, Cell):
            # check if the cell size is within the grid size
            if (0 <= len(positionList) <= self.gridSize):
                for pos in positionList:
                    # check if index within the bound/grid size
                    if not (0 <= pos[0] < self.gridSize and
                            0 <= pos[1] < self.gridSize and
                            0 <= pos[2] < self.gridSize):
                        print("pos is out of bound")
                        return False
                    # Then check if position is occupied
                    if self.board[pos[2]][pos[1]][pos[0]] is not None:
                        print("pos is occupied")
                        return False
            print("pos is valid")
            return True
        return False
    
    # place the cube at the position, return True or False
    def placeCell(self,cell):
        if self.isPosValid(cell):
            # the order of x,y,z is different from the order of the board
            positionList = cell.getPlacementPos()
            for pos in positionList:
                if not (0 <= pos[0] < self.gridSize and 
                        0 <= pos[1] < self.gridSize and 
                        0 <= pos[2] < self.gridSize):
                    return None
                self.board[pos[2]][pos[1]][pos[0]] = cell
            return True
        return False
    
    # get the cube at the position, return the cell or None
    def getCell(self, cell):
        positionList = cell.getPlacementPos()
        for pos in positionList:
            if not (0 <= pos[0] < self.gridSize and 
                    0 <= pos[1] < self.gridSize and 
                    0 <= pos[2] < self.gridSize):
                return None
            if self.board[pos[2]][pos[1]][pos[0]] is not None:
                return self.board[pos[2]][pos[1]][pos[0]]
        return None

    def removeCell(self, app, cell):
        if (0 <= cell.x < self.gridSize and 
            0 <= cell.y < self.gridSize and 
            0 <= cell.z < self.gridSize):
            cell = self.getCell(cell)
            if cell is not None:
                for pos in app.posListAll:
                    app.posListAll.remove(pos)
                    app.currentZ += 1
                return True
        return False

'''
Cell class, frac level I left it for future use
'''
class Cell:
    def __init__(self, x, y, z, fracLevel, app):
        self.fracLevel = fracLevel
        #  x,y,z here is position
        self.x = x
        self.y = y
        self.z = z
        self.size = 1
        self.fracLevel = fracLevel
        self.resizable = True
        self.pattern = [[[True for _ in range(self.size)] 
                            for _ in range(self.size)]
                            for _ in range(self.size)] # 1x1x1 cube
    
    def __repr__(self):
        return f"Cell({self.x}, {self.y}, {self.z}, {self.size})"

    def fracCell(self):
        self.x = self.x *2
        self.y = self.y *2
        self.z = self.z *2

    def getPattern(self):
        return self.pattern

    def isResizable(self):
        return self.resizable
    
    def resize(self, newSize):
        if not self.resizable:
            return False
        elif 1 <= newSize <= 3:
            self.size = newSize
            self.pattern = [[[True for _ in range(self.size)] 
                            for _ in range(self.size)]
                            for _ in range(self.size)]
            return True
        return False
    
    def getPlacementPos(self):
        cellPosList = []
        # get the exact pattern pos of cell, send to board and make it not None (occupied)
        if self.pattern is not None:
            pattern = self.getPattern()
            for x in range(len(pattern)):
                for y in range(len(pattern[x])):
                    for z in range(len(pattern[x][y])):
                        if pattern[x][y][z]:
                            pos = [self.x+x, self.y+y, self.z+z]
                            cellPosList.append(pos)
            # print("getPlacementPos success, cellPosList:")
            # print(cellPosList)
            return cellPosList
        else:
            print("pattern is None")
            return None
    
'''
Various types of cells, inherit from Cell class
'''
class LShapeCell(Cell):
    def __init__(self, x, y, z, fracLevel, app):
        super().__init__(x, y, z, fracLevel, app)
        self.resizable = False
        self.pattern = [[[True, False], [True, True]]] # Use False to represent None, so that at getPlacementPos, it will be ignored

class TShapeCell(Cell):
    def __init__(self, x, y, z, fracLevel, app):
        super().__init__(x, y, z, fracLevel, app)
        self.resizable = False
        self.pattern = [[[True, False, False], [True, True, True], [True, False, False]]]

class StairCell(Cell):
    def __init__(self, x, y, z, fracLevel, app):
        super().__init__(x, y, z, fracLevel, app)
        self.resizable = False
        self.pattern = [[[True, True], [True, False]],
                        [[True, True], [False, False]],
                        [[True, False], [False, False]]]

class ImageCell(Cell):
    def __init__(self, x, y, z, fracLevel, app):
        super().__init__(x, y, z, fracLevel, app)
        self.resizable = False

        # Generating 3dlist from Img
        image_path = "testEdge.jpg" #Hard code for now
        result = self.process_image(image_path)
        newArr = self.reMap(app.gridSize, app.gridSize, result['binary_bool'])
        
        self.pattern = newArr
        if self.pattern is not None:
            print("pattern is not None")
            print(self.pattern)
        # Hard code for debug:
        # self.pattern = [[[True, True, True, True, True],
        #                  [True, True, True, True, True],
        #                  [True, True, True, False, False],
        #                  [True, True, True, False, False],
        #                  [True, True, True, True, True]]]

    # def getPattern(self,app):
    #     super().getPattern()
    #     # Generating 3dlist from Img
    #     image_path = "testEdge.jpg" #Hard code for now
    #     result = self.process_image(image_path)
    #     newArr = self.reMap(app.gridSize, app.gridSize, result['binary_bool'])
        
    #     self.pattern = newArr
    #     if self.pattern is not None:
    #         print("ImageCell pattern is not None")
    #         print(self.pattern)
    #     return self.pattern

    def process_image(self,filename):
        # Load image directly in grayscale
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        # Check if image was successfully loaded
        if img is None:
            print("Error: cant load image" , filename)

        # Binarize using threshold
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Create a true/false numpy array for reference
        print(img)
        print(binary)
        bool_array = img > 20
        
        return {
            'grayscale': img,
            'binary_uint8': binary,
            'binary_bool': bool_array
        }

    def reMap(self,newH, newW, boolArray):
        oriH = len(boolArray)
        oriW = len(boolArray[0])
        
        # scale
        scaleH = oriH / newH
        scaleW = oriW / newW
        
        newArray = []
        print("the remap process:")
        for i in range(newH):
            newArray.append([])
            for j in range(newW):
                oriX = int(j * scaleW)
                oriY = int(i * scaleH)
                newArray[-1].append(boolArray[oriY][oriX])
        newArray = [newArray] # make 2d to 3d
        print("test array:")
        print(len(newArray))
        print(len(newArray[0]))
        print(newArray)
        return newArray

# Mainly reference from https://youtu.be/RRBXVu5UE-U?si=FTBWxNPHmmu-KmW6
class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.prevX = None
        self.prevY = None
        self.prevZ = None
        self.cacheX = None
        self.moveInZ = False
        self.counter = 0
        self.swipeThreshold = 0.05
        self.cap = cv2.VideoCapture(0)

    def detectGesture(self):
        ret, frame = self.cap.read()
        if not ret:
            return self.counter
            
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (480, 400))
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frameRGB)
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            h, w = frame.shape[:2]
            
            currentHandX = hand.landmark[8].x
            currentHandY = hand.landmark[8].y
            # Draw red circle on index finger
            indexX = int(currentHandX * w)
            indexY = int(currentHandY * h)
            cv2.circle(frame, (indexX, indexY), 10, (0, 0, 255), -1)
            
            if hand.landmark[8] and hand.landmark[12]:
                midFingerX = hand.landmark[12].x
                midFingerY = hand.landmark[12].y
                midIndexX = int(midFingerX * w)
                midIndexY = int(midFingerY * h)
                currentHandZ = hand.landmark[12].y
                if abs(midIndexX - indexX) < 50 and abs(midIndexY - indexY) < 50:
                    self.moveInZ = True
                    self.prevZ = currentHandZ
                    cv2.circle(frame,(midIndexX, midIndexY), 10, (0, 255, 0), -1)
                else:
                    self.moveInZ = False
                    
            self.cacheX = self.prevX
            self.prevX = currentHandX
            self.prevY = currentHandY
        else:
            self.prevX = None
            self.prevY = None
            self.prevZ = None
        cv2.imshow('Hand Gesture Counter', frame)
        # something like FPS 帧数
        cv2.waitKey(1)
        # Make sure the value is between 0 and 1
        if self.prevX is not None and self.prevY is not None:
            if self.prevX > 1:
                self.prevX = 1
            elif self.prevX < 0:
                self.prevX = 0
            elif self.prevY > 1:
                self.prevY = 1
            elif self.prevY < 0:
                self.prevY = 0
            if self.prevZ is not None:
                if self.prevZ > 1:
                    self.prevZ = 1
                elif self.prevZ < 0:
                    self.prevZ = 0
        return self.prevX, self.prevY, self.prevZ

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Reference Sudoku-3D\game3D.py, builder\CMU-15112-\isometric.py by Kui Yang Yang, 
# concept: https://skannai.medium.com/projecting-3d-points-into-a-2d-screen-58db65609f24 and modified it to fit my needs
# https://github.com/tcabezon/15112-hnx.py.git -> hnXfunction.py (line 74, twoDToIsometric(app,points), take the concept of proj 3d pts & but not using numpy, but not using cuz 3d rotation is different)
# Formula from below: Taking Y-axis rotation x X-axis rotation
# https://www.quora.com/How-do-you-convert-3D-coordinates-x-y-z-to-2D-coordinates-x-y
# https://en.wikipedia.org/wiki/Rotation_matrix#General_3D_rotations (Basic 3d rotation -> General 3d rotation)
# At the end, use 3d matrix rotation to rotate the 3d points, then project to 2d screen
class Projection3D:
    def __init__(self):
        pass

    # basic projection, isometric view starts from default rotX, rotY
    def basicProj(self, app, x, y, z, rotationY,rotationX):
        # Do Y-axis rotation then X-axis rotation R=R(X)R(Y)
        # scale and translate to screen coordinates
        
        rotX = x * math.cos(rotationY) - y * math.sin(rotationY)
        rotY = x * math.sin(rotationY) + y * math.cos(rotationY)
        
        finalY = rotY * math.cos(rotationX) - z * math.sin(rotationX)
        
        screenX = app.boardLeft + app.boardWidth*(2/3) + rotX * app.cellSize * app.scale
        screenY = app.boardTop + app.boardHeight/2 + finalY * app.cellSize * app.scale
        
        return (screenX, screenY)

class Draw:
    def __init__(self):
        self.cached_posList = []
        self.cached_outputPts = []
        self.cached_outputFaces = []

# Main method:
    def drawGridPlane(self, app, Projection3D):
        gridColor = rgb(200,200,200) #grey
        
        centerPt = Projection3D.basicProj(app, 0, 0, 0, app.rotationY, app.rotationX)
        endOfX = Projection3D.basicProj(app, app.gridSize, 0, 0, app.rotationY, app.rotationX)
        endOfY = Projection3D.basicProj(app, 0, app.gridSize, 0, app.rotationY, app.rotationX)
        endOfZ = Projection3D.basicProj(app, 0, 0, app.gridSize, app.rotationY, app.rotationX)
        
        drawLine(centerPt[0], centerPt[1], endOfX[0], endOfX[1], fill=gridColor, dashes=True)
        drawLine(centerPt[0], centerPt[1], endOfY[0], endOfY[1], fill=gridColor, dashes=True)
        drawLine(centerPt[0], centerPt[1], endOfZ[0], endOfZ[1], fill=gridColor, dashes=True)
        
        drawLabel(f'x', endOfX[0], endOfX[1], size=20)
        drawLabel(f'y', endOfY[0], endOfY[1], size=20)
        drawLabel(f'z', endOfZ[0], endOfZ[1], size=20)
        
        # Draw grid boundaries along X axis
        for i in range(app.gridSize + 1):
            startPt = Projection3D.basicProj(app, i, 0, 0, app.rotationY, app.rotationX)
            endPt = Projection3D.basicProj(app, i, app.gridSize, 0, app.rotationY, app.rotationX)
            drawLine(startPt[0], startPt[1], endPt[0], endPt[1], fill=gridColor, dashes=True)

        # Draw grid boundaries along Y axis
        for i in range(app.gridSize + 1):
            startPt = Projection3D.basicProj(app, 0, i, 0, app.rotationY, app.rotationX)
            endPt = Projection3D.basicProj(app, app.gridSize, i, 0, app.rotationY, app.rotationX)
            drawLine(startPt[0], startPt[1], endPt[0], endPt[1], fill=gridColor, dashes=True)

    '''
    Below are the methods for manipulating cells (which is self wrote)
    getCellPoints: get all points and faces of a cell
    cleanMesh: remove duplicate points and faces
    '''
    def getCellPoints(self, posList):
        # Template cube points and faces
        CUBE_POINTS = [
            [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1],
            [1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0],
        ]
        CUBE_FACES = [
            [0, 1, 2, 3], [3, 2, 4, 5], [5, 4, 6, 7],
            [7, 0, 3, 5], [7, 6, 1, 0], [6, 1, 2, 4],
        ]
        
        # Generate updated points after movement
        allPts = []
        for shift in posList:
            for point in CUBE_POINTS:
                # Add the shift to each point
                new_point = [
                    point[0] + shift[0],
                    point[1] + shift[1],
                    point[2] + shift[2]
                ]
                allPts.append(new_point)
        
        # Generate updated faces after movement
        allFaces = []
        offset = 0
        for shift in posList:
            # For each cube, add its faces with updated indices
            for face in CUBE_FACES:
                # Add the offset to each point index in the face
                new_face = [idx + offset for idx in face]
                allFaces.append(new_face)
            offset += len(CUBE_POINTS)  # offset is 8 for each subsequent cube
        
        return allPts, allFaces

    def getFacePts(self, inputPoints, inputFaces):
        face_centers = []

        for face in inputFaces:
            # Initialize center coordinates
            center = [0.0, 0.0, 0.0]
            
            # Sum up all points of the face
            for point_idx in face:
                point = inputPoints[point_idx]
                center[0] += point[0]
                center[1] += point[1]
                center[2] += point[2]
            
            # Divide by number of points to get average (center)
            num_points = len(face)
            center = [coord / num_points for coord in center]
            
            face_centers.append(tuple(center))

        return face_centers
    
    # Got cleanPts and cleanFaces from getCellPoints
    def cleanMesh(self, posList):
        """Clean mesh by removing duplicate points and faces."""
        allPts, allFaces = self.getCellPoints(posList)
        point_map = dict()
        cleaned_points = []
        old_to_new_idx = dict()
        
        # Process points
        i = 0
        for point in allPts:
            point_tuple = tuple(point)
            if point_tuple not in point_map:
                point_map[point_tuple] = len(cleaned_points)
                cleaned_points.append(point)
            old_to_new_idx[i] = point_map[point_tuple]
            i += 1
        
        # Clean faces and update indices
        face_centers = set()
        cleaned_faces = []
        
        for face in allFaces:
            # Update indices
            new_face = [old_to_new_idx[idx] for idx in face]
            # Check for duplicate faces
            center = self.getFacePts([allPts[idx] for idx in face], [[0,1,2,3]])[0]
            if center not in face_centers:
                face_centers.add(center)
                cleaned_faces.append(new_face)
        
        return cleaned_points, cleaned_faces
    
    '''
    get reference from wikipedia: https://en.wikipedia.org/wiki/Subdivision_surface
    https://rosettacode.org/wiki/Catmull%E2%80%93Clark_subdivision_surface
    from line 331 to 569, from getEdgeFace to getOutput in this file are mostly copied and also modified to this python file
    '''
    def getEdgeFaces(self, inputPoints, inputFaces):
        """
        Get list of edges and the one or two adjacent faces in a list.
        also get center point of edge
        Each edge would be [pointnum_1, pointnum_2, facenum_1, facenum_2, center]
        """
        
        # will have [pointnum_1, pointnum_2, facenum]
        
        edges = []
        
        # get edges from each face
        
        for facenum in range(len(inputFaces)):
            face = inputFaces[facenum]
            num_points = len(face)
            # loop over index into face
            for pointindex in range(num_points):
                # if not last point then edge is curr point and next point
                if pointindex < num_points - 1:
                    pointnum_1 = face[pointindex]
                    pointnum_2 = face[pointindex+1]
                else:
                    # for last point edge is curr point and first point
                    pointnum_1 = face[pointindex]
                    pointnum_2 = face[0]
                # order points in edge by lowest point number
                if pointnum_1 > pointnum_2:
                    temp = pointnum_1
                    pointnum_1 = pointnum_2
                    pointnum_2 = temp
                edges.append([pointnum_1, pointnum_2, facenum])
                
        # sort edges by pointnum_1, pointnum_2, facenum
        
        edges = sorted(edges)
        
        # merge edges with 2 adjacent faces
        # [pointnum_1, pointnum_2, facenum_1, facenum_2] or
        # [pointnum_1, pointnum_2, facenum_1, None]
        
        num_edges = len(edges)
        eindex = 0
        merged_edges = []
        
        while eindex < num_edges:
            e1 = edges[eindex]
            # check if not last edge
            if eindex < num_edges - 1:
                e2 = edges[eindex+1]
                if e1[0] == e2[0] and e1[1] == e2[1]:
                    merged_edges.append([e1[0],e1[1],e1[2],e2[2]])
                    eindex += 2
                else:
                    merged_edges.append([e1[0],e1[1],e1[2],None])
                    eindex += 1
            else:
                merged_edges.append([e1[0],e1[1],e1[2],None])
                eindex += 1
                
        # add edge centers
        
        edges_centers = []
        
        for me in merged_edges:
            p1 = inputPoints[me[0]]
            p2 = inputPoints[me[1]]
            cp = [ (p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2 ]
            edges_centers.append(me+[cp])
                
        return edges_centers
        
    '''
    Set each edge point to be the average of the two neighbouring face points (A,F) and the two endpoints of the edge (M,E)
    '''
    def getEdgePoints(self, inputPoints, edgeFace, faceCenter):
        edgePoints = []
        
        for edge in edgeFace:
            # (M+E)/2
            cp = edge[4]

            fp1 = faceCenter[edge[2]]
            # if not two faces just use one facepoint
            # should not happen for solid like a cube
            if edge[3] == None:
                fp2 = fp1
            else:
                fp2 = faceCenter[edge[3]]

            # (A+F)/2
            cfp = (fp1[0] + fp2[0])/2, (fp1[1] + fp2[1])/2, (fp1[2] + fp2[2])/2
            # (A+F+M+E)/4
            edgePoint = (cp[0] + cfp[0])/2, (cp[1] + cfp[1])/2, (cp[2] + cfp[2])/2
            edgePoints.append(edgePoint)
            
        return edgePoints
    
    def sumPoints(self, p1, p2):
        sp = []
        for i in range(3):
            sp.append(p1[i] + p2[i])
        return sp

    def mulPoint(self, p, m):
        mp = []
        for i in range(3):
            mp.append(p[i]*m)
        return mp

    def get_avg_face_points(self, input_points, input_faces, face_points):
        num_points = len(input_points)
        
        temp_points = []
        
        for pointnum in range(num_points):
            temp_points.append([[0.0, 0.0, 0.0], 0])
        
        for facenum in range(len(input_faces)):
            fp = face_points[facenum]
            for pointnum in input_faces[facenum]:
                tp = temp_points[pointnum][0]
                temp_points[pointnum][0] = self.sumPoints(tp,fp)
                temp_points[pointnum][1] += 1
                
        avg_face_points = []
        
        for tp in temp_points:
            if tp[1] > 0:
                afp = [tp[0][i]/tp[1] for i in range(3)]
            else:
                afp = tp[0]
            avg_face_points.append(afp)
            
        return avg_face_points
        
    def get_avg_mid_edges(self, input_points, edges_faces):
        num_points = len(input_points)
        
        temp_points = []
        
        for pointnum in range(num_points):
            temp_points.append([[0.0, 0.0, 0.0], 0])
            
        for edge in edges_faces:
            cp = edge[4]
            for pointnum in [edge[0], edge[1]]:
                tp = temp_points[pointnum][0]
                temp_points[pointnum][0] = self.sumPoints(tp,cp)
                temp_points[pointnum][1] += 1
        
        avg_mid_edges = []
            
        for tp in temp_points:
            if tp[1] > 0:
                ame = [tp[0][i]/tp[1] for i in range(3)]
            else:
                ame = tp[0]
            avg_mid_edges.append(ame)
        
        return avg_mid_edges

    def get_points_faces(self, input_points, input_faces):
        # initialize list with 0
        
        num_points = len(input_points)
        
        points_faces = []
        
        for pointnum in range(num_points):
            points_faces.append(0)
            
        # loop through faces updating points_faces
        
        for facenum in range(len(input_faces)):
            for pointnum in input_faces[facenum]:
                points_faces[pointnum] += 1
                
        return points_faces

    '''
    F: face center; R: edge center; P: vertex point; n: number of faces
    barycenter of P, R and F with respective weights (n − 3), 2 and 1
    '''
    def get_new_points(self, input_points, points_faces, avg_face_points, avg_mid_edges):
        new_points =[]
        
        for pointnum in range(len(input_points)):
            n = points_faces[pointnum]
            if n > 0:
                m1 = (n - 3.0) / n
                m2 = 1.0 / n
                m3 = 2.0 / n
            else:
                m1 = (n - 3.0)
                m2 = 1.0 
                m3 = 2.0
            old_coords = input_points[pointnum]
            p1 = self.mulPoint(old_coords, m1)
            afp = avg_face_points[pointnum]
            p2 = self.mulPoint(afp, m2)
            ame = avg_mid_edges[pointnum]
            p3 = self.mulPoint(ame, m3)
            p4 = self.sumPoints(p1, p2)
            new_coords = self.sumPoints(p4, p3)
            
            new_points.append(new_coords)
            
        return new_points

    def switch_nums(self,point_nums):
        if point_nums[0] < point_nums[1]:
            return point_nums
        else:
            return (point_nums[1], point_nums[0])

    def cmc_subdiv(self, input_points, input_faces):
        # input_points and input_faces are the cleaned points and faces
        face_points = self.getFacePts(input_points, input_faces)
        edges_faces = self.getEdgeFaces(input_points, input_faces)
        edge_points = self.getEdgePoints(input_points, edges_faces, face_points)
        avg_face_points = self.get_avg_face_points(input_points, input_faces, face_points)
        avg_mid_edges = self.get_avg_mid_edges(input_points, edges_faces) 
        points_faces = self.get_points_faces(input_points, input_faces)
        
        """
        m1 = (n - 3) / n
        m2 = 1 / n
        m3 = 2 / n
        new_coords = (m1 * old_coords)
                + (m2 * avg_face_points)
                + (m3 * avg_mid_edges)
        """
        new_points = self.get_new_points(input_points, points_faces, avg_face_points, avg_mid_edges)
        
        face_point_nums = []
        
        next_pointnum = len(new_points)
        
        for face_point in face_points:
            new_points.append(face_point)
            face_point_nums.append(next_pointnum)
            next_pointnum += 1
        
        edge_point_nums = dict()
        
        for edgenum in range(len(edges_faces)):
            pointnum_1 = edges_faces[edgenum][0]
            pointnum_2 = edges_faces[edgenum][1]
            edge_point = edge_points[edgenum]
            new_points.append(edge_point)
            edge_point_nums[(pointnum_1, pointnum_2)] = next_pointnum
            next_pointnum += 1

        new_faces =[]
        
        for oldfacenum in range(len(input_faces)):
            oldface = input_faces[oldfacenum]
            # 4 point face
            if len(oldface) == 4:
                a = oldface[0]
                b = oldface[1]
                c = oldface[2]
                d = oldface[3]
                face_point_abcd = face_point_nums[oldfacenum]
                edge_point_ab = edge_point_nums[self.switch_nums((a, b))]
                edge_point_da = edge_point_nums[self.switch_nums((d, a))]
                edge_point_bc = edge_point_nums[self.switch_nums((b, c))]
                edge_point_cd = edge_point_nums[self.switch_nums((c, d))]
                new_faces.append((a, edge_point_ab, face_point_abcd, edge_point_da))
                new_faces.append((b, edge_point_bc, face_point_abcd, edge_point_ab))
                new_faces.append((c, edge_point_cd, face_point_abcd, edge_point_bc))
                new_faces.append((d, edge_point_da, face_point_abcd, edge_point_cd))
        
        return new_points, new_faces

    # Final output to draw, pass in to projection to draw
    def getOutput(self, posList, subdLvl, app):
        # if not (app.handCountX or app.handCountY):
        #     # print("pause here")
        #     return self.cached_outputPts, self.cached_outputFaces
        # else:
        # print("run getOutput")
        self.cached_posList = posList
        cleaned_points, cleaned_faces = self.cleanMesh(posList)

        # print(f"cleaned_points ({len(cleaned_points)}):", cleaned_points)
        # print(f"cleaned_faces ({len(cleaned_faces)}):", cleaned_faces)
        
        iterations = subdLvl
        output_points, output_faces = cleaned_points, cleaned_faces

        for i in range(iterations):
            output_points, output_faces = self.cmc_subdiv(output_points, output_faces)

        # save the cache
        self.cached_outputPts, self.cached_outputFaces = output_points, output_faces
        return output_points, output_faces

def drawCell(app, posList, isConfirmed):
    allpts, allfaces = app.draw.getOutput(posList, app.subdLvl,app)
    # First project all points
    projectedpts = []
    for point in allpts:
        projectedpts.append(app.projection.basicProj(app, point[0], point[1], point[2], app.rotationY, app.rotationX))

    # Draw the connecting lines for each face
    for face in allfaces:
        # Connect each point in the face to the next point
        # # there are 4ptsin a face [p1,p2,p3,p4]
        # for i in range(len(face)):
        #     # Get current point and next point (wrapping around to first point)
        #     pt1 = projectedpts[face[i]]
        #     pt2 = projectedpts[face[(i + 1) % len(face)]]
            
        #     # Draw line between the points
        # drawLine(pt1[0], pt1[1], pt2[0], pt2[1], fill='grey' if isConfirmed else 'blue')
        # drawLine(pt1[0], pt1[1], pt2[0], pt2[1], fill='grey' if isConfirmed else 'blue')
        
        pt1 = projectedpts[face[0]]
        pt2 = projectedpts[face[1]]
        pt3 = projectedpts[face[2]]
        pt4 = projectedpts[face[3]]
        
        # # drawLine(pt1[0], pt1[1], pt2[0], pt2[1], fill='grey' if isConfirmed else 'blue')
        # drawLine(pt2[0], pt2[1], pt3[0], pt3[1], fill='grey' if isConfirmed else 'blue')
        # # drawLine(pt3[0], pt3[1], pt4[0], pt4[1], fill='grey' if isConfirmed else 'blue')
        # drawLine(pt4[0], pt4[1], pt1[0], pt1[1], fill='grey' if isConfirmed else 'blue')
        
        drawPolygon(pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1], pt4[0], pt4[1],pt1[0], pt1[1], fill = "white", border ='grey' if isConfirmed else 'blue',borderWidth =1)

    # if not isConfirmed:
    #     # Draw points on top of lines
    #     for projPt in projectedpts:
    #         drawCircle(projPt[0], projPt[1], 2, fill='red')

# Draw all on the grid
def drawGrid(app, posList, isConfirmed):
    if not isConfirmed:
        app.draw.drawGridPlane(app, app.projection)

    app.cell.x = app.currentX
    app.cell.y = app.currentY
    app.cell.z = app.currentZ

    drawCell(app, posList, isConfirmed)

# Import image, the logic here is using tkinter to open the file dialog and get the file path, 
# then assign the file path to app.image
# Cite: 
# https://academy.cs.cmu.edu/cpcs-docs/images_and_sounds
# https://www.geeksforgeeks.org/how-to-specify-the-file-path-in-a-tkinter-filedialog/
# my own project while hack112: https://github.com/hyChia88/hack112.git 
def importImage(app):
    root = tk.Tk()
    root.withdraw()
    try:
        filetypes = (('text files', '*.jpg *.jpeg *.png *.bmp *.gif *.tiff'), 
            ('All files', '*.*')) 
        
        f = fd.askopenfile(filetypes=filetypes, initialdir="D:/Downloads") 
        if f:
            print(f"filePaths:", f.name)
            app.image = f.name
            
    except Exception as e:
        print(f"Error importing image: {e}")
    finally:
        root.destroy()

def init(app):
    app.projection = Projection3D()
    app.draw = Draw()
    
    #board szie
    app.boardLeft = 300
    app.boardTop = 100
    app.boardWidth = 800
    app.boardHeight = 600

    # init the current cell
    app.currentX = 0
    app.currentY = 0
    app.currentZ = 0

    app.currentPosList = [[app.currentX, app.currentY, app.currentZ]]
    app.posListAll = []
    app.lastValidX = 0
    app.lastValidY = 0
    app.gridScale = 1
    app.gridSize = 7
    app.cellSize = 50
    app.newSize = 1
    app.angle = 30
    app.fracLevel = 1
    app.subdLvl = 0
    
    app.rotationY = math.pi/4 # 45 degree
    app.rotationX = math.pi/6 # 30 degree
    app.dragging = False
    app.lastMouseX = 0
    app.lastMouseY = 0
    app.scale = 1
    
    # init the cell, start with default
    app.cell = Cell(app.currentX, app.currentY, app.currentZ, app.fracLevel, app)
    app.grid = Grid3D(app.cellSize, app.gridSize)
    
    #hand gesture starting 
    app.detector = HandGestureDetector()
    app.handCountX = 0
    app.handCountY = 0
    app.handCountZ = 0
    # show centers
    app.showSubd = False
    
    # import the image
    app.image = None
    app.frameImgX = 80
    app.frameImgSize = 250
    app.buttonSize = 30

    # check if the current cell is valid
    app.isPosValid = app.grid.isPosValid(app.cell)
    
    # Draw error hint:
    app.hint = None

def onAppStart(app):
    app.setMaxShapeCount(5000)
    init(app)
    
def onAppStop(app):
    app.detector.cleanup()

def onMousePress(app, mouseX, mouseY):
    app.dragging = True
    app.lastMouseX = mouseX
    app.lastMouseY = mouseY
    if (app.frameImgX + app.frameImgSize/2 - app.buttonSize/2 < mouseX < app.frameImgX + app.frameImgSize/2 + app.buttonSize/2 and
        app.height/2 + app.frameImgSize/2 + app.buttonSize/2 < mouseY < app.height/2 + app.frameImgSize/2 + app.buttonSize*2):
        # app.image = 'C:/Users/huiye/Downloads/—Pngtree—vector forward icon_4184777.png' # for testing
        importImage(app)

def onMouseDrag(app, mouseX, mouseY):
    if app.dragging:
        dx = mouseX - app.lastMouseX
        dy = mouseY - app.lastMouseY
        app.rotationY += dx * 0.01
        app.rotationX += dy * 0.01
        # app.rotationX = app.rotationX % (2 * math.pi)
        if app.rotationX > math.pi/2:
            app.rotationX = math.pi/2
        elif app.rotationX < -math.pi/2:
            app.rotationX = -math.pi/2
        app.lastMouseX = mouseX
        app.lastMouseY = mouseY

def onMouseRelease(app, mouseX, mouseY):
    app.dragging = False

def onKeyPress(app, key):
    # Cube related functions
    if key == 'space':
        # Debug prints to see what's happening
        print("Placing cell!")
        print("Current app.cell:", app.cell, " pattern:", app.cell.pattern, " Cell positions:", app.cell.getPlacementPos())

        if (app.grid.isPosValid(app.cell) and
            app.grid.getCell(app.cell) is None):
            app.posListAll.extend(app.cell.getPlacementPos())
            # app.posListAll.extend(app.currentPosList)
            print(f"app.cell.getPlacementPos() ({len(app.cell.getPlacementPos())}):", app.cell.getPlacementPos())
            print(f"app.currentPosList ({len(app.currentPosList)}):", app.currentPosList)
            print(f"app.posListAll ({len(app.posListAll)}):", app.posListAll)

            placeable = app.grid.placeCell(app.cell)
            if placeable:
                if app.currentZ >= app.gridSize - 1:
                    app.currentZ = 0
                else:
                    app.currentZ += 1
                app.cell = Cell(app.currentX, app.currentY, app.currentZ, app.fracLevel, app)
        
        # Pretty print the board layer by layer
        print("\nUpdated board:")
        for z in range(len(app.grid.board)):
            print(f"Layer: {z}:")
            for y in range(len(app.grid.board[z])):
                row = [cell is not None for cell in app.grid.board[z][y]]
                print(''.join(['X' if cell else '.' for cell in row]))
            
    elif key == 'd':
        currPos = app.cell.getPlacementPos()
        for pos in currPos:
            if pos in app.posListAll:
                # app.posListAll.remove(pos)
                app.grid.removeCell(app, app.cell)
            print("removed cube at:", app.cell.getPlacementPos())
        else:
            print("cube not found at:", app.cell.getPlacementPos())

    elif key == '6':
        if isinstance(app.cell, Cell) and app.cell.resizable and app.newSize < 3:
            app.newSize += 1
            app.cell.resize(app.newSize)

    elif key == '5':
        if isinstance(app.cell, Cell) and app.cell.resizable and app.newSize > 1:
            app.newSize -= 1
            app.cell.resize(app.newSize)

    # Change the grid size
    elif key == 'up':
        if app.gridSize < 32:
            app.gridSize = app.gridSize *2
            app.grid = Grid3D(app.cellSize, app.gridSize)
            print(f"gridSize to: {app.gridSize}") 

    elif key == 'down':
        if app.gridSize > 1:
            app.gridSize = app.gridSize //2
            app.grid = Grid3D(app.cellSize, app.gridSize)
            print(f"gridSize to: {app.gridSize}")
            
    # Reset the game
    elif key == 'r':
        onAppStart(app)

    # Show subdivision
    elif key == 's':
        app.showSubd = not app.showSubd
    elif key == ']':
        if app.subdLvl == 2:
            app.subdLvl = 2
        else:
            app.subdLvl += 1
        print(f"subdLvl to: {app.subdLvl}") 
    elif key == '[':
        if app.subdLvl == 0:
            app.subdLvl = 0
        else:
            app.subdLvl -= 1
        print(f"subdLvl to: {app.subdLvl}")  
    elif key == 'left':
        app.scale -= 0.1
    elif key == 'right':
        app.scale += 0.1
        
    # Change the cell type
    if key in ['1', '2', '3', '4','x','X']:
        if key == '1':
            app.cell = Cell(app.currentX, app.currentY, app.currentZ, app.fracLevel, app)
        elif key == '2':
            app.cell = LShapeCell(app.currentX, app.currentY, app.currentZ, app.fracLevel, app)
        elif key == '3':
            app.cell = TShapeCell(app.currentX, app.currentY, app.currentZ, app.fracLevel, app)
        elif key == '4':
            app.cell = StairCell(app.currentX, app.currentY, app.currentZ, app.fracLevel, app)
        elif key == 'x' or 'X':
            print('imageCell try!')
            app.cell = ImageCell(app.currentX, app.currentY, app.currentZ, app.fracLevel, app)

def onStep(app):
    # it will be x & y
    app.handCountX, app.handCountY, app.handCountZ = app.detector.detectGesture()
    if app.handCountZ:
        # Use the last valid X/Y positions when moving Z
        app.currentX = app.lastValidX
        app.currentY = app.lastValidY
        mappedZ = int(app.handCountZ * (app.gridSize))
        app.currentZ = app.grid.gridSize - mappedZ
    elif app.handCountX is not None and app.handCountY is not None:
        # Update X/Y positions and store them as last valid positions
        mappedX = int(app.handCountX * (app.gridSize-1))
        mappedY = int(app.handCountY * (app.gridSize-1))
        app.currentX = mappedX
        app.currentY = mappedY
        app.lastValidX = mappedX
        app.lastValidY = mappedY

def redrawAll(app):
    # title
    drawLabel('Build Game',app.width/2, 20, size=24)
    
    # Main drawing
    # problem here : keep calling the drawGrid
    drawGrid(app, app.posListAll, True)
    drawGrid(app, app.cell.getPlacementPos(), False)
    
    # Instructions section
    instructionY = 60
    spacing = 15
    
    # Movement controls
    drawLabel('Controls:', app.width/2, instructionY, size=15, bold=True)
    drawLabel('• Use hand gestures to move the cube in X/Y plane', app.width/2, instructionY + spacing,size=12)
    drawLabel('• Hold index and middle fingers together to move in Z axis', app.width/2, instructionY + spacing*2,size=12)
    
    # Building controls
    drawLabel('Building:', app.width/2, instructionY + spacing*4, size=15, bold=True)
    drawLabel('• SPACE: Place cube', app.width/2, instructionY + spacing*5,size=12)
    drawLabel('• 1-4: Change block type (1:Default, 2:L-Shape, 3:T-Shape, 4:Stair)', app.width/2, instructionY + spacing*6,size=12)
    drawLabel('• Q/E: Increase/Decrease cube size', app.width/2, instructionY + spacing*7,size=12)
    
    # View controls
    drawLabel('View Controls:', app.width/2, instructionY + spacing*9, size=15, bold=True)
    drawLabel('• Drag mouse: Rotate view', app.width/2, instructionY + spacing*10,size=12)
    drawLabel('• Left/Right arrows: Zoom in/out', app.width/2, instructionY + spacing*11,size=12)
    drawLabel('• Up/Down arrows: Change grid size', app.width/2, instructionY + spacing*12,size=12)
    
    # Special features
    drawLabel('Special Features:', app.width/2, instructionY + spacing*14, size=15, bold=True)
    drawLabel('• d: delete current cell', app.width/2, instructionY + spacing*15,size=12)
    drawLabel('• [ ]: Adjust subdivide level (current: ' + str(app.fracLevel) + ')', app.width/2, instructionY + spacing*16,size=12)
    drawLabel('• R: Reset game', app.width/2, instructionY + spacing*17,size=12)
    
    # Current position and hand detection status
    drawLabel(f'Current Position: ({app.currentX}, {app.currentY}, {app.currentZ})', 
             app.width/2, app.height - spacing*2,size=12)
    
    if app.hint:
        drawLabel('Hint!', app.width/2, app.height - spacing*3,size=12, fill='red')
    
    if app.handCountX or app.handCountY:
        drawLabel(f'Hand Position: ({pythonRound(app.handCountX, 2)}, {pythonRound(app.handCountY, 2)})', 
                 app.width/2, app.height - spacing,size=12)
    else:
        drawLabel('Hand not detected - Move your hand to control the cube', 
                 app.width/2, app.height - spacing,size=12, fill='red')
    
    # import the image 
    if app.image is not None:
        drawImage(app.image, app.frameImgX, app.height/2-app.frameImgSize/2, width=app.frameImgSize, height=app.frameImgSize)
    drawRect(app.frameImgX, app.height/2-app.frameImgSize/2, app.frameImgSize, app.frameImgSize, fill=None, border='black')
    drawImage('importIcon.png', app.frameImgX + app.frameImgSize/2 - app.buttonSize/2, app.height/2 + app.frameImgSize/2 + app.buttonSize, width=app.buttonSize, height=app.buttonSize)

def build():
    print("build start!")
    runApp(width=1200, height=750)
    
build()