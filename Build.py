'''
This is the building page that allows the user to build on a grid with blocks (cells)
'''
from re import L
import numpy as np
from cmu_graphics import *
import math
import cv2
# import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog

from getFacePts import clean_mesh

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
            for pos in positionList:
                # Check bounds first
                if not (0 <= pos[0] < self.gridSize and
                       0 <= pos[1] < self.gridSize and
                       0 <= pos[2] < self.gridSize):
                    return False
                # Then check if position is occupied
                if self.board[pos[2]][pos[1]][pos[0]] is not None:
                    return False
            return True
        return False
    
    # place the cube at the position, return True or False
    def placeCell(self,cell):
        if self.isPosValid(cell):
            # the order of x,y,z is different from the order of the board
            positionList = cell.getPlacementPos()
            for pos in positionList:
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

    def removeCell(self, cell):
        if 0 <= cell.x < self.gridSize and 0 <= cell.y < self.gridSize and 0 <= cell.z < self.gridSize:
            cell = self.getCell(cell)
            if cell is not None:
                positionList = cell.getPlacementPos()
                for pos in positionList:
                    self.board[pos[2]][pos[1]][pos[0]] = None
                return True
        return False

class CellFactory:
    @staticmethod
    def createCell(cellType, x, y, z, fracLevel=1):
        cell_types = {
            'default': Cell,
            'L': LShapeCell,
            'T': TShapeCell,
            'stair': StairCell
        }
        CellClass = cell_types.get(cellType)
        if CellClass:
            return CellClass(x, y, z, fracLevel)
        raise ValueError(f"Unknown cell type: {cellType}")

'''
Cell class, frac level I left it for future use
'''
class Cell:
    def __init__(self, x, y, z, fracLevel):
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

    '''return a list of (x,y,z), get the exact pattern pos of cell, send to board and make it not None (occupied)'''
    def getPlacementPos(self):
        posList = []
        # get the exact pattern pos of cell, send to board and make it not None (occupied)
        pattern = self.getPattern()
        for x in range(len(pattern)):
            for y in range(len(pattern[x])):
                for z in range(len(pattern[x][y])):
                    if pattern[x][y][z] is True:
                        pos = (self.x+x, self.y+y, self.z+z)
                        posList.append(pos)
        return posList
    
'''
Various types of cells, inherit from Cell class
'''
class LShapeCell(Cell):
    def __init__(self, x, y, z, fracLevel):
        super().__init__(x, y, z, fracLevel)
        self.resizable = False
        self.pattern = [[[True, False], [True, True]]] # Use False to represent None, so that at getPlacementPos, it will be ignored

class TShapeCell(Cell):
    def __init__(self, x, y, z, fracLevel):
        super().__init__(x, y, z, fracLevel)
        self.resizable = False
        self.pattern = [[[True, False, False], [True, True, True], [True, False, False]]]
        
class LineCell(Cell):
    def __init__(self, x, y, z, fracLevel):
        super().__init__(x, y, z, fracLevel)
        self.resizable = False
        self.pattern = [[[True, True, True]]]

class StairCell(Cell):
    def __init__(self, x, y, z, fracLevel=1):
        super().__init__(x, y, z, fracLevel)
        self.resizable = False
        self.pattern = [[[True, True], [True, False]],
                        [[True, True], [False, False]],
                        [[True, False], [False, False]]]

# Mainly reference from https://youtu.be/RRBXVu5UE-U?si=FTBWxNPHmmu-KmW6
class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.prevX = None
        self.prevY = None
        self.prevZ = None
        self.moveInZ = False
        self.counter = 0
        self.swipeThreshold = 0.05
        self.cap = cv2.VideoCapture(0)

    def detectGesture(self):
        ret, frame = self.cap.read()
        if not ret:
            return self.counter
            
        frame = cv2.flip(frame, 1)
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

# Reference Sudoku-3D\game3D.py, builder\CMU-15112-\isometric.py by Kui Yang Yang, https://skannai.medium.com/projecting-3d-points-into-a-2d-screen-58db65609f24 and modified it to fit my needs
class Projection3D:
    def __init__(self):
        pass

    def basicProj(self, app, x, y, z, rotationY,rotationX):
        '''
        Do Y-axis rotation first, then X-axis rotation
        scale and translate to screen coordinates
        '''
        rotX = x * math.cos(rotationY) - y * math.sin(rotationY)
        rotY = x * math.sin(rotationY) + y * math.cos(rotationY)
        
        finalY = rotY * math.cos(rotationX) - z * math.sin(rotationX)
        finalZ = rotY * math.sin(rotationX) + z * math.cos(rotationX)
        
        screenX = app.boardLeft + app.boardWidth/2 + rotX * app.cellSize * app.scale
        screenY = app.boardTop + app.boardHeight/2 + finalY * app.cellSize * app.scale
        
        return (screenX, screenY, finalZ)

class Draw:
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
        for i, shift in enumerate(posList):
            # For each cube, add its faces with updated indices
            offset = i * len(CUBE_POINTS)  # offset is 8 for each subsequent cube
            for face in CUBE_FACES:
                # Add the offset to each point index in the face
                new_face = [idx + offset for idx in face]
                allFaces.append(new_face)
        
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
        for i, point in enumerate(allPts):
            point_tuple = tuple(point)
            if point_tuple not in point_map:
                point_map[point_tuple] = len(cleaned_points)
                cleaned_points.append(point)
            old_to_new_idx[i] = point_map[point_tuple]
        
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
    from line 331 to 569 copied and modified to this python file
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
            afp = [tp[0][i]/tp[1] for i in range(3)]
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
            ame = [tp[0][i]/tp[1] for i in range(3)]
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
            m1 = (n - 3.0) / n
            m2 = 1.0 / n
            m3 = 2.0 / n
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
    def getOutput(self, posList):
        cleaned_points, cleaned_faces = self.cleanMesh(posList)
        print(f"cleaned_points ({len(cleaned_points)}):", cleaned_points)
        print(f"cleaned_faces ({len(cleaned_faces)}):", cleaned_faces)
        
        iterations = int(1)
        output_points, output_faces = cleaned_points, cleaned_faces

        for i in range(iterations):
            output_points, output_faces = self.cmc_subdiv(output_points, output_faces)

        return output_points, output_faces

def drawCell(app, posList):
    output_points, output_faces = app.draw.getOutput(posList)
    # First project all points
    projectedpts = []
    for point in output_points:
        projectedpts.append(app.projection.basicProj(app, point[0], point[1], point[2], app.rotationY, app.rotationX))

    # Draw the connecting lines for each face
    for face in output_faces:
        # Connect each point in the face to the next point
        for i in range(len(face)):
            # Get current point and next point (wrapping around to first point)
            pt1 = projectedpts[face[i]]
            pt2 = projectedpts[face[(i + 1) % len(face)]]
            
            # Draw line between the points
            drawLine(pt1[0], pt1[1], pt2[0], pt2[1], fill='blue')

    # Draw points on top of lines
    for projPt in projectedpts:
        drawCircle(projPt[0], projPt[1], 2, fill='red')

# Draw all on the grid
def drawGrid(app, posList):
    app.draw.drawGridPlane(app, app.projection)
    cubesToDraw = []
    
    for z in range(app.grid.gridSize):
        for y in range(app.grid.gridSize):
            for x in range(app.grid.gridSize):
                existCell = app.grid.board[z][y][x]
                if existCell is not None:
                    # draw the cell at the center of the cube
                    depth = app.projection.basicProj(app, x+0.5, y+0.5, z+0.5, app.rotationY, app.rotationX)[2]
                    cubesToDraw.append((depth, existCell, False))
                    
    app.cell.x = app.currentX
    app.cell.y = app.currentY
    app.cell.z = app.currentZ
    print(f"cubesToDraw ({len(cubesToDraw)}):", cubesToDraw)
    
    # from cubesToDraw, get the output points and faces
    # output_points, output_faces = app.draw.getOutput(posList)
    drawCell(app, posList)

def importImage(app):
    root = tk.Tk()
    root.withdraw()
    try:
        filePaths = filedialog.askopenfilenames(title='Select Image', 
                                                filetypes=[('Image Files', '*.jpg *.jpeg *.png *.bmp *.gif *.tiff')])
        if filePaths:
            app.image = filePaths[0]
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
    
    app.currentPos = [[app.currentX, app.currentY, app.currentZ]]
    
    app.posListAll = []
    app.lastValidX = 0
    app.lastValidY = 0
    app.gridScale = 1
    app.gridSize = 5
    app.cellSize = 50
    app.newSize = 1
    app.angle = 30
    app.fracLevel = 1
    
    app.rotationY = math.pi/4 # 45 degree
    app.rotationX = math.pi/6 # 30 degree
    app.dragging = False
    app.lastMouseX = 0
    app.lastMouseY = 0
    app.scale = 1
    
    # init the cell
    app.cellType = 'default'
    app.cellFactory = CellFactory()
    app.cell = Cell(app.currentX, app.currentY, app.currentZ, app.fracLevel)
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

def onAppStart(app):
    init(app)
    
def onAppStop(app):
    app.detector.cleanup()

def onMousePress(app, mouseX, mouseY):
    app.dragging = True
    app.lastMouseX = mouseX
    app.lastMouseY = mouseY
    if (app.frameImgX + app.frameImgSize/2 - app.buttonSize/2 < mouseX < app.frameImgX + app.frameImgSize/2 + app.buttonSize/2 and
        app.height/2 + app.frameImgSize/2 + app.buttonSize/2 < mouseY < app.height/2 + app.frameImgSize/2 + app.buttonSize*2):
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
        print("Current app.cell:", app.cell)
        print("Cell pattern:", app.cell.pattern)
        print("Cell positions:", app.cell.getPlacementPos())
        
        if (app.grid.isPosValid(app.cell) and
            app.grid.getCell(app.cell) is None):
            app.posListAll.extend(app.cell.getPlacementPos())
            print(app.cell.getPlacementPos())
            print(f"space -> app.posListAll ({len(app.posListAll)}):", app.posListAll)
            placeable = app.grid.placeCell(app.cell)
            print("Can place, Placed cell at:", app.cell.x, app.cell.y, app.cell.z) 
            if placeable:
                if app.currentZ >= app.gridSize - 1:
                    app.currentZ = 0
                else:
                    app.currentZ += 1
                app.cell = app.cellFactory.createCell(app.cellType, app.currentX, app.currentY, app.currentZ, app.fracLevel)
        
        # Pretty print the board layer by layer
        print("\nUpdated board:")
        for z in range(len(app.grid.board)):
            print(f"Layer: {z}:")
            for y in range(len(app.grid.board[z])):
                row = [cell is not None for cell in app.grid.board[z][y]]
                print(''.join(['X' if cell else '.' for cell in row]))
            
    elif key == 'd':
        if isinstance(app.cell, Cell) and app.cell.resizable and app.newSize > 1:
            app.newSize -= 1
            print("removed cube at:", app.currentX, app.currentY, app.currentZ)

    elif key == 'q':
        if isinstance(app.cell, Cell) and app.cell.resizable and app.newSize < 3:
            app.newSize += 1
            app.cell.resize(app.newSize)

    elif key == 'e':
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
        if app.fracLevel == 3:
            app.fracLevel = 3
        else:
            app.fracLevel += 1
        print(f"fracLevel to: {app.fracLevel}") 
    elif key == '[':
        if app.fracLevel == 1:
            app.fracLevel = 1
        else:
            app.fracLevel -= 1
        print(f"fracLevel to: {app.fracLevel}")  
    elif key == 'left':
        app.scale -= 0.1
    elif key == 'right':
        app.scale += 0.1
        
    # Change the cell type
    if key in ['1', '2', '3', '4']:
        if key == '1': app.cellType = 'default'
        elif key == '2': app.cellType = 'L'
        elif key == '3': app.cellType = 'T'
        elif key == '4': app.cellType = 'stair'
        # Create new cell with updated type
        app.cell = app.cellFactory.createCell(app.cellType, app.currentX, app.currentY, app.currentZ, app.fracLevel)

def onStep(app):
    # it will be x & y
    app.handCountX, app.handCountY, app.handCountZ = app.detector.detectGesture()
    if app.handCountZ:
        # Use the last valid X/Y positions when moving in Z
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
        app.currentPos = [[app.currentX, app.currentY, app.currentZ]]

def redrawAll(app):
    # title
    drawLabel('Build Game',app.width/2, 20, size=24)
    
    # Main drawing
    drawGrid(app, app.posListAll)
    drawGrid(app, app.currentPos)
    
    # Instructions section
    instructionY = 60
    spacing = 15
    
    # Movement controls
    drawLabel('Controls:', app.width/2, instructionY, size=15, bold=True)
    drawLabel('• Use hand gestures to move the cube in X/Y plane', app.width/2, instructionY + spacing,size=12)
    drawLabel('• Hold index and middle fingers together to move in Z axis', app.width/2, instructionY + spacing*2,size=12)
    
    # Building controls
    drawLabel('Building:', app.width/2, instructionY + spacing*3.5, size=15, bold=True)
    drawLabel('• SPACE: Place cube', app.width/2, instructionY + spacing*4.5,size=12)
    drawLabel('• 1-4: Change block type (1:Default, 2:L-Shape, 3:T-Shape, 4:Stair)', app.width/2, instructionY + spacing*5.5,size=12)
    drawLabel('• Q/E: Increase/Decrease cube size', app.width/2, instructionY + spacing*6.5,size=12)
    
    # View controls
    drawLabel('View Controls:', app.width/2, instructionY + spacing*8, size=15, bold=True)
    drawLabel('• Drag mouse: Rotate view', app.width/2, instructionY + spacing*9,size=12)
    drawLabel('• Left/Right arrows: Zoom in/out', app.width/2, instructionY + spacing*10,size=12)
    drawLabel('• Up/Down arrows: Change grid size', app.width/2, instructionY + spacing*11,size=12)
    
    # Special features
    drawLabel('Special Features:', app.width/2, instructionY + spacing*12.5, size=15, bold=True)
    drawLabel('• S: Toggle subdivision view', app.width/2, instructionY + spacing*13.5,size=12)
    drawLabel('• [ ]: Adjust fractal level (current: ' + str(app.fracLevel) + ')', app.width/2, instructionY + spacing*14.5,size=12)
    drawLabel('• R: Reset game', app.width/2, instructionY + spacing*15.5,size=12)
    
    # Current position and hand detection status
    drawLabel(f'Current Position: ({app.currentX}, {app.currentY}, {app.currentZ})', 
             app.width/2, app.height - spacing*2,size=12)
    
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

def main():
    runApp(width=1200, height=800)

main()