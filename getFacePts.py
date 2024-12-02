'''
this set of code is to get points and faces to draw from cell.x, cell.y, cell.z (which is "shift")
return 2 set: all points, faces that to build faces
'''
import copy

def moveCell(movement):
    # Template cube points and faces
    CUBE_POINTS = [
        [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1],
        [1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0],
    ]
    # back top left, front top left, front top right, back top right,
    # front bottom right, back bottom right, front bottom left, back bottom left
    
    CUBE_FACES = [
        [0, 1, 2, 3],
        [3, 2, 4, 5],
        [5, 4, 6, 7],
        [7, 0, 3, 5],
        [7, 6, 1, 0],
        [6, 1, 2, 4],
    ]
    
    # Generate updated points
    allPts = []
    for shift in movement:
        for point in CUBE_POINTS:
            # Add the shift to each point
            new_point = [
                point[0] + shift[0],
                point[1] + shift[1],
                point[2] + shift[2]
            ]
            allPts.append(new_point)
    
    # Generate updated faces
    allFaces = []
    for i, shift in enumerate(movement):
        # For each cube, add its faces with updated indices
        offset = i * len(CUBE_POINTS)  # offset is 8 for each subsequent cube
        for face in CUBE_FACES:
            # Add the offset to each point index in the face
            new_face = [idx + offset for idx in face]
            allFaces.append(new_face)
    
    return allPts, allFaces

'''
get center of each face
'''
def getFacePts(allPts, allFaces):
    face_centers = []
    
    for face in allFaces:
        # Initialize center coordinates
        center = [0.0, 0.0, 0.0]
        
        # Sum up all points of the face
        for point_idx in face:
            point = allPts[point_idx]
            center[0] += point[0]
            center[1] += point[1]
            center[2] += point[2]
        
        # Divide by number of points to get average (center)
        num_points = len(face)
        center = [coord / num_points for coord in center]
        
        face_centers.append(tuple(center))
    
    return face_centers

def clean_mesh(points, faces):
    """Clean mesh by removing duplicate points and faces."""
    # Create point mapping dictionary
    point_map = dict()
    cleaned_points = []
    old_to_new_idx = dict()
    
    # Process points
    for i, point in enumerate(points):
        point_tuple = tuple(point)
        if point_tuple not in point_map:
            point_map[point_tuple] = len(cleaned_points)
            cleaned_points.append(point)
        old_to_new_idx[i] = point_map[point_tuple]
    
    # Clean faces and update indices
    face_centers = dict()
    cleaned_faces = []
    
    for face in faces:
        # Update indices
        new_face = [old_to_new_idx[idx] for idx in face]
        # Check for duplicate faces
        center = getFacePts([points[idx] for idx in face], [[0,1,2,3]])[0]
        if center not in face_centers:
            face_centers[center] = face_centers.get(center, 0) + 1
            if face_centers[center] == 1:
                cleaned_faces.append(new_face)
    print(face_centers)
    
    return cleaned_points, cleaned_faces


'''
Defination of "clean":
1. remove duplicate points (mark down the index of the dup point in allPts)
2. remove duplicate faces by matching their center
3. replace the point in each face by the first point in ptsDict[tuple(pt)]
'''
# move cell to test
shifts = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2]]

allPts, allFaces = moveCell(shifts)

# # With this:
cleaned_points, cleaned_faces = clean_mesh(allPts, allFaces)
print(f"cleaned points ({len(cleaned_points)}):", cleaned_points)
print(f"cleaned faces ({len(cleaned_faces)}):", cleaned_faces)