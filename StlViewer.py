from cmu_graphics import *
import struct
import math

class Mesh3D:
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.rotation = [0, 0, 0]
        self.scale = 1.0
        
    def load_stl(self, filename):
        self.vertices = []
        self.faces = []
        try:
            with open(filename, 'rb') as f:
                # Skip header
                f.seek(80)
                # Read number of triangles
                num_triangles = struct.unpack('I', f.read(4))[0]
                print(f"Loading {num_triangles} triangles...")
                
                for _ in range(num_triangles):
                    # Skip normal
                    f.seek(12, 1)
                    # Read vertices
                    for _ in range(3):
                        x, y, z = struct.unpack('fff', f.read(12))
                        self.vertices.append([x, y, z])
                    # Add face
                    face_start = len(self.vertices) - 3
                    self.faces.append([face_start, face_start + 1, face_start + 2])
                    # Skip attribute
                    f.seek(2, 1)
                
                print(f"Loaded {len(self.vertices)} vertices and {len(self.faces)} faces")
        except Exception as e:
            print(f"Error loading STL file: {e}")
            # Create a default cube if file loading fails
            self.create_default_cube()
    
    def create_default_cube(self):
        # Create a simple cube as fallback
        self.vertices = [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]
        self.faces = [
            [0, 1, 2], [0, 2, 3],  # front
            [1, 5, 6], [1, 6, 2],  # right
            [5, 4, 7], [5, 7, 6],  # back
            [4, 0, 3], [4, 3, 7],  # left
            [3, 2, 6], [3, 6, 7],  # top
            [4, 5, 1], [4, 1, 0]   # bottom
        ]
        
    def transform_point(self, point):
        # Scale
        x, y, z = [c * self.scale for c in point]
        
        # Rotate Y (simplified rotation)
        cosa = math.cos(self.rotation[1])
        sina = math.sin(self.rotation[1])
        x, z = x * cosa - z * sina, x * sina + z * cosa
        
        return [x, y, z]
    
    def project_point(self, point, width, height):
        x, y, z = point
        # Simple perspective projection
        scale = 100
        z_offset = 5
        
        if z + z_offset != 0:
            factor = scale / (z + z_offset)
            x = x * factor + width/2
            y = y * factor + height/2
            return [x, y]
        return [width/2, height/2]

def onAppStart(app):
    # Initialize the app
    app.mesh = Mesh3D()
    app.mesh.load_stl('cube.stl')  # Make sure cube.stl exists or it will create default cube
    app.mesh.scale = 0.5  # Adjust scale to see the full model
    
    # View controls
    app.rotating = True
    app.rotation_speed = 0.02
    app.show_faces = True
    app.show_edges = True

def redrawAll(app):
    # Clear background
    drawRect(0, 0, 400, 300, fill='black')
    
    # Update rotation if animation is on
    if app.rotating:
        app.mesh.rotation[1] += app.rotation_speed
    
    # Project all vertices
    projected = []
    for vertex in app.mesh.vertices:
        # Transform in 3D
        transformed = app.mesh.transform_point(vertex)
        # Project to 2D
        screen_point = app.mesh.project_point(transformed, 400, 300)
        projected.append(screen_point)
    
    # Draw faces
    if app.show_faces:
        for face in app.mesh.faces:
            # Get the three vertices of this face
            v1 = projected[face[0]]
            v2 = projected[face[1]]
            v3 = projected[face[2]]
            
            # Draw the triangle
            drawPolygon(v1[0], v1[1], v2[0], v2[1], v3[0], v3[1],
                       fill=None, border='blue')
    
    # Draw edges
    if app.show_edges:
        for face in app.mesh.faces:
            for i in range(3):
                start = projected[face[i]]
                end = projected[face[(i + 1) % 3]]
                drawLine(start[0], start[1], end[0], end[1],
                        fill='white', opacity=50)
    
    # Draw UI
    drawLabel('Space: Play/Pause   F: Toggle Faces   E: Toggle Edges',
              200, 20, fill='white')

def onKeyPress(app, key):
    if key == 'space':
        app.rotating = not app.rotating
    elif key == 'f':
        app.show_faces = not app.show_faces
    elif key == 'e':
        app.show_edges = not app.show_edges

def main():
    runApp(width=400, height=300)

main()