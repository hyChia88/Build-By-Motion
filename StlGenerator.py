import numpy as np
import math

class STLGenerator:
    def __init__(self):
        self.vertices = []
        self.faces = []
        
    def reset(self):
        self.vertices = []
        self.faces = []
    
    def add_triangle(self, v1, v2, v3):
        idx = len(self.vertices)
        self.vertices.extend([v1, v2, v3])
        self.faces.append([idx, idx+1, idx+2])
    
    def generate_cube(self, size=1.0, center=True):
        self.reset()
        if center:
            offset = -size/2
        else:
            offset = 0
            
        vertices = np.array([
            [offset, offset, offset],
            [offset+size, offset, offset],
            [offset+size, offset+size, offset],
            [offset, offset+size, offset],
            [offset, offset, offset+size],
            [offset+size, offset, offset+size],
            [offset+size, offset+size, offset+size],
            [offset, offset+size, offset+size]
        ])
        
        # Define faces
        faces = [
            [0,1,2], [0,2,3],  # bottom
            [4,6,5], [4,7,6],  # top
            [0,4,1], [1,4,5],  # front
            [1,5,2], [2,5,6],  # right
            [2,6,3], [3,6,7],  # back
            [3,7,0], [0,7,4]   # left
        ]
        
        for face in faces:
            self.add_triangle(vertices[face[0]], vertices[face[1]], vertices[face[2]])
    
    def generate_pyramid(self, base_size=1.0, height=1.5):
        self.reset()
        # Base vertices
        base = [
            [-base_size/2, 0, -base_size/2],
            [base_size/2, 0, -base_size/2],
            [base_size/2, 0, base_size/2],
            [-base_size/2, 0, base_size/2]
        ]
        apex = [0, height, 0]
        
        # Base triangles
        self.add_triangle(base[0], base[2], base[1])
        self.add_triangle(base[0], base[3], base[2])
        
        # Side triangles
        for i in range(4):
            self.add_triangle(base[i], base[(i+1)%4], apex)
    
    def generate_sphere(self, radius=1.0, resolution=20):
        self.reset()
        for i in range(resolution):
            lat0 = math.pi * (-0.5 + float(i) / resolution)
            z0 = math.sin(lat0)
            zr0 = math.cos(lat0)
            
            lat1 = math.pi * (-0.5 + float(i+1) / resolution)
            z1 = math.sin(lat1)
            zr1 = math.cos(lat1)
            
            for j in range(resolution):
                lng = 2 * math.pi * float(j) / resolution
                lng2 = 2 * math.pi * float(j+1) / resolution
                
                x0 = math.cos(lng)
                y0 = math.sin(lng)
                x1 = math.cos(lng2)
                y1 = math.sin(lng2)
                
                v0 = [x0*zr0*radius, y0*zr0*radius, z0*radius]
                v1 = [x1*zr0*radius, y1*zr0*radius, z0*radius]
                v2 = [x1*zr1*radius, y1*zr1*radius, z1*radius]
                v3 = [x0*zr1*radius, y0*zr1*radius, z1*radius]
                
                if i != 0:
                    self.add_triangle(v0, v1, v2)
                if i != (resolution-1):
                    self.add_triangle(v0, v2, v3)

    def generate_cylinder(self, radius=1.0, height=2.0, segments=20):
        self.reset()
        # Generate vertices for top and bottom circles
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            # Bottom circle
            v1 = [x, y, 0]
            v2 = [x, y, height]
            
            # Next points
            angle2 = 2 * math.pi * ((i+1) % segments) / segments
            x2 = radius * math.cos(angle2)
            y2 = radius * math.sin(angle2)
            v3 = [x2, y2, 0]
            v4 = [x2, y2, height]
            
            # Add side faces
            self.add_triangle(v1, v2, v3)
            self.add_triangle(v2, v4, v3)
            
            # Add top and bottom faces
            self.add_triangle([0, 0, 0], v1, v3)
            self.add_triangle([0, 0, height], v4, v2)
    
    def save_stl(self, filename):
        with open(filename, 'wb') as f:
            # Write header
            f.write(b'\x00' * 80)
            # Write number of triangles
            f.write(np.array(len(self.faces), dtype=np.uint32).tobytes())
            
            # Write each triangle
            for face in self.faces:
                # Calculate normal
                v1 = np.array(self.vertices[face[1]]) - np.array(self.vertices[face[0]])
                v2 = np.array(self.vertices[face[2]]) - np.array(self.vertices[face[0]])
                normal = np.cross(v1, v2)
                if np.any(normal):  # Avoid division by zero
                    normal = normal / np.linalg.norm(normal)
                
                # Write normal
                f.write(normal.astype(np.float32).tobytes())
                # Write vertices
                for vertex_idx in face:
                    f.write(np.array(self.vertices[vertex_idx], dtype=np.float32).tobytes())
                # Write attribute byte count
                f.write(np.array(0, dtype=np.uint16).tobytes())

# Example usage and generation of multiple shapes
def generate_sample_files():
    generator = STLGenerator()
    
    # Generate cube
    generator.generate_cube(size=2.0)
    generator.save_stl('cube.stl')
    
    # Generate pyramid
    generator.generate_pyramid(base_size=2.0, height=3.0)
    generator.save_stl('pyramid.stl')
    
    # Generate sphere
    generator.generate_sphere(radius=1.5, resolution=30)
    generator.save_stl('sphere.stl')
    
    # Generate cylinder
    generator.generate_cylinder(radius=1.0, height=2.0, segments=30)
    generator.save_stl('cylinder.stl')

if __name__ == "__main__":
    generate_sample_files()