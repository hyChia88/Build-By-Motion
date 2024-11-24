import numpy as np
import math

class Projection3D:
    def __init__(self):
        self.near = 0.1    # Near clipping plane
        self.far = 1000.0  # Far clipping plane
        self.fov = 60      # Field of view in degrees
        self.aspect = 16/9 # Aspect ratio (width/height)
        
    def basic_projection(self, x, y, z, rotation_y, rotation_x):
        """
        Basic trigonometric projection (like in your original code)
        Applies rotation matrices manually
        """
        # Y-axis rotation
        rotX = x * math.cos(rotation_y) - y * math.sin(rotation_y)
        rotY = x * math.sin(rotation_y) + y * math.cos(rotation_y)
        
        # X-axis rotation
        finalY = rotY * math.cos(rotation_x) - z * math.sin(rotation_x)
        finalZ = rotY * math.sin(rotation_x) + z * math.cos(rotation_x)
        
        # Basic perspective division (makes farther objects appear smaller)
        scale = 1.0 / (finalZ + 5)  # +5 to avoid division by zero
        screenX = rotX * scale
        screenY = finalY * scale
        
        return (screenX, screenY, finalZ)
    
    def matrix_projection(self, point):
        """
        Professional matrix-based projection using homogeneous coordinates
        This is how most 3D engines implement projection
        """
        # Convert point to homogeneous coordinates
        point = np.array([point[0], point[1], point[2], 1.0])
        
        # Create perspective projection matrix
        f = 1.0 / math.tan(math.radians(self.fov) / 2.0)
        aspect = self.aspect
        near = self.near
        far = self.far
        
        projection = np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
            [0, 0, -1, 0]
        ])
        
        # Apply projection
        result = np.dot(projection, point)
        
        # Perform perspective division
        if result[3] != 0:
            result = result / result[3]
            
        return (result[0], result[1], result[2])
    
    def orthographic_projection(self, point, width=2, height=2):
        """
        Orthographic projection - no perspective, used in CAD and architectural drawings
        """
        return (point[0]/width, point[1]/height, point[2])
    
    def cabinet_projection(self, point, alpha=45):
        """
        Cabinet projection - a type of oblique projection
        Popular in technical drawings
        """
        # Convert angle to radians
        alpha = math.radians(alpha)
        
        # Cabinet projection matrix
        cabinet = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0.5 * math.cos(alpha), 0.5 * math.sin(alpha), 1]
        ])
        
        point = np.array([point[0], point[1], point[2]])
        result = np.dot(cabinet, point)
        
        return (result[0], result[1], result[2])

    def isometric_projection(self, point):
        """
        Isometric projection - equal angle projection
        Common in technical illustrations and games
        """
        # Isometric angles
        angle = math.radians(30)
        
        # Isometric transformation matrix
        isometric = np.array([
            [math.cos(angle), -math.cos(angle), 0],
            [math.sin(angle), math.sin(angle), 1],
            [0, 0, 1]
        ])
        
        point = np.array([point[0], point[1], point[2]])
        result = np.dot(isometric, point)
        
        return (result[0], result[1], result[2])

# Example usage
if __name__ == "__main__":
    projector = Projection3D()
    
    # Test point
    point3d = (1.0, 1.0, 2.0)
    
    # Try different projections
    basic = projector.basic_projection(*point3d, math.pi/4, math.pi/6)
    perspective = projector.matrix_projection(point3d)
    ortho = projector.orthographic_projection(point3d)
    cabinet = projector.cabinet_projection(point3d)
    iso = projector.isometric_projection(point3d)

    print(basic)
    print(perspective)
    print(ortho)
    print(cabinet)
    print(iso)
        
