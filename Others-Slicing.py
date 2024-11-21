from cmu_graphics import *
import numpy as np
from stl import mesh
from PIL import Image
import os

class StlSlicer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.model = None
        self.layers = []
        
    def read_stl(self, filename):
        """Read STL file and store the model"""
        try:
            # Check if file exists
            if not os.path.exists(filename):
                print(f"File not found: {filename}")
                print(f"Current working directory: {os.getcwd()}")
                return False
                
            # Try to load the STL file
            print(f"Attempting to load STL file: {filename}")
            self.model = mesh.Mesh.from_file(filename)
            
            # Verify model loaded correctly
            if self.model is None:
                print("Model failed to load (is None)")
                return False
                
            # Print model information for debugging
            print(f"Model loaded successfully:")
            print(f"Number of vertices: {len(self.model.vectors)}")
            print(f"Model dimensions:")
            print(f"X: {self.model.x.min():.2f} to {self.model.x.max():.2f}")
            print(f"Y: {self.model.y.min():.2f} to {self.model.y.max():.2f}")
            print(f"Z: {self.model.z.min():.2f} to {self.model.z.max():.2f}")
            
            return True
            
        except Exception as e:
            print(f"Error reading STL file: {str(e)}")
            import traceback
            print(f"Full error traceback:")
            print(traceback.format_exc())
            return False
    
    def generate_test_model(self):
        """Generate a simple test model if no STL file is available"""
        # Create a simple cube mesh
        print("Generating test cube model...")
        # Define the 8 vertices of the cube
        vertices = np.array([\
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1]])
        # Define the 12 triangles composing the cube
        faces = np.array([\
            [0,3,1], [1,3,2],  # Bottom
            [0,1,5], [0,5,4],  # Front
            [1,2,6], [1,6,5],  # Right
            [2,3,7], [2,7,6],  # Back
            [3,0,4], [3,4,7],  # Left
            [4,5,6], [4,6,7]]) # Top
        
        # Create the mesh
        cube = mesh.Mesh(np.zeros(12, dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                cube.vectors[i][j] = vertices[f[j],:]
        
        self.model = cube
        print("Test cube model generated successfully")
        return True
            
    def slice_model(self, layer_height):
        """Slice the STL model into layers"""
        if self.model is None:
            print("No model loaded - attempting to generate test model")
            if not self.generate_test_model():
                return False
            
        self.layers = []
        try:
            z_min = self.model.z.min()
            z_max = self.model.z.max()
            
            print(f"Slicing model from Z={z_min:.2f} to Z={z_max:.2f} at {layer_height}mm intervals")
            
            for z in np.arange(z_min, z_max, layer_height):
                layer_bitmap = self.generate_bitmap_for_layer(z)
                self.layers.append(layer_bitmap)
                
            print(f"Successfully generated {len(self.layers)} layers")
            return True
            
        except Exception as e:
            print(f"Error slicing model: {str(e)}")
            import traceback
            print(f"Full error traceback:")
            print(traceback.format_exc())
            return False
    
    def generate_bitmap_for_layer(self, z):
        """Generate a bitmap for a specific layer height"""
        # Create a blank white image
        image = Image.new('RGB', (self.width, self.height), 'white')
        
        try:
            # Find triangles that intersect with this z-plane
            vertices = self.model.vectors
            
            # Calculate model bounds for scaling
            x_min, x_max = self.model.x.min(), self.model.x.max()
            y_min, y_max = self.model.y.min(), self.model.y.max()
            
            # Scale factors
            scale_x = (self.width * 0.8) / (x_max - x_min)
            scale_y = (self.height * 0.8) / (y_max - y_min)
            scale = min(scale_x, scale_y)
            
            # Offsets for centering
            offset_x = self.width / 2
            offset_y = self.height / 2
            
            intersecting_triangles = []
            for triangle in vertices:
                z_vals = [v[2] for v in triangle]
                if min(z_vals) <= z <= max(z_vals):
                    # Scale and center the points
                    points = []
                    for v in triangle:
                        x = (v[0] - x_min) * scale + (self.width - (x_max - x_min) * scale) / 2
                        y = (v[1] - y_min) * scale + (self.height - (y_max - y_min) * scale) / 2
                        points.append((int(x), int(y)))
                    intersecting_triangles.append(points)
            
            # Draw all intersecting triangles
            for points in intersecting_triangles:
                self.draw_triangle(image, points)
                
        except Exception as e:
            print(f"Error generating bitmap for layer at z={z}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
        return image
    
    def draw_triangle(self, image, points):
        """Draw a triangle on the image"""
        try:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(image)
            draw.polygon(points, outline='black')
        except Exception as e:
            print(f"Error drawing triangle: {str(e)}")
        return image

def onAppStart(app):
    app.slicer = StlSlicer()
    app.currentLayer = 0
    app.layerHeight = 1.0  # 1mm layer height
    
    # Try to load the STL file
    stl_path = 'test.stl'
    print(f"\nStarting STL Slicer application")
    print(f"Attempting to load STL file: {stl_path}")
    
    if app.slicer.read_stl(stl_path):
        print("Successfully loaded STL file")
        if app.slicer.slice_model(app.layerHeight):
            print("Successfully sliced model")
            if len(app.slicer.layers) > 0:
                app.currentImage = app.slicer.layers[0]
            else:
                print("No layers generated - creating blank image")
                app.currentImage = Image.new('RGB', (800, 600), 'white')
        else:
            print("Failed to slice model - creating blank image")
            app.currentImage = Image.new('RGB', (800, 600), 'white')
    else:
        print("Failed to load STL file - creating blank image")
        app.currentImage = Image.new('RGB', (800, 600), 'white')
    
    # Convert PIL image to path for CMU Graphics
    try:
        app.currentImage.save('temp_layer.png')
        app.imagePath = 'temp_layer.png'
        print("Successfully saved initial layer image")
    except Exception as e:
        print(f"Error saving image: {str(e)}")

def onKeyPress(app, key):
    if key == 'up' and app.currentLayer < len(app.slicer.layers) - 1:
        app.currentLayer += 1
        app.slicer.layers[app.currentLayer].save('temp_layer.png')
    elif key == 'down' and app.currentLayer > 0:
        app.currentLayer -= 1
        app.slicer.layers[app.currentLayer].save('temp_layer.png')

def redrawAll(app):
    # Draw title and layer information
    drawLabel(f'STL Slicer Viewer', app.width/2, 20, size=20, bold=True)
    
    if hasattr(app.slicer, 'layers') and len(app.slicer.layers) > 0:
        drawLabel(f'Layer {app.currentLayer + 1} of {len(app.slicer.layers)}', 
                 app.width/2, 50, size=16)
    else:
        drawLabel('No layers available - Check console for errors', 
                 app.width/2, 50, size=16, fill='red')
    
    # Draw the current layer
    if os.path.exists('temp_layer.png'):
        drawImage('temp_layer.png', app.width/2, app.height/2, align='center')
    
    # Draw instructions
    drawLabel('Use UP/DOWN arrows to navigate layers', 
              app.width/2, app.height - 30, size=14)

def main():
    try:
        print("\nStarting STL Slicer application...")
        runApp(width=800, height=600)
    except Exception as e:
        print(f"Error running application: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    main()