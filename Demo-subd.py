from cmu_graphics import *
import math

# Initialize parameters
subdivisions = 10
radius = 50
centerX, centerY = 200, 200

def draw_smooth_corner(app):
    # Clear canvas
    app.clear()
    
    # Define edge vectors (static example)
    edge1 = [(centerX, centerY), (centerX + 100, centerY - 100)]
    edge2 = [(centerX, centerY), (centerX - 100, centerY - 100)]
    
    # Draw edges
    drawLine(*edge1[0], *edge1[1], lineWidth=2, fill='blue')
    drawLine(*edge2[0], *edge2[1], lineWidth=2, fill='green')
    
    # Calculate arc points
    points = smooth_corner(edge1, edge2, radius, app.subdivisions)
    
    # Draw the arc
    for i in range(len(points) - 1):
        drawLine(*points[i], *points[i + 1], lineWidth=2, fill='red')

def smooth_corner(edge1, edge2, radius, subdivisions):
    # Extract endpoints
    (x1, y1), (x2, y2) = edge1
    (x3, y3), (x4, y4) = edge2
    
    # Edge vectors
    vec1 = [x2 - x1, y2 - y1]
    vec2 = [x4 - x3, y4 - y3]
    
    # Normalize vectors
    mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    vec1 = [vec1[0] / mag1, vec1[1] / mag1]
    vec2 = [vec2[0] / mag2, vec2[1] / mag2]
    
    # Intersection point (assume corner is at x2, y2 = x3, y3)
    P = (x2, y2)
    
    # Calculate bisector
    bisector = [vec1[0] + vec2[0], vec1[1] + vec2[1]]
    mag_bisector = math.sqrt(bisector[0]**2 + bisector[1]**2)
    bisector = [bisector[0] / mag_bisector, bisector[1] / mag_bisector]
    
    # Center of the arc
    perp = [-bisector[1], bisector[0]]
    angle_between = math.acos(vec1[0] * bisector[0] + vec1[1] * bisector[1])
    C = (P[0] + perp[0] * radius / math.cos(angle_between / 2),
         P[1] + perp[1] * radius / math.cos(angle_between / 2))
    
    # Generate arc points
    theta1 = math.atan2(vec1[1], vec1[0])
    theta2 = math.atan2(vec2[1], vec2[0])
    if theta2 < theta1:
        theta2 += 2 * math.pi  # Ensure proper ordering
    
    angles = [theta1 + i * (theta2 - theta1) / (subdivisions - 1) for i in range(subdivisions)]
    arc_points = [(C[0] + radius * math.cos(a), C[1] + radius * math.sin(a)) for a in angles]
    
    return arc_points

def onKeyPress(app, key):
    if key == 'up':
        app.subdivisions = min(app.subdivisions + 1, 50)  # Limit to 50 subdivisions
    elif key == 'down':
        app.subdivisions = max(app.subdivisions - 1, 3)  # Minimum 3 subdivisions

def onStep(app):
    draw_smooth_corner(app)

def appStarted(app):
    app.subdivisions = subdivisions

def main():
    runApp()

main()