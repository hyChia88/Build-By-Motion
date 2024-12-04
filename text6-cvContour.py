import cv2
import numpy as np

def process_image(filename):
    # Load image directly in grayscale
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # Check if image was successfully loaded
    if img is None:
        print("Error: Could not load image from", filename)
        return None
    
    # Print image dimensions and data type as additional verification
    print("Image dimensions:", img.shape)
    print("Image data type:", img.dtype)
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

# def reMap(div, boolArray):
#     newArray = []
#     print("the remap process:")
#     print(len(boolArray))
#     # print(boolArray)
#     for i in range(0, len(boolArray), div):
#         newArray.append([])
#         for j in range(0, len(boolArray[i]), div):
#             newArray[-1].append(boolArray[i][j])
#     print("the new array:")
#     print(len(newArray))
#     print(newArray)
#     return newArray

def reMap(newH, newW, boolArray):
    oriH = len(boolArray)
    oriW = len(boolArray[0])
    
    # scale
    scaleH = oriH // newH
    scaleW = oriW // newW
    
    newArray = []
    print("the remap process:")
    for i in range(newH):
        newArray.append([])
        for j in range(newW):
            mapX = int(j * scaleW)
            mapY = int(i * scaleH)
            newArray[-1].append(boolArray[mapY][mapX])

    print("old array len", len(boolArray))
    print("the new array:")
    print(len(newArray))
    print(len(newArray[0]))
    print(newArray)
    return newArray


# Test
gridSize = 5
image_path = "testEdge.jpg"
result = process_image(image_path)
print(result['binary_bool'])
reMap(gridSize,gridSize, result['binary_bool'])

# Visualization
cv2.imshow('Grayscale', result['grayscale'])
cv2.imshow('Binary (uint8)', result['binary_uint8'])
# cv2.imshow('Binary (bool) - remap', result['binary_bool'])
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
It needs to be in this:
[[[None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None]],
[[None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None]], [[None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None]], [[None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None]], [[None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None]]]
'''