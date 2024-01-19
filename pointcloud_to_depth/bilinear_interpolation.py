import numpy as np
import matplotlib.pyplot as plt

def weight(distances, max_distance):
    # Simple linear weight function, you can adjust this as needed
    weights = []
    for distance in distances:
        weights.append(max(0.0, 1 - distance / max_distance))
    return np.asarray(weights, dtype=np.float64)

def bilinear_interpolation(grid: np.ndarray):
    height, width = grid.shape
    image = grid.copy()
    vertical, horizental = 15, 3
    i = 0
    for y in range(height):
        i += 1
        for x in range(width):
            i += 1
            if grid[y, x] == 0:
                # Extract the rectangle centered at (y, x)
                window = grid[max(0, y - vertical):min(height, y + vertical+1), 
                              max(0, x - horizental):min(width, x + horizental+1)]

                # Find non-missing values and their indices in the window
                non_zero_indices = np.where(window != 0)
                non_zero_values = window[non_zero_indices]
                distances = np.sqrt((non_zero_indices[0] - vertical) ** 2 + (non_zero_indices[1] - horizental) ** 2)

                if len(non_zero_values) > 0:
                    # Calculate weights based on distances
                    weights = weight(distances, np.sqrt(vertical**2 + horizental**2)+0.1)
                    if np.sum(weights) == 0: print(f"zero {i}")
                    weights /= np.sum(weights)  # Normalize the weights
                    
                    image[y, x] = np.sum(non_zero_values*weights)

    return image
