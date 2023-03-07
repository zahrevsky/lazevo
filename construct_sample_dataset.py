import math

# Define the center and radius of the ball
cylinder_center = (2, 2)
radius = 2

# Define the step size for each coordinate
step = 0.5

# Calculate the number of steps required to cover the range of 0 to 10
num_steps = int(10.0 / step) + 1

# Create an empty list to store the valid points
points = []

# Loop through each coordinate in the grid
for x in range(num_steps):
    for y in range(num_steps):
        for z in range(num_steps):
            # Convert the integer coordinates to floats and multiply by the step size
            xf = x * step
            yf = y * step
            zf = z * step
            
            if cylinder_center[0] - radius < xf < cylinder_center[0] + radius:
                points.append((xf, yf, zf))

# Print the list of valid points
with open('input/cube_without_cylinder.txt', 'w') as out:
    for point in points:
        out.write(f"{point[0]} {point[1]} {point[2]}\n")
