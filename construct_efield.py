import numpy as np
import matplotlib.pyplot as plt

# Define the function that calculates the electric field at a given point
def electric_field(q, r, point):
    k = 10
    distance_vector = point - r
    distance = np.linalg.norm(distance_vector)
    return k*q*distance_vector/distance

# Define the parameters of the electric field
q1 = 2 # Charge of particle 1 in Coulombs
r1 = np.array([4, 2, 1]) # Position of particle 1
q2 = 1 # Charge of particle 2 in Coulombs
r2 = np.array([2, 8, 1]) # Position of particle 2
q3 = -5 # Charge of particle 2 in Coulombs
r3 = np.array([7, 3, 1]) # Position of particle 2
start = 0 # Starting point for each coordinate
end = 10 # Ending point for each coordinate
step = 1 # Step size for each coordinate

# Create a 3D grid of points
x, y, z = np.meshgrid(np.arange(start, end, step),
                      np.arange(start, end, step),
                      np.arange(start, end, step))
points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

# Calculate the electric field at each point
e1 = electric_field(q1, r1, points)
e2 = electric_field(q2, r2, points)
e3 = electric_field(q3, r3, points)
e_total = e1 + e2 + e3

# Calculate the endpoints of the electric field vectors
end_points = points + e_total

# Output the points and endpoints to text files
np.savetxt("input/efield.txt", points)
np.savetxt("input/efield_reconstruction.txt", end_points)

# Plot a quiver of the electric field
fig, ax = plt.subplots()
ax.quiver(points[:, 0], points[:, 1], e_total[:, 0], e_total[:, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.savefig("output/constructed_efield.png", dpi=300)
