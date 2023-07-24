import matplotlib.pyplot as plt
import numpy as np

# Function to draw a vector as an arrow
def draw_arrow(ax, start, end, color):
    ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
             head_width=0.1, head_length=0.1, fc=color, ec=color, linewidth=2)

# Create a figure and axis
fig, ax = plt.subplots()

# Draw the circle with radius 1
circle = plt.Circle((0, 0), 1, color='blue', fill=False)
ax.add_patch(circle)

# Define the vectors
radius_length = 1
larger_vector_length = 1.5

# Draw the vector with the same length as the radius (direction: 45 degrees)
start_point = (0, 0)
end_point_radius = (radius_length * np.cos(np.pi/4), radius_length * np.sin(np.pi/4))
draw_arrow(ax, start_point, end_point_radius, 'green')

# Draw the vector larger than the radius (direction: 135 degrees)
end_point_larger = (larger_vector_length * np.cos(3*np.pi/4), larger_vector_length * np.sin(3*np.pi/4))
draw_arrow(ax, start_point, end_point_larger, 'red')

# Set axis limits and remove ticks
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xticks([])
ax.set_yticks([])

# Remove the plot border
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set aspect ratio to be equal, so the circle appears as a circle
ax.set_aspect('equal', adjustable='box')

# Show the plot
plt.savefig('norm.svg')