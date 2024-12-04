import numpy as np
import matplotlib.pyplot as plt
import random
from math import cos, sin, pi
import time

def seed():
    """
    generate a seed for random number from current system time
    :return: return the seed value
    """
    # Get current time components
    current_time = time.localtime()
    year = current_time.tm_year
    month = current_time.tm_mon
    day = current_time.tm_mday
    hour = current_time.tm_hour
    minute = current_time.tm_min
    second = current_time.tm_sec

    # Calculate a seed similar to the Fortran logic
    seed = year + 70 * (month + 12 * (day + 31 * (hour + 23 * (minute + 59 * second))))
    return seed

def generate_random_array(n, seed_val):
    """
    generate an array of random number
    :param n: total number
    :param seed_val: seed for the random series
    :return: np array of random number
    """
    random.seed(seed_val)
    return np.array([random.random() for _ in range(n)])

# Define transformations: rotation by 90, 180, 270 degrees and reflection
transformations = [
    # Rotations around the x-axis
    lambda x, y, z: (x, -z, y),  # 90 degrees around x
    lambda x, y, z: (x, -y, -z),  # 180 degrees around x
    lambda x, y, z: (x, z, -y),  # 270 degrees around x

    # Rotations around the y-axis
    lambda x, y, z: (-z, y, x),  # 90 degrees around y
    lambda x, y, z: (-x, y, -z),  # 180 degrees around y
    lambda x, y, z: (z, y, -x),  # 270 degrees around y

    # Rotations around the z-axis
    lambda x, y, z: (-y, x, z),  # 90 degrees around z
    lambda x, y, z: (-x, -y, z),  # 180 degrees around z
    lambda x, y, z: (y, -x, z),  # 270 degrees around z

    # Reflections across the coordinate planes
    lambda x, y, z: (-x, y, z),  # Reflection across yz plane
    lambda x, y, z: (x, -y, z),  # Reflection across xz plane
    lambda x, y, z: (x, y, -z),  # Reflection across xy plane
    lambda x, y, z: (-x, -y, -z),  # Diagonal across O
]


def initialize_saw(length, rand_val):
    """Initialize a linear self-avoiding walk of a given length."""
    # generate a walk initially pointing to a random direction
    if 0 <= rand_val < 1/6: # x+
        walk = [(i, 0, 0) for i in range(length)]
    if 1/6 <= rand_val < 2/6: # y+
        walk = [(0, i, 0) for i in range(length)]
    if 2/6 <= rand_val < 3/6: # z+
        walk = [(0, 0, i) for i in range(length)]
    if 3/6 <= rand_val < 4/6: # x-
        walk = [(-i, 0, 0) for i in range(length)]
    if 4/6 <= rand_val < 5/6: # y-
        walk = [(0, -i, 0) for i in range(length)]
    if 5/6 <= rand_val <= 1: # z-
        walk = [(0, 0, -i) for i in range(length)]

    return walk


def apply_transformation(walk, pivot, transformation):
    """Apply a transformation to the walk around the pivot point."""
    pivot_point = walk[pivot]
    new_walk = walk[:pivot + 1]

    for x, y, z in walk[pivot + 1:]:
        dx, dy, dz = x - pivot_point[0], y - pivot_point[1], z-pivot_point[2]
        new_x, new_y, new_z = transformation(dx, dy, dz)
        new_walk.append((pivot_point[0] + new_x, pivot_point[1] + new_y, pivot_point[2] + new_z))

    return new_walk


def is_self_avoiding(walk):
    """Check if a walk is self-avoiding (no overlaps)."""
    return len(walk) == len(set(walk)) # remove duplicate points


def pivot_algorithm(walk, num_iterations):
    """Perform the pivot algorithm on an initial self-avoiding walk."""
    num_success=0 # count the number of successful walks
    while True:
        # Choose a random pivot point (not including the first point)
        pivot = random.randint(1, len(walk) - 1)

        # Choose a random transformation
        transformation = random.choice(transformations)

        # Apply transformation and check if the result is self-avoiding
        new_walk = apply_transformation(walk, pivot, transformation)
        if is_self_avoiding(new_walk):
            walk = new_walk  # Accept the new configuration
            num_success+=1

        if num_success == num_iterations: # stop when reaching the goal number
            break

    return walk


def plot_3d_walk(walk):
    """
    Plots a 3D walk.

    Parameters:
    - walk: A list of tuples representing the 3D coordinates of the walk (e.g., [(x1, y1, z1), (x2, y2, z2), ...]).
    """
    # Extract x, y, z coordinates
    x, y, z = zip(*walk)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the walk
    ax.plot(x, y, z, marker='o', label='3D Walk')
    ax.scatter(x[0], y[0], z[0], color='green', s=100, label='Start')  # Starting point
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='End')  # Ending point

    # Label the axes
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Set title and legend
    ax.set_title('3D Self-Avoiding Walk')
    ax.legend()

    # Show the plot
    plt.show()


def Combine_walks(w1, w2):
    # Now since we have initialized the walks with a random direction
    # we don't need to rotate a walk here before combining two
    w1_reversed = list(reversed(w1))
    combined_walk = w1_reversed+w2[1:]
    return combined_walk

def Calculate_distance(walk):
    R_square = 0
    for i in range(len(walk[0])):
        R_square += (walk[-1][i]-walk[0][i])**2
    return R_square

def Exponent_calculation(num_simulations, walk_length_list, start=True):
    if start:
        with open("SAW_3D.txt", "w") as file:
            file.write("walk_length\tB\tR_square\n")

    for walk_length in walk_length_list:
        rands = generate_random_array(10 ** 7, seed())  # random number array
        # define end to end distance R^2
        R_square = 0
        B = 0
        num_iterations = 25*walk_length
        for i in range(num_simulations):
            # Generate an initial walk and apply the pivot algorithm
            # only need to generate one of them with random initial direction
            # (we only need them to be relatively random directed
            w1 = initialize_saw(walk_length, rands[i])
            w1 = pivot_algorithm(w1, num_iterations)

            w2 = initialize_saw(walk_length, 0)
            w2 = pivot_algorithm(w2, num_iterations)

            combined_walk = Combine_walks(w1, w2)

            R_square+= (Calculate_distance(w1)+Calculate_distance(w2)) / (2*num_simulations)

            if is_self_avoiding(combined_walk):
                B+= 1

        with open("SAW_3D.txt", "a") as file:
            file.write(f"{walk_length}\t{B}\t{R_square}\n")



# Parameters
walk_length_list = [10, 20, 50, 100, 200, 500]
# Exponent_calculation(1000, walk_length_list)

w=initialize_saw(50,generate_random_array(1,seed())[0])
plot_3d_walk(w)

w=pivot_algorithm(w,1000)
plot_3d_walk(w)