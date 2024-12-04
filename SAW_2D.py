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
    lambda x, y: (y, -x),  # 90-degree rotation
    lambda x, y: (-x, -y),  # 180-degree rotation
    lambda x, y: (-y, x),  # 270-degree rotation
    lambda x, y: (-x, y),  # y-reflection
    lambda x, y: (x, -y)  # x-reflection
]


def initialize_saw(length):
    """Initialize a linear self-avoiding walk of a given length."""
    walk = [(i, 0) for i in range(length+1)]
    return walk


def apply_transformation(walk, pivot, transformation):
    """Apply a transformation to the walk around the pivot point."""
    pivot_point = walk[pivot]
    new_walk = walk[:pivot + 1]

    for x, y in walk[pivot + 1:]:
        dx, dy = x - pivot_point[0], y - pivot_point[1]
        new_x, new_y = transformation(dx, dy)
        new_walk.append((pivot_point[0] + new_x, pivot_point[1] + new_y))

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


def plot_walk(walks):
    """Plot the self-avoiding walk."""
    plt.figure(figsize=(8, 8))
    for walk in walks:
        x, y = zip(*walk)
        plt.plot(x, y, marker='o', markersize=5)
        plt.plot(x[0], y[0], marker='o', markersize=10)

    plt.title("Self-Avoiding Walk (SAW)")
    plt.axis('equal')
    plt.show()


def Combine_walks(w1, w2, rand_val):
    # global rotation w1, to make the walk isotropic
    if 0<=rand_val<1/3:
        w1_rotated = apply_transformation(w1, 0, transformations[0])
    if 1/3<=rand_val<2/3:
        w1_rotated = apply_transformation(w1, 0, transformations[1])
    if 1/3<=rand_val<=1:
        w1_rotated = apply_transformation(w1, 0, transformations[2])
    w1_reversed = list(reversed(w1_rotated))
    combined_walk = w1_reversed+w2[1:]
    return combined_walk

def Calculate_distance(walk):
    R_square = 0
    for i in range(len(walk[0])):
        R_square += (walk[-1][i]-walk[0][i])**2
    return R_square

def Exponent_calculation(num_simulations, walk_length_list):
    with open("SAW_2D.txt", "w") as file:
        file.write("walk_length\tB\tR_square\n")

    for walk_length in walk_length_list:
        # define end to end distance R^2
        R_square = 0
        rands = generate_random_array(10 ** 7, seed())  # random number array
        B = 0
        num_iterations = 25*walk_length
        for i in range(num_simulations):
            # Generate an initial walk and apply the pivot algorithm
            w1 = initialize_saw(walk_length)
            w1 = pivot_algorithm(w1, num_iterations)

            w2 = initialize_saw(walk_length)
            w2 = pivot_algorithm(w2, num_iterations)

            combined_walk = Combine_walks(w1, w2, rands[i])
            R_square+= (Calculate_distance(w1)+Calculate_distance(w2)) / (2*num_simulations)
            if is_self_avoiding(combined_walk):
                B+= 1

        with open("SAW_2D.txt", "a") as file:
            file.write(f"{walk_length}\t{B}\t{R_square}\n")

def moving_average(data, window_size):
    """Smooth data using a simple moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def Test(walk_length):
    R_square_list = []
    walk = initialize_saw(walk_length)
    # print(Calculate_distance(walk))
    R_square_list.append(Calculate_distance(walk))
    for i in range(1000):
        walk = pivot_algorithm(walk, 1)
        # print(Calculate_distance(walk))
        R_square_list.append(Calculate_distance(walk))
    num_pivots = range(len(R_square_list))  # Number of transformations (0 to len-1)
    plt.figure(figsize=(8, 6))
    plt.plot(num_pivots[:-19], moving_average(R_square_list, 20), label=r"$R^2$ over Pivot Transformations", color='b')

    plt.xlabel("Number of Pivot Transformations", fontsize=12)
    plt.ylabel(r"$R^2$", fontsize=12)
    plt.title("Evolution of $R^2$ with Pivot Transformations", fontsize=14)
    plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=10)
    plt.show()

# Parameters
walk_length_list = [10, 20, 50, 100, 200, 500]
# Exponent_calculation(1000, walk_length_list)

Test(50)

# w1 = initialize_saw(20)
# w1 = pivot_algorithm(w1, 500)
# w2 = initialize_saw(20)
# w2 = pivot_algorithm(w1, 500)
# plot_walk([w1,w2])
# w1 = apply_transformation(w1, 0, transformations[0])
# plot_walk([w1,w2])
# w1 = apply_transformation(w1, 0, transformations[0])
# plot_walk([w1,w2])
# w1 = apply_transformation(w1, 0, transformations[0])
# plot_walk([w1,w2])