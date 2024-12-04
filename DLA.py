import numpy as np
import random
from math import cos, sin, sqrt, pi, log
import matplotlib.pyplot as plt
import time

# Constants
NUM_PARTICLES = 5*10**4 # simulate the number of particles in all (not the actual number that grow successfully)
GRID_SIZE = 1001 # The length of the grid we work on
CENTER = (GRID_SIZE + 1) // 2 # the center coordinate of the grid

# generate the mesh grid we work on
mesh = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
# 0 for space, 1 for growed particle
mesh[CENTER, CENTER] = 1


def generate_random_array(n, seed_val):
    """
    generate an array of random number
    :param n: total number
    :param seed_val: seed for the random series
    :return: np array of random number
    """
    random.seed(seed_val)
    return np.array([random.random() for _ in range(n)])

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

def walkx(x, rand_val):
    """
    random walk in x direction
    :param x: current x coordinator
    :param rand_val: random value to decide the walk direction
    :return: new x coordinator after walking
    """
    if rand_val < 0.25:
        x += 1
    elif 0.25 <= rand_val <= 0.5:
        x -= 1
    return x

def walky(y, rand_val):
    """
    random walk in y direction
    :param y: current y coordinator
    :param rand_val: random value to decide the walk direction
    :return: new y coordinator after walking
    """
    if 0.5 < rand_val <= 0.75:
        y += 1
    elif rand_val > 0.75:
        y -= 1
    return y

def judge(mesh, CENTER, x, y, edge):
    """
    Check current status of the particle
    :param mesh: mesh grid
    :param CENTER: Center of the mesh grid
    :param x: particle's current x
    :param y: particle's current y
    :param edge: edge of the walk region
    :return: judgement
    """
    # the particle is out of the walk region, should disappear
    distance = sqrt((x - CENTER) ** 2 + (y - CENTER) ** 2)
    if distance > edge:
        return 2
    # the particle is next to the crystal, should grow
    elif (mesh[x + 1, y] == 1 or mesh[x - 1, y] == 1 or
          mesh[x, y + 1] == 1 or mesh[x, y - 1] == 1):
        return 1
    # keep walking
    else:
        return 0

def DLA_simulation():
    """
    Simulate the DLA process
    :return: mesh grid with the DLA growed on it
    """
    R_MAX = 0 # the maximum radius so far has growed
    rands = generate_random_array(10**7, seed()) # random number array
    count = 0 # index for random number array
    n_eff=0 # number of particles that grows successfully
    Rg=0 # Radius of gyration
    output=1
    with open("Rg.txt", "w") as rg_file, open("n_eff.txt", "w") as neff_file:
        # initialize the position of a particle
        for i in range(1, NUM_PARTICLES + 1):
            if 1.5*(R_MAX + 5) < (GRID_SIZE - 1) / 2 - 10:
                edge = 1.5*(5 + R_MAX)
                start = R_MAX + 5
            else:
                edge = (GRID_SIZE - 1) / 2 - 10
                start = R_MAX + 5

            x = CENTER + int(start * cos(2 * pi * rands[count]))
            y = CENTER + int(start * sin(2 * pi * rands[count]))
            count += 1
            if count >= 10**7:
                rands = generate_random_array(10**7, seed())
                count = 0
            # walking simulation
            while True:
                x = walkx(x, rands[count])
                y = walky(y, rands[count])
                count += 1
                if count >= 10**7:
                    rands = generate_random_array(10**7, seed())
                    count = 0
                # check the status of a walking particle
                status = judge(mesh, CENTER, x, y, edge)
                if status == 1:
                    # grow on the system
                    n_eff+=1
                    mesh[x, y] = 1
                    Rg+= (x - CENTER)**2 + (y - CENTER)**2
                    if n_eff == 2**output:
                        rg_file.write(f"{log(Rg)}\n")
                        neff_file.write(f"{log(n_eff)}\n")
                        output+=1
                    R_MAX = max(R_MAX, sqrt((x - CENTER) ** 2 + (y - CENTER) ** 2))
                    break
                elif status == 2:
                    # walk out of the region, disappear
                    break

            # Save snapshots
            if i in [100, 1000, 10**4, 5*10**4]:
                np.savetxt(f"mesh_{i}.txt", mesh, fmt='%d')

    #     return the mesh matrix for fractal analysis
    return mesh

def pic(filename):
    mesh_data = np.loadtxt(filename, dtype=int)
    plt.imshow(mesh_data,cmap=plt.cm.gray)
    plt.show()

def sand_box(mesh):
    """
    Use the sand box method to calculate the critical exponent
    :param mesh: 2D np array of the mesh grid
    :return:
    """
    # calculate the log of radius, and count of particles
    log_r = []
    log_count = []

    # Perform the sandbox analysis
    for i in range(1, 10):
        radius = 2 ** i
        count = 0

        # Define the square region centered at `center` with side length `radius`
        start_x = CENTER - radius // 2
        end_x = CENTER + radius // 2
        start_y = CENTER - radius // 2
        end_y = CENTER + radius // 2

        # Count the number of growed particles
        for j in range(start_x, end_x + 1):
            for k in range(start_y, end_y + 1):
                if mesh[j, k] == 1:
                    count += 1

        # process data with log
        log_r.append(np.log(float(radius)))
        log_count.append(np.log(float(count)))

    # Save the results to files
    np.savetxt("log_r.txt", log_r)
    np.savetxt("log_count.txt", log_count)

# mesh = DLA_simulation()
# sand_box(mesh)

# pic("mesh_100.txt")
# pic("mesh_1000.txt")
# pic("mesh_10000.txt")
pic("mesh_50000.txt")
print("Simulation complete.")
