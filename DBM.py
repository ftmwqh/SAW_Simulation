import math

import numpy as np
import random
from math import sqrt, pi, atan
import matplotlib.pyplot as plt

def seed():
    """Generate a random seed based on the current time."""
    from datetime import datetime
    now = datetime.now()
    return int(now.timestamp() * 1000000)


def random_array(n, seed_val):
    """Generate an array of n random numbers in [0, 1]."""
    random.seed(seed_val)
    return np.array([random.random() for _ in range(n)])


def rand_num(seed_val):
    """Generate a random number based on the previous seed."""
    random.seed(seed_val)
    return random.random()


def sum_array(arr):
    """Compute the sum of an array."""
    return np.sum(arr)


def walkx(x, rand_val):
    """Random walk in x-direction."""
    if rand_val < 0.25:
        x += 1
    elif rand_val > 0.25 and rand_val < 0.5:
        x -= 1
    return x


def walky(y, rand_val):
    """Random walk in y-direction."""
    if rand_val > 0.5 and rand_val < 0.75:
        y += 1
    elif rand_val > 0.75:
        y -= 1
    return y


def judge(mesh, L, x, y, edge):
    """
    check the current position
    :param mesh: the grid
    :param L: size of grid
    :param x: coordinate x
    :param y: coordiniate y
    :param edge: edge scale
    :return: 2: disappear, 1: grow, 0: keep walking
    """""
    center = (L + 1) // 2
    # check if the particle is out of edge
    if sqrt((x - center) ** 2 + (y - center) ** 2) > edge:
        return 2
    # check if the particle reaches the cluster
    elif (
        mesh[x + 1, y] == 1
        or mesh[x - 1, y] == 1
        or mesh[x, y + 1] == 1
        or mesh[x, y - 1] == 1
    ):
        return 1
    # keep walking
    else:
        return 0


def grow(mesh, L, t, n, eta, rand_grow, rmax):
    """Determine the growth position."""
    m = 2**31 - 1
    center = (L + 1) // 2
    edge = min((rmax+10) , (L - 1) // 2 - 1)
    rand_val = seed()
    i=0
    rands = random_array(10**7, seed())
    start = []
    # find the edge of the cluster, where 0 and 1 connects
    for x in range(center - int(edge), center + int(edge) + 1):
        for y in range(center - int(edge), center + int(edge) + 1):
            if mesh[x, y] == 0 and judge(mesh, L, x, y, edge) == 1:
                start.append((x, y))

    # calculate potential phi
    phi = np.zeros(n)
    p = np.zeros(len(start))
    for j, (x0, y0) in enumerate(start):
        for k in range(n):
            x, y = x0, y0
            while True:
                # rand_val = rand_num(rand_val)
                x = walkx(x, rands[i])
                y = walky(y, rands[i])
                i += 1
                if i == m:
                    rands=random_array(10**7, seed())
                    i=0

                if judge(mesh, L, x, y, edge) == 2:
                    phi[k] = 0
                    break
                elif mesh[x, y] == 1:
                    phi[k] = 1.0
                    break

        p[j] = (t + 1) * (1.0 - np.mean(phi)) ** eta
    # calculate the probability of a grid being occupied
    p /= np.sum(p)
    cum_prob = np.cumsum(p)
    for j, prob in enumerate(cum_prob):
        if rand_grow < prob:
            return start[j]

def pic(filename):
    mesh_data = np.loadtxt(filename, dtype=int)
    plt.imshow(mesh_data,cmap=plt.cm.gray)
    plt.show()

def main():
    t = 10**3
    n = 10**2
    L = 501
    eta = 2
    # Size of grid
    center = (L + 1) // 2
    rmax = 0

    mesh = np.zeros((L, L), dtype=int)
    mesh[center, center] = 1

    rand_vals = random_array(t, seed())

    for k in range(1, t + 1):
        next_pos = grow(mesh, L, k, n, eta, rand_vals[k - 1], rmax)
        mesh[next_pos[0], next_pos[1]] = 1
        rmax = max(rmax, math.sqrt((next_pos[0]-center)**2+(next_pos[1]-center)**2))

        if k in [100, 300, 700, 1000, 1500, 2000]:
            np.savetxt(f"DBM_{k}.txt", mesh, fmt="%d")

    print("finish")

if __name__ == "__main__":
    # simulate DBM
    # main()

    # plot DBM
    pic("DBM_100.txt")
    pic("DBM_300.txt")
    pic("DBM_700.txt")
    pic("DBM_1000.txt")
