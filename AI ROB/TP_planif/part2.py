import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from matplotlib import collections as mc

random_seed = 17

min_step_length = 1e-3

class Env:
    # A 2D environment with bounds [-1, 1] x [-1, 1]
    def __init__(self, walls=[]):
        self.walls = [([-1.0, -1.0], [1.0, -1.0]),
                      ([1.0, -1.0], [1.0, 1.0]),
                      ([1.0, 1.0], [-1.0, 1.0]),
                      ([-1.0, 1.0], [-1.0, -1.0])] + walls

    @staticmethod
    def intersect(a, b, c, d):
        # If line segments ab and cd have a true intersection, return the intersection point. Otherwise, return False
        # a, b, c and d are 2D points of the form [x, y]
        x1, x2, x3, x4 = a[0], b[0], c[0], d[0]
        y1, y2, y3, y4 = a[1], b[1], c[1], d[1]
        denom = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3)
        if denom == 0:
            return False
        else:
            t = ((y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3)) / denom
            if t <= 0 or t >= 1:
                return False
            else:
                t = ((y1 - y2) * (x1 - x3) + (x2 - x1) * (y1 - y3)) / denom
                if t <= 0 or t >= 1:
                    return False
                else:
                    return [x3 + t * (x4 - x3), y3 + t * (y4 - y3)]

    @staticmethod
    def dist(a, b):
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    @staticmethod
    def random_oriented_action(orientation):
        rho = random.uniform(min_step_length, 0.1)
        theta = random.uniform(orientation - 25.0 * np.pi/180.0, orientation + 25.0 * np.pi/180.0)
        return [rho * np.cos(theta), rho * np.sin(theta)], theta

    def step(self, state, action, full=False):
        candidate = [state[0] + action[0], state[1] + action[1]]
        dist = np.infty
        for w in self.walls:  # Naive way to check for collisions
            pt = Env.intersect(state, candidate, w[0], w[1])
            if pt:
                candidate = pt
        if full:
            return candidate
        else:
            newstate = [state[0] + 0.99*(candidate[0] - state[0]), state[1] + 0.99*(candidate[1] - state[1])]
            if Env.dist(state, newstate) < min_step_length:  # Reject steps that are too small
                return False
            else:
                return newstate

    def plotwalls(self, ax):
        lines = []
        rgbs = []
        for w in self.walls:
            lines.append(w)
            rgbs.append((0, 0, 0, 1))
        ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=2))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis('equal')


class Tree:
    def __init__(self, init_state, orientation, parent=None, root=True):
        self.parent = parent
        self.state = init_state
        self.orientation = orientation
        self.local_inverse_density = 1
        self.successors = []
        self.root = root
        self.all_nodes = []

    def __all_edges(self):
        if not self.successors:
            return [], []
        else:
            lines = []
            rgbs = []
            for s in self.successors:
                lines.append((self.state, s.state))
                rgbs.append((1, 0, 0, 1))
                ladd, rgbadd = s.__all_edges()
                lines += ladd
                rgbs += rgbadd
            return lines, rgbs

    def plot(self, ax):
        lines, rgbs = self.__all_edges()
        ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=1))


def random_walls(env, n):
    for i in range(n):
        start = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
        progress = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
        end = env.step(start, progress, True)
        env.walls.append((start, end))


def random_expansion(t, env):
    s = random.choice(t.all_nodes + [t])
    orientation = s.orientation
    action, new_orientation = Env.random_oriented_action(orientation)
    new_state = env.step(s.state, action)
    if new_state:
        new_node = Tree(new_state, new_orientation, s, False)
        s.successors.append(new_node)
        t.all_nodes.append(new_node)


def random_rrt_expansion(t, env):
    sample = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
    nearest_neighbor = t
    d = Env.dist(t.state, sample)
    for s in t.all_nodes:  # Naive way to get the nearest neighbor
        d_tmp = Env.dist(s.state, sample)
        if d_tmp < d:
            nearest_neighbor = s
            d = d_tmp
    orientation = nearest_neighbor.orientation
    action, new_orientation = Env.random_oriented_action(orientation)
    new_state = env.step(nearest_neighbor.state, action)
    if new_state:
        new_node = Tree(new_state, new_orientation, nearest_neighbor, False)
        nearest_neighbor.successors.append(new_node)
        t.all_nodes.append(new_node)


fig, ax = plt.subplots(1, 2)

for k, func in enumerate([random_expansion, random_rrt_expansion]):
    random.seed(random_seed)
    env = Env()
    random_walls(env, 20)
    t = Tree([-0.5, -0.5], 0.0)

    for i in range(1000):
        func(t, env)

    t.plot(ax[k])
    env.plotwalls(ax[k])

plt.show()

