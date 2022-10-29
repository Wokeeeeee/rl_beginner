import copy
import random
import sys

import numpy as np

HEIGHT, WIDTH = 4, 4

ELIMINATE_REWARD = 3
SURVIVE_REWARD = 1


def stack(flat, layers=16):
    representation = 2 ** (np.arange(layers, dtype=int) + 1)

    # layered is the flat board repeated layers times
    layered = np.repeat(flat[:, :, np.newaxis], layers, axis=-1)

    # Now set the values in the board to 1 or zero depending whether they match representation.
    # Representation is broadcast across a number of axes
    layered = np.where(layered == representation, 1, 0)

    return layered


class Game2048Env(object):
    def __init__(self):
        self.action = range(4)  # 0 up 1 down 2 right 3 left
        self.n_actions = len(self.action)
        self.board = np.zeros(shape=[HEIGHT, WIDTH], dtype=np.int32)
        self.n_obs = len(self.board)
        self.n_features = np.array([HEIGHT, WIDTH, 1])
        self.reset()

    def reset(self):
        self.score = 0
        self.repeat_counter = 0
        self.board = np.zeros(shape=[HEIGHT, WIDTH], dtype=np.int32)
        self.add_tile()
        self.add_tile()
        return stack(self.board)

    def add_tile(self):
        possible_tiles = [2, 4]
        val = random.choice(possible_tiles)
        empties = self.empties()
        if empties.shape[0] == 0: return empties.shape[0]
        empty_idx = random.choice(range(empties.shape[0]))
        empty = empties[empty_idx]
        self.set(empty[0], empty[1], val)

    def get(self, x, y):
        """Return the value of one square."""
        return self.board[x, y]

    def set(self, x, y, val):
        """Set the value of one square."""
        self.board[x, y] = val

    def empties(self):
        """Return a 2d numpy array with the location of empty squares."""
        return np.argwhere(self.board == 0)

    def zero_clear(self, line):
        for i in range(len(line) - 1, -1, -1):
            if line[i] == 0: line.pop(i)
        return line

    def shift(self, line):
        """
        向左合并一列数组
        :param line:
        :return:
        """
        scores = 0
        result = [0] * HEIGHT
        skip = False
        index = 0
        line = self.zero_clear(line)
        for i in range(len(line)):
            if skip:
                skip = False
                continue
            if i < len(line) - 1 and line[i] == line[i + 1]:
                result[index] = line[i] + line[i + 1]
                scores += ELIMINATE_REWARD #* np.log2(result[index])
                # scores += ELIMINATE_REWARD * result[index]
                skip = True
                index += 1
            else:
                result[index] = line[i]
                index += 1
        return scores, result

    def move(self, action, trial=False):
        """

        :param action:
        :param trial: true -> 只是试一下有没有方向可以走 false -> 可以set
        :return:
        """
        scores = 0
        rx = list(range(WIDTH))
        ry = list(range(HEIGHT))
        order = list(range(HEIGHT))
        changed = False
        if action == 0:  # up
            for y in range(HEIGHT):
                old = [self.get(x, y) for x in rx]
                ms, new = self.shift(old)
                scores += ms
                if not trial:
                    for x in order: self.set(x, y, new[x])
                if old != new: changed = True
        if action == 1:  # down
            for y in range(HEIGHT):
                rx_reverse = copy.deepcopy(rx)
                rx_reverse.reverse()
                old = [self.get(x, y) for x in rx_reverse]
                ms, new = self.shift(old)
                scores += ms
                if not trial:
                    for x, index in zip(rx_reverse, order): self.set(x, y, new[index])
                if old != new: changed = True
        if action == 3:  # left
            for x in range(WIDTH):
                old = [self.get(x, y) for y in ry]
                ms, new = self.shift(old)
                scores += ms
                if not trial:
                    for y in order: self.set(x, y, new[y])
                if old != new: changed = True
        if action == 2:  # right
            for x in range(WIDTH):
                ry_reverse = copy.deepcopy(ry)
                ry_reverse.reverse()
                old = [self.get(x, y) for y in ry_reverse]
                ms, new = self.shift(old)
                scores += ms
                if not trial:
                    for y, index in zip(ry_reverse, order): self.set(x, y, new[index])
                if old != new: changed = True
        return scores, changed

    def highest(self):
        """Report the highest tile on the board."""
        return np.max(self.board)

    def render(self):
        s = 'Score: {}\n'.format(self.score)
        s += 'Highest: {}\n'.format(self.highest())
        npa = np.array(self.board)
        grid = npa.reshape((WIDTH, HEIGHT))
        s += "{}\n".format(grid)
        sys.stdout.write(s)

    def step(self, action):
        info = {
            'nowhere': False,
            'action': action
        }
        score, changed = self.move(action)
        self.score += score
        self.add_tile()
        done = self.isend()

        if done:
            info['nowhere'] = True
            score = 0
        else:
            score += SURVIVE_REWARD

        if not changed:
            score = 0

        info['highest'] = self.highest()
        if self.highest() == 2048:
            self.render()
        return stack(self.board), score, done, info, changed

    def isend(self):
        """Has the game ended. Game ends if there is a tile equal to the limit
           or there are no legal moves. If there are empty spaces then there
           must be legal moves."""

        for direction in range(4):
            _, changed = self.move(direction, trial=True)
            if changed: return False
        return True

    def mean_value(self):
        return np.mean(self.board) * SURVIVE_REWARD
