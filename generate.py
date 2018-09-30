import random
from typing import *
import time
import math
from colorama import init, Fore, Back, Style
init()


class Grid:
    def __init__(self, size: int):
        self.size = size
        self.generate()
        self.islands = []

    def generate(self) -> None:
        self.tiles = [[0 for x in range(self.size)] for y in range(self.size)]

    def print(self) -> None:
        for row in self.tiles:
            for tile in row:
                if tile == 0:
                    print(Fore.BLUE + "0" + Style.RESET_ALL, end=" ")
                if tile == 1:
                    print(Fore.GREEN + "1" + Style.RESET_ALL, end=" ")
                if tile == 2:
                    print(Fore.YELLOW + "2" + Style.RESET_ALL, end=" ")
            print()

    def get_random_loc(self) -> Tuple[int, int]:
        r = random.randrange(0, self.size)
        c = random.randrange(0, self.size)

        return r, c

    def set(self, r, c, value) -> None:
        g.tiles[r][c] = value

    def add_island(self):
        r, c = self.get_random_loc()
        i = Island(r, c, self)
        self.islands.append(i)


class Island:
    def __init__(self, r: int, c: int, grid: Grid):
        # self.ocean_prob
        self.grid = grid
        self.core = r, c

        self.queue = [self.core]
        self.tiles = [self.core]
        self.generate()

    def generate(self) -> None:
        # print(self.queue)
        self.grid.set(self.core[0], self.core[1], 1)
        
        # Enqueue valid neighbors and turn them into 1s
        while self.queue:
            offset = 100
            ocean_prob = 1/(1+(math.e ** (offset - len(self.tiles))))
            # Review 100
            # print(len(self.tiles))
            # print(ocean_prob)
            target = self.queue.pop(0)
            self.process_neighbors(target, ocean_prob)
            # print(self.queue)
            # self.grid.print()
            # time.sleep(0.01)
            

    def process_neighbors(self, coord_pair, ocean_prob):
        r, c = coord_pair

        neighbors = [(r + 1, c),(r - 1, c),(r, c - 1),(r, c + 1)]
        for i, n_coord_pair in enumerate(neighbors):
            valid = True
            # Check bounds
            for coord in n_coord_pair:
                if not (0 <= coord < self.grid.size):
                    # Neighbor out of bounds
                    valid = False
                    break

            if valid:
                r_neighbor, c_neighbor = n_coord_pair 
                # Water -> Sand
                if self.grid.tiles[r_neighbor][c_neighbor] == 0:
                    dist_from_core = math.sqrt((r-r_neighbor)**2 + (c-c_neighbor)**2)
                    mod = 0.5 * (dist_from_core)
                    shore_flag = random.choice([True] * int(mod * ocean_prob * 100) + [False] * int((1- mod * ocean_prob) * 100))
                    if shore_flag:
                        self.grid.set(r_neighbor, c_neighbor, 2)
                    else:
                        self.queue.append(n_coord_pair)
                        self.tiles.append(n_coord_pair)
                        self.grid.set(r_neighbor, c_neighbor, 1)


if __name__ == "__main__":
    g = Grid(50)    
    g.add_island()
    g.add_island()
    g.print()


    

