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
                    print(Fore.BLUE + "~" + Style.RESET_ALL, end=" ")
                elif tile == 1:
                    print(Fore.GREEN + "-" + Style.RESET_ALL, end=" ")
                elif tile == 2:
                    print(Fore.YELLOW + "." + Style.RESET_ALL, end=" ")
                elif tile == 3:
                    print(Fore.WHITE + "^" + Style.RESET_ALL, end=" ")
            print()

    def get_random_loc(self) -> Tuple[int, int]:
        r = random.randrange(0, self.size)
        c = random.randrange(0, self.size)

        return r, c

    def set(self, r, c, value) -> None:
        g.tiles[r][c] = value

    def generate_island(self):
        r, c = self.get_random_loc()
        i = Island(r, c, self)
        self.islands.append(i)

    def generate_mountains(self):
        import itertools
        # for tile in itertools.chain(*self.tiles):
        for r, row in enumerate(self.tiles):
            for c, tile in enumerate(row):
                if tile == 2:
                    num_water = num_grass = 0
                    for n in self.neighbors((r,c)):
                        r_n, c_n = n
                        if self.tiles[r_n][c_n] == 0:
                            num_water += 1
                        elif self.tiles[r_n][c_n] == 1:
                            num_grass += 1
                    mountain_prob = num_grass / (num_grass + num_water)
                    mountain_flag = random.choice([True] * int(mountain_prob * 100) + [False] * int((1 -mountain_prob) * 100))

                    if mountain_flag:
                        self.tiles[r][c] = 3

    def neighbors(self, coord_pair):
        r, c = coord_pair
        neighbs = [(r + 1, c),
                    (r - 1, c),
                    (r, c - 1),
                    (r, c + 1)]

        for n_coord_pair in neighbs:
            # Check bounds
            r_n, c_n = n_coord_pair
            if 0 <= r_n < self.size and 0 <= c_n < self.size:
                yield n_coord_pair


class Island:
    # Play around with different models
    def __init__(self, r: int, c: int, grid: Grid):
        self.grid = grid
        self.core = r, c

        self.queue = [self.core]
        self.tiles = [self.core]
        self.generate()

    def generate(self) -> None:
        self.grid.set(self.core[0], self.core[1], 1)
        
        # Enqueue valid neighbors and turn them into 1s
        while self.queue:
            offset = 100 # REVIEW
            ocean_prob = 1/(1+(math.e ** (offset - len(self.tiles))))

            target = self.queue.pop(0)
            self.process_neighbors(target, ocean_prob)



    def process_neighbors(self, coord_pair, ocean_prob) -> None:
        r, c = coord_pair

        for n_coord_pair in self.grid.neighbors(coord_pair):
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
    g = Grid(400)    
    i = 0
    while i < 40:
        g.generate_island()
        i+=1
    g.generate_mountains()
    g.print()


    

