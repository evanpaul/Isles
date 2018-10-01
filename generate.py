import random
from typing import *
import time
import math
from colorama import init, Fore, Back, Style
from enum import Enum, auto
import itertools
import sys
init()
# TODO Make Tiles class
# TODO Make tundra and deserts: use poles and temperature probability gradients
# TODO Merge close islands


def weighted_random(*choice_probs: List[Tuple[Any, float]]):
    options = []
    for choice, prob in choice_probs:
        options += [choice] * int(prob * 100)

    return random.choice(options)


class Tile(Enum):
    OCEAN=0
    PLAINS=auto()
    SAND=auto()
    MOUNTAINS=auto()


class Grid:
    def __init__(self, size: int):
        self.size = size
        self.islands = []
        self.tiles = [[0 for x in range(self.size)] for y in range(self.size)]

    def generate(self, num_islands=5, mountains=True, echo=True) -> None:
        g.generate_islands(5)
        g.generate_mountains()
        g.print()

    def print(self) -> None:
        for row in self.tiles:
            for tile in row:
                if Tile(tile) == Tile.OCEAN:
                    print(Fore.BLUE + "0" + Style.RESET_ALL, end=" ")
                elif Tile(tile) == Tile.PLAINS:
                    print(Fore.GREEN + "1" + Style.RESET_ALL, end=" ")
                elif Tile(tile) == Tile.SAND:
                    print(Fore.YELLOW + "2" + Style.RESET_ALL, end=" ") #█
                elif Tile(tile) == Tile.MOUNTAINS:
                    print(Fore.WHITE + "3" + Style.RESET_ALL, end=" ")
            print()

    def get_random_loc(self) -> Tuple[int, int]:
        r = random.randrange(0, self.size)
        c = random.randrange(0, self.size)

        return r, c

    def set(self, r, c, value) -> None:
        self.tiles[r][c] = value

    def get(self, r, c) -> int:
        return self.tiles[r][c]

    def generate_islands(self, num=1) -> None:
        for i in range(num):
            r, c = self.get_random_loc()
            self.islands.append(Island(r, c, self))

    def generate_mountains(self) -> None:
        # for tile in itertools.chain(*self.tiles):
        if not self.islands:
            print("[WARN] Use generate_islands() before generate_mountains()")
            return
        for r, row in enumerate(self.tiles):
            for c, tile in enumerate(row):
                if tile == 2:
                    num_ocean = num_plains = 0
                    for n in self.neighbors((r,c)):
                        r_n, c_n = n
                        if self.tiles[r_n][c_n] == Tile.OCEAN.value:
                            num_ocean += 1
                        elif self.tiles[r_n][c_n] == Tile.PLAINS.value:
                            num_plains += 1
                    # If sand is surrounded by plains, we turn it into a mountain
                    # REVIEW: What about deserts?
                    mountain_prob = num_plains / (num_plains + num_ocean)
                    mountain_flag = weighted_random((True, mountain_prob), (False, 1 - mountain_prob))

                    if mountain_flag:
                        self.tiles[r][c] = Tile.MOUNTAINS

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
    # TODO Play around with different models
    def __init__(self, r: int, c: int, grid: Grid):
        self.grid = grid
        self.core = r, c

        self.queue = [self.core]
        self.tiles = [self.core]
        self.generate()

    def generate(self) -> None:
        self.grid.set(self.core[0], self.core[1], Tile.PLAINS.value)
        
        # Enqueue valid neighbors and turn them into Tile.PLAINS
        while self.queue:
            # REVIEW: Sigmoid
            ocean_prob = 1 / (1 + (math.e ** -len(self.tiles)))

            target = self.queue.pop(0)
            self.process_neighbors(target, ocean_prob)

    def __add__(self, other):
        if self.queue or other.queue:
            sys.exit("[ERROR] Cannot add Islands that are still generating")
            return None
        r_new = int((self.r + other.r)/2)
        c_new = int((self.c + other.c)/2)

        isle_new = Island(r_new, c_new, self.grid)
        isle_new.tiles = list(set(self.tiles) + set(other.tiles))

        return 

    def process_neighbors(self, coord_pair, ocean_prob) -> None:
        r, c = coord_pair
        STABILITY = 0.45 # REVIEW: Too low and the whole map fills. Too high and you get diamonds

        for n_coord_pair in self.grid.neighbors(coord_pair):
            r_neighbor, c_neighbor = n_coord_pair 
            # OCEAN -> PLAINS
            if self.grid.get(r_neighbor, c_neighbor) == Tile.OCEAN.value:
                dist_from_core = math.sqrt((r - r_neighbor) ** 2 + (c - c_neighbor) ** 2)
                mod = STABILITY * (dist_from_core)
                shore_flag = weighted_random((True, mod * ocean_prob), (False, 1 - (mod * ocean_prob)))

                if shore_flag:
                    self.grid.set(r_neighbor, c_neighbor, Tile.SAND.value)
                else:
                    self.queue.append(n_coord_pair)
                    self.tiles.append(n_coord_pair)
                    self.grid.set(r_neighbor, c_neighbor, Tile.PLAINS.value)

    
if __name__ == "__main__":
    g = Grid(50)
    g.generate(num_islands=5, mountains=True, echo=True)


    

