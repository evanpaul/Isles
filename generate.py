import random
from typing import *
import time
import math
from colorama import init, Fore, Back, Style
from enum import Enum, auto
import itertools
import sys
init()
# TODO Make tundra and deserts: use poles and temperature probability gradients
# TODO Fix merge connected islands: how are single numbers getting in the coords list?


def weighted_random(*choice_probs: List[Tuple[Any, float]]):
    options = []
    for choice, prob in choice_probs:
        options += [choice] * int(prob * 100)

    return random.choice(options)


class TileType(Enum):
    OCEAN=0
    PLAINS=1
    SAND=2
    MOUNTAINS=3
    CORE=4
    TEST=5

class Tile:
    def __init__(self, typ: TileType=TileType.OCEAN):
        self.type = typ
    
    def __repr__(self) -> str:
        return str(self.type.value)

    def set_type(self, typ: TileType):
        self.type = typ


class Grid:
    def __init__(self, size: int):
        self.size = size
        self.islands = []
        self.tiles: List[Tile] = [[Tile() for x in range(self.size)] for y in range(self.size)]

    def generate(self, num_islands=5, mountains=True, echo=True) -> None:
        g.generate_islands(5)
        g.generate_mountains()
        g.print()

    def print(self) -> None:
        for row in self.tiles:
            for tile in row:
                if tile.type is TileType.OCEAN:
                    print(Fore.BLUE + "0" + Style.RESET_ALL, end=" ")
                elif tile.type is TileType.PLAINS:
                    print(Fore.GREEN + "1" + Style.RESET_ALL, end=" ")
                elif tile.type is TileType.SAND:
                    print(Fore.YELLOW + "2" + Style.RESET_ALL, end=" ") #█
                elif tile.type is TileType.MOUNTAINS:
                    print(Fore.WHITE + "3" + Style.RESET_ALL, end=" ")
                elif tile.type is TileType.CORE:
                    print(Fore.RED + "4" + Style.RESET_ALL, end=" ")
                else:
                    print(Fore.RED + "." + Style.RESET_ALL, end=" ")
            print()

    def get_random_loc(self) -> Tuple[int, int]:
        r = random.randrange(0, self.size)
        c = random.randrange(0, self.size)

        return r, c

    def set(self, r: int, c: int, typ: TileType) -> None:
        self.tiles[r][c].set_type(typ)

    def get(self, r: int, c: int) -> int:
        return self.tiles[r][c]

    def generate_islands(self, num=1) -> None:
        for i in range(num):
            # r, c = self.get_random_loc()
            new_isle = Island(*self.get_random_loc(), self)
            merged = False
            # Check already generated islands to see if merges possible
            for j, isle in enumerate(self.islands):
                if isle.can_merge_with(new_isle):
                    
                    new_isle_core_debug = isle.merge(new_isle)
                    # self.islands.append(merged_island)
                    # self.islands[j] = merged_island
                    merged = True
                    print("merged:", isle.core, new_isle.core, "->", new_isle_core_debug)
                    print([x for x in isle.coords if isinstance(x, int)])
                    print([x for x in new_isle.coords if isinstance(x, int)])
                    print(isle.coords.intersection(new_isle.coords))
            if not merged:
                self.islands.append(new_isle)

    def generate_mountains(self) -> None:
        # for tile in itertools.chain(*self.tiles):
        if not self.islands:
            print("[WARN] Use generate_islands() before generate_mountains()")
            return
        for r, row in enumerate(self.tiles):
            for c, tile in enumerate(row):
                if tile.type is TileType.SAND:
                    num_ocean = num_plains = 0
                    for n in self.neighbors((r,c)):
                        r_n, c_n = n
                        if self.tiles[r_n][c_n].type is TileType.OCEAN:
                            num_ocean += 1
                        elif self.tiles[r_n][c_n].type is TileType.PLAINS:
                            num_plains += 1
                    # If sand is surrounded by plains, we turn it into a mountain
                    # REVIEW: What about deserts?
                    # ERROR: Division by zero
                    mountain_prob = num_plains / (num_plains + num_ocean + 0.1)
                    mountain_flag = weighted_random((True, mountain_prob), (False, 1 - mountain_prob))

                    if mountain_flag:
                        self.tiles[r][c].set_type(TileType.MOUNTAINS)

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
        self.core = (r, c)

        self.queue = [self.core]
        if isinstance(self.core, int):
            print("!")
        self.coords = set(self.core)
        self.generate()
    def __eq__(self, other):
        return self.core == other.core
    def generate(self) -> None:
        self.grid.set(*self.core, TileType.CORE)
        
        # Enqueue valid neighbors and turn them into TileType.PLAINS
        while self.queue:
            # REVIEW: Sigmoid
            ocean_prob = 1 / (1 + (math.e ** -len(self.coords)))

            target = self.queue.pop(0)
            self.process_neighbors(target, ocean_prob)

    def merge(self, other):
        if self.queue or other.queue:
            sys.exit("[ERROR] Cannot add Islands that are still generating")
            return None
        # REVIEW: error out?
        if not self.can_merge_with(other):
            print("[WARNING] Islands are not connected!")

        r_new = int( (self.core[0] + other.core[0]) / 2 )
        c_new = int( (self.core[1] + other.core[1]) / 2 )

        isle_new = Island(r_new, c_new, self.grid)
        isle_new.coords = self.coords.union(other.coords) # REVIEW

        self.grid.set(*self.core, TileType.TEST)
        other.grid.set(*other.core, TileType.TEST)


        if self in self.grid.islands:
            self.grid.islands.remove(self)
        if other in other.grid.islands:
            other.grid.islands.remove(other)
    
        # print("*" * 20)
        # print(self.coords)
        # print()
        # print(isle_new.coords)
        # print()
        # print(other.coords)
        # print("*" * 20)

        # return isle_new
        self.grid.islands.append(isle_new)
        return isle_new.core
    
    def can_merge_with(self, other) -> bool:
        return bool(self.coords.intersection(other.coords))

    def process_neighbors(self, coord_pair, ocean_prob) -> None:
        r, c = coord_pair
        STABILITY = 0.45 # REVIEW: Too low and the whole map fills. Too high and you get diamonds

        for n_coord_pair in self.grid.neighbors(coord_pair):
            r_neighbor, c_neighbor = n_coord_pair 
            # OCEAN -> PLAINS
            if self.grid.get(r_neighbor, c_neighbor).type is TileType.OCEAN:
                dist_from_core = math.sqrt( (r - r_neighbor) ** 2 + (c - c_neighbor) ** 2 )
                mod = STABILITY * dist_from_core
                shore_flag = weighted_random((True, mod * ocean_prob), (False, 1 - (mod * ocean_prob)))

                if shore_flag:
                    self.grid.set(r_neighbor, c_neighbor, TileType.SAND)
                else:
                    self.queue.append(n_coord_pair)
                    if not isinstance(n_coord_pair, tuple):
                        print("[!]", n_coord_pair)
                    self.coords.add(n_coord_pair)
                    self.grid.set(r_neighbor, c_neighbor, TileType.PLAINS)

    
if __name__ == "__main__":
    g = Grid(50)
    g.generate(num_islands=5, mountains=True, echo=True)


    

