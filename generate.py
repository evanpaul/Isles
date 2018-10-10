import random
from typing import *
import time
import math
from colorama import init, Fore, Back, Style
from enum import Enum, auto
import itertools
import sys
init()
# TODO Fix merge connected islands: how are single numbers getting in the coords list?
    # color each island one color 
# REVIEW Having both islands.coords and grid.tiles is a bit clunky
    # Grid.tiles is cleaner, but you have to search more
# TODO Make tundra and deserts: use poles and temperature probability gradients

def weighted_random(*choice_probs: List[Tuple[Any, float]]):
    '''Weighted randomly choose an item from a list given respective probabilities'''
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
    DEBUG=6


class Tile:
    def __init__(self, r, c, typ: TileType=TileType.OCEAN, isle=None):
        self.type = typ
        self.island = isle
        self.coord = r, c
        self.is_core = False
    
    def __repr__(self) -> str:
        return str(self.type.value)

    def set_type(self, typ: TileType):
        self.type = typ

    def set_island(self, isle):
        self.island = isle


class Grid:
    def __init__(self, size: int):
        self.size = size
        self.islands = []
        # REVIEW: y,x or x,y?
        self.tiles: List[Tile] = [[Tile(y, x) for x in range(self.size)] for y in range(self.size)]

    def generate(self, num_islands=5, mountains=True, echo=True) -> None:
        g.generate_islands(num_islands)
        if mountains:
            g.generate_mountains()
        if echo:
            g.print()
            input()
            g.debug_label_islands()
            g.print()

    def print(self) -> None:
        # print("   ", end="")
        # for c in range(self.size):
        #     print(hex(c)[2], end=" ")
        # print()
        for r, row in enumerate(self.tiles):
            # print(hex(r)[2], end="  ")
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
                    print(Fore.RED + "C" + Style.RESET_ALL, end=" ")
                elif tile.type is TileType.TEST:
                    print(Fore.RED + "." + Style.RESET_ALL, end=" ")
                elif tile.type is TileType.DEBUG:
                    print(Fore.BLACK + "&" + Style.RESET_ALL, end=" ")


            print()

    def get_random_loc(self) -> Tuple[int, int]:
        '''Get a random coordinate from the grid
        
        #REVIEW Perhaps make a wrapper function that only gets random ocean tiles? 
        '''
        r = random.randrange(0, self.size)
        c = random.randrange(0, self.size)

        return r, c

    def set(self, r: int, c: int, typ: TileType) -> None:
        #REVIEW Clunky. Should the grid wrap all tile accesses or none?
        #REVIEW self.get_tile(...).set(...)
        self.tiles[r][c].set_type(typ)

    def get_tile(self, r: int, c: int) -> int:
        return self.tiles[r][c]

    def generate_islands(self, num=1) -> None:
        print("[DEBUG] Generating %d islands" % num)
        for _ in range(num):
            new_isle = Island(*self.get_random_loc(), self)
            self.islands.append(new_isle)
            # merged = False
            # Check already generated islands to see if merges possible
            # REVIEW Modifying list in for block is probably the issue here 
            for isle in self.islands:
                if isle.can_merge_with(new_isle):
                    new_isle_core_debug = isle.merge(new_isle)
                    # merged = True
                    print("[DEBUG] Merged:", isle.core, new_isle.core, "->", new_isle_core_debug)
                    print(self.islands)

            # if not merged:
            #     self.islands.append(new_isle)

    def generate_mountains(self) -> None:
        if not self.islands:
            print("[WARN] Use generate_islands() before generate_mountains()")
            return

        for r, row in enumerate(self.tiles):
            for c, tile in enumerate(row):
                # Count ocean and plains to calculate odds of generating mountains
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
                    # REVIEW: 0.1 to avoid division by zero :P
                    mountain_prob = num_plains / (num_plains + num_ocean + 0.1)
                    mountain_flag = weighted_random((True, mountain_prob), (False, 1 - mountain_prob))

                    if mountain_flag:
                        self.tiles[r][c].set_type(TileType.MOUNTAINS)

    def neighbors(self, coord_pair):
        '''Generate valid grid neighbors from given coord_pair
        
        # REVIEW: Should this remain agnostic to TileType?
        '''
        r, c = coord_pair
        neighbors_coords = [(r + 1, c),
                            (r - 1, c),
                            (r, c - 1),
                            (r, c + 1)]

        for n_coord_pair in neighbors_coords:
            # Check bounds
            r_n, c_n = n_coord_pair
            if 0 <= r_n < self.size and 0 <= c_n < self.size:
                yield n_coord_pair

    def debug_label_islands(self):
        '''Label islands for clearer view of generation
        
        This SOMETIMES generates cool 'territories', but not always
        '''
        for t in itertools.chain(*self.tiles):
            if t.island and t.island not in self.islands:
                print("[ERROR] t.island is not in list")
                t.island.print()
                print("----")
                for si in self.islands:
                    si.print()
                for j in self.neighbors(t.coord):
                    self.get_tile(*j).set_type(TileType.DEBUG)
                self.print()
                # sys.exit()
            elif t.island and t.type is not TileType.OCEAN and not t.is_core:
                t.set_type(TileType(self.islands.index(t.island) + 1))
                #ValueError: <__main__.Island object at 0x10abfa128> is not in list
            # elif not t.island:
            #     print("?")
class Island:
    # TODO Play around with different models
    def __init__(self, r: int, c: int, grid: Grid, generateFlag=True):
        self.generated = False
        self.grid = grid
        self.core = (r, c)
        self.grid.get_tile(r, c).set_island(self)
        self.grid.get_tile(r, c).is_core = True
        self.coords = {self.core}
        self.queue = []

        if generateFlag:
            self.queue.append(self.core)
            self.generate()

    def __eq__(self, other):
        return self.core == other.core

    def print(self):
        print("core:", self.core)

    def generate(self) -> None:
        if self.generated:
            sys.exit("[ERROR] Island already generated")
        self.grid.set(*self.core, TileType.CORE)
        
        # Enqueue valid neighbors and turn them into TileType.PLAINS
        while self.queue:
            # REVIEW: Sigmoid
            ocean_prob = 1 / (1 + (math.e ** - len(self.coords)))
            target = self.queue.pop(0)
            self.process_neighbors(target, ocean_prob)

        self.generated = True # Not particularly useful, just for debug

    def merge(self, other):
        assert(self.grid is other.grid)
        g = self.grid
        
        if self.queue or other.queue:
            print("[ERROR] Cannot add Islands that are still generating")
            return None
        # REVIEW: error out?
        if not self.can_merge_with(other):
            print("[WARNING] Merged islands are not connected!")

        print("[DEBUG: MERGE START]{")
        print(self, end=" "); self.print()
        print(other, end=" "); other.print()
        print(g.islands)
        print("}")
        
        # Use midpoint as new core
        # TODO: Ensure it can't be an ocean tile
        r_new = int( (self.core[0] + other.core[0]) / 2 )
        c_new = int( (self.core[1] + other.core[1]) / 2 )
        # Create new island and assign it the intersection of self and other
        isle_new = Island(r_new, c_new, g, generateFlag=False)
        coords_new = self.coords.union(other.coords)
        isle_new.coords = coords_new
        # Update island reference for all new coords (not needed for every coord, but some)
        for coord_pair in isle_new.coords: # SO CLUNKY 
            g.get_tile(*coord_pair).island = isle_new

        # Set tiles
        g.set(*self.core, TileType.TEST)
        g.get_tile(*self.core).is_core = False

        g.set(*other.core, TileType.TEST)
        g.get_tile(*other.core).is_core = False

        g.set(r_new, c_new, TileType.CORE)



        # Remove old island references so only merged remains
        if self in g.islands:
            g.islands.remove(self)
        else:
            print("um?")
        if other in g.islands:
            g.islands.remove(other)
        else:
            print("UM?")
            
        g.islands.append(isle_new)


        print("[DEBUG: MERGE END]{")
        print(self, end=" "); self.print()
        print(other, end=" "); other.print()
        print(isle_new, end=" "); isle_new.print()
        print(g.islands)
        print("}")

        return isle_new.core
    
    def can_merge_with(self, other) -> bool:
        ''' Test if two Islands are able to be merged together

        This logic needs to be reworked. There may not be an intersection
        Need to check neighbors. Perhaps reconsider how grid is represented
        '''
        if self == other:
            print("[WARNING] Cannot merge an island with itself")
            print(self.core); print(other.core)
            return False
        intersects = bool(self.coords.intersection(other.coords))
        if intersects: # simple case
            print('[DEBUG] Simple case')
            return True

        mergeable = False
        for c in self.coords: # Sick Big-O yo
            for n in self.grid.neighbors(c):
                for d in other.coords:
                        if n == d:
                            print('[DEBUG] Neighbors case')
                            return True
        for p in other.coords:
            for n in other.grid.neighbors(p):
                for q in self.coords:
                        if q == p:
                            print('[DEBUG] Neighbors case 2')
                            return True
        return False

    def process_neighbors(self, coord_pair, ocean_prob) -> None:
        ''' Go through neighbors of coord_pair and generate tiles'''
        r, c = coord_pair
        # REVIEW: Too low and the whole map fills. Too high and you get diamonds
        STABILITY = 0.45 

        for n_coord_pair in self.grid.neighbors(coord_pair):
            r_neighbor, c_neighbor = n_coord_pair 
            # OCEAN -> PLAINS
            if self.grid.get_tile(r_neighbor, c_neighbor).type is TileType.OCEAN:
                # Calculate distance from core to determine probability of 
                # generating edge of island (i.e. shoreline)
                dist_from_core = math.sqrt( (r - r_neighbor) ** 2 + (c - c_neighbor) ** 2 )
                mod = STABILITY * dist_from_core
                shore_flag = weighted_random((True, mod * ocean_prob), (False, 1 - (mod * ocean_prob)))

                if shore_flag:
                    self.grid.set(r_neighbor, c_neighbor, TileType.SAND)
                else:
                    self.queue.append(n_coord_pair)
                    self.coords.add(n_coord_pair)
                    self.grid.set(r_neighbor, c_neighbor, TileType.PLAINS)
                self.coords.add(n_coord_pair)
                self.grid.get_tile(*n_coord_pair).set_island(self)

    
if __name__ == "__main__":
    while True:
        g = Grid(50)
        g.generate(num_islands=4, mountains=True, echo=True)
        input()

    

