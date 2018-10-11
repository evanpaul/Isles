import itertools
import math
import random
import sys
import time
from enum import Enum, auto
from typing import *
from collections import defaultdict
from pprint import pprint

from colorama import Back, Fore, Style, init
import uuid

init()
# TODO Fix merge connected islands: how are single numbers getting in the coords list?
    # color each island one color 
# REVIEW Having both islands.coords and grid.tiles is a bit clunky
    # Grid.tiles is cleaner, but you have to search more
    # Shift tile logic to... Tile i.e. detach coords from Islands/Grid.
# TODO Make tundra and deserts: use poles and temperature probability gradients
# TODO Add logging


def weighted_random(choices: List[Any], probs: List[float]):
    '''Weighted randomly choose an item from a list given respective probabilities'''
    s = int(sum(probs))
    if s != 1:
        print("[WARNING] weighted_random(): probabilities should sum to 1.0, got: ", s)

    if len(choices) != len(probs):
        sys.exit("[ERROR] weighted_random(): must provide as many probalities as choices")

    options: List[Any] = []
    for choice, prob in zip(choices, probs):
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
    DEBUG2=7


class Tile:
    def __init__(self, coord: Tuple[int, int], typ: TileType=TileType.OCEAN, isle=None) -> None:
        self.type = typ
        self.island = isle
        self.coord = coord
        self.is_core = False
    
    def __repr__(self) -> str:
        return str(self.type.value)
    
    def debug(self) -> None:
        print("[%s](%d, %d): %d" % (self.island, *self.coord, self.type.value))

    def set_type(self, typ: TileType) -> None:
        self.type = typ

    def set_island(self, isle) -> None:
        self.island = isle


class Grid:
    def __init__(self, size: int) -> None:
        self.size = size
        self.islands: Dict[Island, Set[Tuple[int, int]]] = {} 
        # REVIEW: y,x or x,y?
        self.tiles: List[List[Tile]] = [[Tile((r, c)) for c in range(self.size)] for r in range(self.size)]

    def generate(self, num_islands=5, mountains=True, echo=True) -> None:
        g.generate_islands(num_islands)
        if mountains:
            g.generate_mountains()
        if echo:
            g.print()
            input()
            print()
            g.debug_label_islands()
            g.print()

    def tally(self):
        counter = defaultdict(int)
        for t in itertools.chain(*self.tiles):
            counter[t.island] += 1
        pprint(dict(counter))

    def print(self) -> None:
        display_scheme = {
            TileType.OCEAN.value: (Fore.BLUE, "0"),
            TileType.PLAINS.value: (Fore.GREEN, "1"),
            TileType.SAND.value: (Fore.YELLOW, "2"),
            TileType.MOUNTAINS.value: (Fore.WHITE, "3"),
            TileType.CORE.value: (Fore.RED, "C"),
            TileType.TEST.value: (Fore.RED, "X"),
            TileType.DEBUG.value: (Fore.BLACK, "&"),
            TileType.DEBUG2.value: (Fore.CYAN, "?"),
            8: (Fore.CYAN, "?"),
            9: (Fore.CYAN, "?")
        }
        # self.tally()

        for row in self.tiles:
            for tile in row:
                # if tile.island:
                #     tile.debug()
                scheme = display_scheme[tile.type.value]
                print(scheme[0] + scheme[1] + Style.RESET_ALL, end=" ")
            print()

    def get_random_loc(self) -> Tuple[int, int]:
        '''Get a random coordinate from the grid
        
        # REVIEW Perhaps make a wrapper function that only gets random ocean tiles? 
        '''
        r = random.randrange(0, self.size)
        c = random.randrange(0, self.size)

        return r, c

    def set(self, coords: Tuple[int, int], typ: TileType) -> None:
        #REVIEW self.get_tile(...).set(...), clunky
        r, c = coords
        self.tiles[r][c].set_type(typ)

    def get_tile(self, coords: Tuple[int, int]) -> Tile:
        r, c = coords
        return self.tiles[r][c]
    

    def generate_islands(self, num=1) -> None:
        print("[DEBUG] Generating %d islands" % num)
        for _ in range(num):
            new_isle = Island(self.get_random_loc(), self)
            print("!" * 10)
            pprint(self.islands)
            self.islands[new_isle] = set()
            print("-" * 10)
            pprint(self.islands)
            print("!" * 10)
            # merged = False
            # Check already generated islands to see if merges possible
            # REVIEW Modifying list in for block is probably the issue here
            i = 0
            print("*" * 10)
            pprint(self.islands)
            self.tally()
            
            for isle in self.islands:
                # print(isle,":", len(self.islands[isle]))

                if isle.can_merge_with(new_isle):
                    new_isle_core_debug = isle.merge(new_isle)
                    # merged = True
                    print("[DEBUG] Merged:", isle.core, new_isle.core, "->", new_isle_core_debug)
                    print(self.islands)
                i += 1
            print("*" * 10)

    def generate_mountains(self) -> None:
        if not self.islands:
            print("[WARN] Use generate_islands() before generate_mountains()")
            return

        for r, row in enumerate(self.tiles):
            for c, tile in enumerate(row):
                # Count ocean and plains to calculate odds of generating mountains
                if tile.type == TileType.SAND:
                    num_ocean = num_plains = 0
                    for neighb in self.neighbors((r,c)):
                        r_n, c_n = neighb
                        if self.tiles[r_n][c_n].type == TileType.OCEAN:
                            num_ocean += 1
                        elif self.tiles[r_n][c_n].type == TileType.PLAINS:
                            num_plains += 1
                    # If sand is surrounded by plains, we turn it into a mountain
                    # REVIEW: Division by zero...?
                    mountain_prob = num_plains / (num_plains + num_ocean + 0.1)
                    mountain_flag = weighted_random([True, False],
                                                    [mountain_prob, 1 - mountain_prob])

                    if mountain_flag:
                        self.tiles[r][c].set_type(TileType.MOUNTAINS)

    def neighbors(self, coord_pair):
        '''Generate valid grid neighbors from given coord_pair
        
        # REVIEW: Should this remain agnostic to TileType?
        '''
        r, c = coord_pair
        # REVIEW: Diagonals?
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
                sys.exit()
                for si in self.islands:
                    si.print()
                for j in self.neighbors(t.coord):
                    self.set(j, TileType.DEBUG)
                self.print()
                # sys.exit()
            elif t.island and t.type is not TileType.OCEAN and not t.is_core:
                tt = TileType(list(self.islands).index(t.island) + 1)
                # print("tt:", tt)
                t.set_type(tt)
                #ValueError: <__main__.Island object at 0x10abfa128> is not in list
            # elif not t.island:
            #     print("?")
class Island:
    # TODO Play around with different models
    def __init__(self, core: Tuple[int, int], grid: Grid, generateFlag=True) -> None:
        self.id = uuid.uuid4().hex
        r, c = core

        self.generated = False
        self.grid = grid
        self.core = core
        self.grid.get_tile((r, c)).set_island(self)
        self.grid.get_tile((r, c)).is_core = True
        self.grid.islands[self] = {core}
        self.queue: List[Tuple[int, int]] = []

        if generateFlag:
            self.queue.append(self.core)
            self.generate()

    def __eq__(self, other):
        return self.core == other.core

    def __hash__(self):
        return int(self.id, 16)

    def print(self):
        print("core:", self.core)

    def generate(self) -> None:
        if self.generated:
            sys.exit("[ERROR] Island already generated")
        self.grid.set(self.core, TileType.CORE)
        
        # Enqueue valid neighbors and turn them into TileType.PLAINS
        while self.queue:
            # REVIEW: Sigmoid
            ocean_prob = 1 / (1 + (math.e ** - len(self.grid.islands[self])))
            target = self.queue.pop(0)
            self.process_neighbors(target, ocean_prob)

        self.generated = True # Not particularly useful, just for debug
        print("[DEBUG] Generation of [%s] complete:" % str(self))
        print("[DEBUG] => %d tiles" % len(self.grid.islands[self]))
        # pprint(self.grid.islands)

    def merge(self, other):
        assert(self.grid is other.grid)
        g = self.grid
        isles = g.islands
        
        if self.queue or other.queue:
            print("[ERROR] Cannot add Islands that are still generating")
            return None
        # REVIEW: error out?
        # if not self.can_merge_with(other):
        #     print("[WARNING] Merged islands are not connected!")

        print("[DEBUG: MERGE START]{")
        print(self, end=" "); self.print()
        print(other, end=" "); other.print()
        print(isles)
        print("}")

        print("[DEBUG: MERGE COUNT BEFORE]")
        g.tally()
        # Use midpoint as new core
        # TODO: Ensure it can't be an ocean tile
        r_new = int( (self.core[0] + other.core[0]) / 2 )
        c_new = int( (self.core[1] + other.core[1]) / 2 )
        # Create new island
        isle_new = Island((r_new, c_new), g, generateFlag=False)
        
        # New coords are intersection of merged islands
        isles[isle_new] = isles[self].union(isles[other])
        # Update island reference for all new coords (not needed for every coord, but some)
        for coord_pair in isles[isle_new]:
            g.get_tile(coord_pair).set_island(isle_new)

        # Set tiles
        g.set(self.core, TileType.TEST)
        g.get_tile(self.core).is_core = False

        g.set(other.core, TileType.TEST)
        g.get_tile(other.core).is_core = False

        g.set(isle_new.core, TileType.CORE)

        # Remove old island references so only merged remains
        if self in isles:
            del isles[self]
        else:
            print("[ERROR] self [%s] not in grid.islands" % self)
        if other in isles:
            del isles[other]
        else:
            print("[ERROR] other [%s] not in grid.islands" % other)
        
        print("[DEBUG: MERGE COUNT AFTER]")
        g.tally()

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
            return False

        assert(self.grid is other.grid)
        g = self.grid
        isles = g.islands

        intersects = bool(isles[self].intersection(isles[other]))
        if intersects: # simple case
            print('[DEBUG] Simple case')
            return True

        for c in isles[self]: # Sick Big-O yo
            for n in g.neighbors(c):
                for d in isles[other]:
                    if n == d:
                        print('[DEBUG] Neighbors case')
                        return True

        return False

    def process_neighbors(self, coord_pair, ocean_prob) -> None:
        ''' Go through neighbors of coord_pair and generate tiles'''
        r, c = coord_pair
        g = self.grid
        # REVIEW: Too low and the whole map fills. Too high and you get diamonds
        STABILITY = 0.45 

        for n_coord_pair in g.neighbors(coord_pair):
            r_neighbor, c_neighbor = n_coord_pair 
            # OCEAN -> PLAINS
            if g.get_tile(n_coord_pair).type is TileType.OCEAN:
                # Calculate distance from core to determine probability of 
                # generating edge of island (i.e. shoreline)
                dist_from_core = math.sqrt( (r - r_neighbor) ** 2 + (c - c_neighbor) ** 2 )
                mod = STABILITY * dist_from_core
                shore_flag = weighted_random([True, False], 
                                             [mod * ocean_prob, 1 - (mod * ocean_prob)] )

                if shore_flag:
                    g.set(n_coord_pair, TileType.SAND)
                else:
                    self.queue.append(n_coord_pair)
                    g.islands[self].add(n_coord_pair)
                    g.set(n_coord_pair, TileType.PLAINS)

                g.get_tile(n_coord_pair).set_island(self)

    
if __name__ == "__main__":
    g = Grid(50)
    g.generate(num_islands=4, mountains=True, echo=True)
