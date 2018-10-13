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
# TODO Fix core ending up in ocean: if core surrounded by ocean -> pick random point in g.islands[self]
    # TODO Also maybe rename g.islands -> g.island_coords
# TODO Add logging
# TODO Make tundra and deserts: use poles and temperature probability gradients



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
    TEST=4
    DEBUG=5
    DEBUG2=6
    CORE=7


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
        self.islands: DefaultDict[Island, Set[Tuple[int, int]]] = defaultdict(set) 
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


    def tally2(self):
        counter = {}
        for isle in self.islands:
            counter[isle] = len(self.islands[isle])
        pprint(counter)

    def print(self) -> None:
        display_scheme = {
            TileType.OCEAN.value: (Fore.BLUE, "0"),
            TileType.PLAINS.value: (Fore.GREEN, "1"),
            TileType.SAND.value: (Fore.YELLOW, "2"),
            TileType.MOUNTAINS.value: (Fore.WHITE, "3"),
            TileType.CORE.value: (Fore.RED, "C"),
            TileType.TEST.value: (Fore.RED, "4"),
            TileType.DEBUG.value: (Fore.BLACK, "&"),
            TileType.DEBUG2.value: (Fore.CYAN, "?")
        }

        for row in self.tiles:
            for tile in row:
                scheme = display_scheme[tile.type.value]
                print(scheme[0] + scheme[1] + Style.RESET_ALL, end=" ")
            print()

    def get_random_loc(self, ocean=True) -> Tuple[int, int]:
        '''Get a random coordinate from the grid
        
        # REVIEW Perhaps make a wrapper function that only gets random ocean tiles? 
        '''
        not_valid = True
        while not_valid:
            r = random.randrange(0, self.size)
            c = random.randrange(0, self.size)
            tile = self.get_tile((r,c))

            if tile.type is TileType.OCEAN:
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
            Island(self.get_random_loc(), self)
            # REVIEW I cannot believe calling it twice fixes the bug
            # Why? Because sometimes a merge would then create an eligible
            # subsequent merge that would not be processed

            # TODO Make this more elegant. Perhaps something tracking the isles
            # being processed outside of the scope of merge_islands() would allow
            # us to loop through until we're done
            self.merge_islands()
            print("[DEBUG] Second call")
            self.merge_islands()

    def merge_islands(self):
        processed: Dict[Island, bool] = {isle: False for isle in self.islands}
        merge_queue = list(itertools.combinations(self.islands, 2))
        merged = []
        for isle1, isle2 in merge_queue:
            valid_merge = not processed[isle1] and not processed[isle2]
            if valid_merge and isle1.can_merge_with(isle2):
                new_isle, union = isle1.merge(isle2)
                merged.append( (isle1, isle2, new_isle, union) )
                print("[DEBUG] Merged: %s(%d,%d) + %s(%d,%d) -> %s(%d,%d)" % (
                    isle1, *isle1.core, 
                    isle2, *isle2.core,
                    new_isle, *new_isle.core)
                )
                processed[isle1] = processed[isle2] = True

        
        if merged:
            print("[DEBUG] Finalizing merge")
        for quartet in merged:
            isl1, isl2, isl_new, union = quartet

            for coord_pair in union:
                self.get_tile(coord_pair).set_island(isl_new)
            self.islands[isl_new] = union
            isl_new.setup_core()

            self.set(isl1.core, TileType.TEST)
            self.get_tile(isl1.core).is_core = False
            del self.islands[isl1]

            self.set(isl2.core, TileType.TEST)
            self.get_tile(isl2.core).is_core = False
            del self.islands[isl2]

        print("*" * 20)
        self.tally()
        print("." * 20)
        self.tally2()
        print("*" * 20)


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
            elif t.island and t.type is not TileType.OCEAN and not t.is_core:
                t.set_type(TileType(list(self.islands).index(t.island) + 1))

class Island:
    # TODO Play around with different models
    def __init__(self, core: Tuple[int, int], grid: Grid, generateFlag=True) -> None:
        r, c = core
        self.id = uuid.uuid4().hex
        self.generated = False
        self.grid = grid
        self.core = core
        self.queue: List[Tuple[int, int]] = []
        self.generateFlag = generateFlag

        if self.generateFlag:
            self.generate()

    def __eq__(self, other):
        return self.core == other.core

    def __hash__(self):
        return int(self.id, 16)

    def print(self):
        print("core:", self.core)

    def setup_core(self):
        self.grid.get_tile(self.core).set_island(self)
        self.grid.get_tile(self.core).is_core = True
        self.grid.set(self.core, TileType.CORE)
        self.grid.islands[self].add(self.core)

    def generate(self) -> None:
        if self.generated:
            sys.exit("[ERROR] Island already generated")

        self.setup_core()
        self.queue.append(self.core)
        
        # Enqueue valid neighbors and turn them into TileType.PLAINS
        while self.queue:
            # REVIEW: Sigmoid
            ocean_prob = 1 / (1 + (math.e ** - len(self.grid.islands[self])))
            target = self.queue.pop(0)
            self.process_neighbors(target, ocean_prob)

        self.generated = True # Not particularly useful, just for debug
        print("[DEBUG] Generation of %s(%d, %d) complete:" % (str(self), *self.core))
        print("[DEBUG] => %d tiles" % len(self.grid.islands[self]))
        # self.grid.tally()

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
                    g.set(n_coord_pair, TileType.PLAINS)

                g.islands[self].add(n_coord_pair)
                g.get_tile(n_coord_pair).set_island(self)

    def merge(self, other):
        assert(self.grid is other.grid)
        g = self.grid
        isles = g.islands
        
        if self.queue or other.queue:
            print("[ERROR] Cannot add Islands that are still generating")
            return None
        # TODO: Ensure it can't be an ocean tile
        r_new = int( (self.core[0] + other.core[0]) / 2 )
        c_new = int( (self.core[1] + other.core[1]) / 2 )
        # Create new island
        isle_new = Island((r_new, c_new), g, generateFlag=False)
        union = isles[self].union(isles[other])
        
        return isle_new, union
    
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
        if intersects:
            print('[DEBUG] Simple case')
            return True

        ''' DEBUG
        debug_map = [["." for c in range(g.size)] for r in range(g.size)]
        for coord in isles[self]:
            r, c = coord
            debug_map[r][c] = Fore.GREEN + "X" + Style.RESET_ALL
        for coord in isles[other]:
            r, c = coord
            debug_map[r][c] = Fore.YELLOW + "Y" + Style.RESET_ALL
        '''

        for self_coord in isles[self]: # Sick Big-O yo
            for self_neighb in g.neighbors(self_coord):
                if self_neighb in isles[other]:
                    print('[DEBUG] Neighbors case')
                    return True
                # r_s, c_s = self_neighb
                # if debug_map == Fore.YELLOW + "Y" + Style.RESET_ALL:
                #     print("sus")
                # debug_map[r_s][c_s] = Fore.RED + "X" + Style.RESET_ALL
            

        # out = ""   
        # for row in debug_map:
        #     for item in row:
        #         out += item + " "
        #     out += "\n"
        # print(out)
        # input()

        return False


if __name__ == "__main__":
    g = Grid(50)
    g.generate(num_islands=4, mountains=True, echo=True)
