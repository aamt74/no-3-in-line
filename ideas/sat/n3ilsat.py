#!/usr/bin/env python

# ---------------------------------------------------- IMPORTS --------------------------------------------------------
#region

import sys, argparse, shlex, os, math, png
from itertools import combinations
from typing import Set, TypeAlias, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum

#endregion

# --------------------------------------------------- CONSTANTS -------------------------------------------------------
#region

OK = 0          # return code if program runs without error
FAIL = 1        # return code if program runs erroneously
EPSILON = 1e-10 # accuracy in comparing floating points
PNG_FACTOR = 20 # in png every lattice point is scaled with this factor

#endregion

# ---------------------------------------------------- LATTICE --------------------------------------------------------
#region

class Lattice:
    """N*N lattice with all of its lines"""
    
    @dataclass(frozen=True)
    class Point:
        """A point in a lattice of size N*N, with 1 <= row,col <= N"""
        row: int
        col: int
        def __repr__(self):
            return '({0}, {1})'.format(self.row, self.col)

    @dataclass(frozen=True)
    class Line:
        """A line in a lattice, (end1.col <= end2.col)"""
        end1: "Lattice.Point"
        end2: "Lattice.Point"
        rise: int
        run: int
        @property
        def slope(self) -> float:
            return math.inf if self.run == 0 else 1.0 * self.rise / self.run
        @property
        def is_vertical(self) -> bool:
            return self.run == 0
        @property
        def is_horizontal(self) -> bool:
            return self.run != 0 and self.rise == 0
        @property
        def is_slanted(self) -> bool:
            return self.run != 0 and self.rise != 0
        def __repr__(self):
            return '{2:5.2f} : {0} --> {1}'.format(self.end1, self.end2, self.slope)
    
    PointList: TypeAlias = list["Lattice.Point"]
    PointPair: TypeAlias = tuple["Lattice.Point", "Lattice.Point"]
    LineList: TypeAlias = list["Lattice.Line"]

    def __init__(self, N: int):
        """Initialize the N*N lattice"""
        self._N = N
        self._points = self._create_points()
        self._lines = self._create_lines()

    def __getitem__(self, row_col: Tuple[int, int]):
        """Index operator, used like 'pt = lattice_instance[row, col]'"""
        row, col = row_col
        return self._points[(row - 1) * self._N + (col - 1)]
    
    @property
    def N(self) -> int:
        """Returns the value N of this N*N lattice"""
        return self._N
    
    @property
    def points(self) -> "Lattice.PointList":
        """Returns all points in the lattice, ordered from left-bottom to right-upper"""
        return self._points
    
    @property
    def lines(self) -> "Lattice.LineList":
        """Returns all lines in the lattice, ordered on slope ascendingly"""
        return self._lines
    
    def expand_line(self, line: "Lattice.Line") -> PointList:
        """Enumerates all points on a line within the lattice"""
        points_on_line: Lattice.PointList = []
        if line.is_vertical:
            for row in range(1, self._N + 1):
                pt = self[row, line.end1.col]
                points_on_line.append(pt)
            return points_on_line
        elif line.is_horizontal:
            for col in range(1, self._N + 1):
                pt = self[line.end1.row, col]
                points_on_line.append(pt)
            return points_on_line
        elif line.is_slanted:
            points_on_line.append(line.end1)
            pt = line.end1
            row = pt.row
            col = pt.col
            while pt != line.end2:
                row += line.rise
                col += line.run
                pt = self[row, col]
                points_on_line.append(pt)
            return points_on_line
        else:
            raise ValueError("Line created from a single point?")
    
    def incidents(self, point: "Lattice.Point") -> LineList:
        """Enumerates all lines a point is incident with"""
        result: Lattice.LineList = []
        for line in self._lines:
            points_on_line = self.expand_line(line)
            if point in points_on_line:
                result.append(line)
        return result

    def verify(self, placed_points: "PointList") -> bool:
        """Checks if (1) h/v lines have 2 pts and (2) no slanted line has more than 2 pts"""
        nof_pt_on_line: Dict[Lattice.Line, int] = {}
        for point in placed_points:
            incident_lines = self.incidents(point)
            for line in incident_lines:
                if line not in nof_pt_on_line:
                    nof_pt_on_line[line] = 1
                else:
                    nof_pt_on_line[line] += 1
        is_valid = True
        for line, incidents in nof_pt_on_line.items():
            if line.is_slanted:
                if incidents > 2:
                    print(line, line.run, line.rise)
                    print(incidents)
                    is_valid = False
                    break
            else: # hor or ver line
                if incidents != 2:
                    is_valid = False
                    break
        return is_valid

    def _create_points(self) -> PointList:
        points: Lattice.PointList = []
        for row in range(1, self._N + 1):
            for col in range(1, self._N + 1):
                pt = Lattice.Point(row, col)
                points.append(pt)
        return points
    
    def _create_lines(self) -> LineList:
        lines: Set[Lattice.Line] = set()  # use set so that the extremes uniquely determine a lines!
        for i in range(len(self._points)):
            for j in range(i+1, len(self._points)):
                rise, run = self._calc_slope(self._points[i], self._points[j])
                end1, end2 = self._calc_extremes(self._points[i], self._points[j], rise, run)
                l = Lattice.Line(end1, end2, rise, run)
                lines.add(l)
        sorted = list(lines)
        sorted.sort(key=lambda x: x.slope)
        return sorted
    
    def _calc_slope(self, pt_from: "Lattice.Point", pt_to: "Lattice.Point") -> Tuple[int, int]:
        rise = pt_to.row - pt_from.row
        run = pt_to.col - pt_from.col
        if run == 0:
            return (rise, run)
        gcd = math.gcd(rise, run)
        rise = rise // gcd
        run = run // gcd
        return (rise, run)

    def _calc_extremes(self, pt1: "Lattice.Point", pt2: "Lattice.Point", rise: int, run: int) -> PointPair:
        if pt1.col > pt2.col: # make sure column is going up
            pt1, pt2 = pt2, pt1
        if run == 0: # vertical
            return (Lattice.Point(row=1, col=pt1.col),
                    Lattice.Point(row=self._N, col=pt1.col))
        elif rise == 0: # horizontal
            return (Lattice.Point(row=pt1.row, col=1),
                    Lattice.Point(row=pt1.row, col=self._N))
        else: # slanted
            end1 = Lattice.Point(**pt1.__dict__) # clone
            while 1 <= end1.col - run <= self._N and 1 <= end1.row - rise <= self._N:
                end1 = Lattice.Point(end1.row - rise, end1.col - run)
            end2 = Lattice.Point(**pt2.__dict__) # clone
            while 1 <= end2.col + run <= self._N and 1 <= end2.row + rise <= self._N:
                end2 = Lattice.Point(end2.row + rise, end2.col + run)
            return (end1, end2)

#endregion

# ---------------------------------------------------- SYMMETRY -------------------------------------------------------
#region

class Symmetry(Enum):
    """Flammenkamp's symmetry configurations"""
    IDEN = '.'
    ROT2 = ':'
    DIA1 = '/'
    ORT1 = '-'
    ROT4 = 'o'
    NEAR = 'c'
    DIA2 = 'x'
    ORT2 = '+'
    FULL = '*'


def calc_equiv_points(lat: Lattice, sym: Symmetry) -> Dict[Lattice.Point, Lattice.Point]:
    """Calculates a dictionary that maps a mirrored point to the point which value it takes over"""
    result: Dict[Lattice.Point, Lattice.Point] = {}
    parity = 1 if lat.N % 2 == 1 else 0
    match sym:
        case Symmetry.IDEN:
            pass
        case Symmetry.ROT4:
            for row in range(1, lat.N // 2 + 1):
                for col in range(1, lat.N // 2 + parity + 1):
                    result[lat[lat.N - col + 1, row]] = lat[row, col]
                    result[lat[lat.N - row + 1, lat.N - col + 1]] = lat[row, col]
                    result[lat[col, lat.N - row + 1]] = lat[row, col]
        case _:
            raise NotImplementedError('TODO: implement symmetry case')
    return result

#endregion

# ---------------------------------------------------- IMAGING --------------------------------------------------------
#region

def create_png(N: int, placed_points: Lattice.PointList):
    width = N * PNG_FACTOR + N + 1
    height = N * PNG_FACTOR + N + 1
    img = []
    pixrow = ()
    for _ in range(width):
        pixrow = pixrow + (0, 0, 0)
    img.append(pixrow)
    for row in range(1,N+1):
        pixrow = ()
        pixrow = pixrow + (0, 0, 0)
        for col in range(1,N+1):
            if Lattice.Point(row, col) in placed_points:
                for _ in range(PNG_FACTOR):
                    pixrow = pixrow + (255, 0, 0)
                pixrow = pixrow + (0, 0, 0)
            else:
                for _ in range(PNG_FACTOR):
                    pixrow = pixrow + (255, 255, 255)
                pixrow = pixrow + (0, 0, 0)
        for _ in range(PNG_FACTOR):  
            img.append(pixrow)
        pixrow = ()
        for _ in range(width):
            pixrow = pixrow + (0, 0, 0)
        img.append(pixrow)
        
    with open('{}x{}-solution.png'.format(N, N), 'wb') as f:
        w = png.Writer(width, height, greyscale=False)
        w.write(f, img)

#endregion

# ----------------------------------------------------- ENCODE --------------------------------------------------------
#region

Variable: TypeAlias = int
PointVar: TypeAlias = Variable
PairVar: TypeAlias = Variable
PointVarPair: TypeAlias = tuple[PointVar]
PointMap: TypeAlias = dict[Lattice.Point, PointVar]
PairMap: TypeAlias = dict[PointVarPair, PairVar]
LineMap: TypeAlias = dict[Lattice.Line, set[PairVar]]
Clause: TypeAlias = tuple[Variable]
ClauseSet: TypeAlias = set[Clause]


def create_point_map(lat: Lattice, sym: Symmetry) -> PointMap:
    """Creates variables for the lattice points, which form a bijection: (1,1) |-> 1, (1,2) |-> 2, etc"""
    point_map: PointMap = {}
    mirror_map = calc_equiv_points(lat, sym)
    nof_vars = 0
    for pt in lat.points:
        if pt not in mirror_map:
            nof_vars += 1
            point_map[pt] = nof_vars
    for pt in lat.points:
        if pt in mirror_map:
            orig_pt = mirror_map[pt]
            orig_var = point_map[orig_pt]
            point_map[pt] = orig_var
    return point_map


def create_pair_map(lat: Lattice, point_map: PointMap) -> PairMap:
    """Creates variables for unique pairs throughout the lattice"""
    pair_map: PairMap = {}
    unique_pairs: Set[PointVarPair] = set()
    nof_vars = len(set(point_map.values()))
    for line in lat.lines:
        points_on_line = lat.expand_line(line)
        if line.is_slanted and len(points_on_line) == 2:
            continue
        for pair in combinations(points_on_line, 2):
            var_1st = point_map[pair[0]]
            var_2nd = point_map[pair[1]]
            unique_pair = PointVarPair(sorted((var_1st, var_2nd)))
            if unique_pair not in unique_pairs:
                nof_vars += 1
                pair_map[unique_pair] = nof_vars
                unique_pairs.add(unique_pair)
    return pair_map


def create_line_map(lat: Lattice, point_map: PointMap, pair_map: PairMap) -> LineMap:
    """Collects the variables for the pairs on a line """
    line_map: LineMap = {}
    for line in lat.lines:
        unique_pair_vars: Set[PairVar] = set()
        points_on_line = lat.expand_line(line)
        if line.is_slanted and len(points_on_line) == 2:
            continue
        for pair in combinations(points_on_line, 2):
            var_1st = point_map[pair[0]]
            var_2nd = point_map[pair[1]]
            unique_pair = PointVarPair(sorted((var_1st, var_2nd)))
            unique_pair_vars.add(pair_map[unique_pair])
        line_map[line] = unique_pair_vars
    return line_map


def encode_rot4(lat: Lattice, point_map: PointMap, pair_map: PairMap, line_map: LineMap) -> ClauseSet:
    result: ClauseSet = set()
    parity = 1 if lat.N % 2 == 1 else 0
    for i in range(1, lat.N // 2 + parity):
        lines = lat.incidents(Lattice.Point(i, i))
        hline: Lattice.Line = None
        vline: Lattice.Line = None
        for line in lines:
            if line.is_horizontal:
                hline = line
            elif line.is_vertical:
                vline = line
            if (hline is not None) and (vline is not None):
                break
        points: Set[PointVar] = set()
        points.update(lat.expand_line(hline))
        points.update(lat.expand_line(vline))
        vars: List[PointVar] = []
        for pt in points:
            is_not_left_of_i_i: bool = i <= pt.col <= lat.N // 2 + parity
            is_not_down_of_i_i: bool = i <= pt.row <= lat.N // 2
            if is_not_left_of_i_i and is_not_down_of_i_i:
                vars.append(point_map[pt])
        pair_vars: Set[PairVar] = set()
        for pair in combinations(vars, 2):
            var_1st = pair[0]
            var_2nd = pair[1]
            unique_pair = PointVarPair(sorted((var_1st, var_2nd)))
            pair_vars.add(pair_map[unique_pair])
        if len(pair_vars) > 1:
            for pair_of_pair_vars in combinations(pair_vars, 2):
                clause = Clause(sorted((-pair_of_pair_vars[0], -pair_of_pair_vars[1]))) # at most one pair
                result.add(clause)
    return result


def encode(N: int, sym: Symmetry):
    """Encodes the no-three-in-line problem as a SAT problem"""

    # Transform to SAT:
    lat = Lattice(N)
    point_map = create_point_map(lat, sym)
    pair_map = create_pair_map(lat, point_map)
    line_map = create_line_map(lat, point_map, pair_map)
    nof_point_vars = len(set(point_map.values()))
    nof_pair_vars = len(set(pair_map.values()))
    clauses: ClauseSet = set()
    for line in lat.lines:
        if line not in line_map:
            continue
        pair_vars = line_map[line]
        if line.is_vertical or line.is_horizontal:
            clause = Clause(pair_vars) # at least one pair on hor,vert lines
            clauses.add(clause)
        if len(pair_vars) > 1:
            for pair_of_pair_vars in combinations(pair_vars, 2):
                clause = Clause(sorted((-pair_of_pair_vars[0], -pair_of_pair_vars[1]))) # at most one pair on any line
                clauses.add(clause)
    for point_var_pair, pair_var in pair_map.items():
        pt1_var = point_var_pair[0]
        pt2_var = point_var_pair[1]
        clauses.add(Clause(sorted((-pair_var, pt1_var))))               # (x,y) = 1 <=> x = 1 && y = 1
        clauses.add(Clause(sorted((-pair_var, pt2_var))))               # :
        clauses.add(Clause(sorted((-pt1_var, -pt2_var, pair_var))))     # :

    # Transformations specific to the symmetry
    match sym:
       case Symmetry.IDEN: pass
       case Symmetry.ROT4:
            for clause in encode_rot4(lat, point_map, pair_map, line_map):
                clauses.add(clause)
       case _:
           raise NotImplementedError('TODO: implement symmetry case')

    # Output the DIMACS CNF representation
    symm_str = "{}".format(sym).replace('Symmetry.', '')
    print("c SAT-encoding for the no-three-in-line problem of dimension N=%d (%s)" % (N, symm_str))
    print("c Generated by n3ilsat.py")
    print("p cnf %d %d" % (nof_point_vars + nof_pair_vars, len(clauses)))
    for cls in clauses:
        print(" ".join([str(var) for var in cls])+" 0")

#endregion

# ----------------------------------------------------- DECODE --------------------------------------------------------
#region

def decode_iden(N: int, sat_model: List[int]) -> Lattice.PointList:
    """Decodes the sat model, knowing that it was generated with identity symmetry"""
    placed_points: Lattice.PointList = []
    row = 1
    col = 1
    for var in sat_model:
        if var > 0:
            placed_points.append(Lattice.Point(row, col))
        col += 1
        if col > N:
            col = 1
            row += 1
        if row > N:
            break
    return placed_points


def decode_rot4(N: int, sat_model: List[int]) -> Lattice.PointList:
    """Decodes the sat model, knowing that it was generated with rot4 symmetry"""
    placed_points: Lattice.PointList = []
    row = 1
    col = 1
    parity = 1 if N % 2 == 1 else 0
    for var in sat_model:
        if var > 0:
            placed_points.append(Lattice.Point(row, col))
            placed_points.append(Lattice.Point(N - col + 1, row))
            placed_points.append(Lattice.Point(N - row + 1, N - col + 1))
            placed_points.append(Lattice.Point(col, N - row + 1))
        col += 1
        if col > N // 2 + parity:
            col = 1
            row += 1
        if row > N // 2:
            break
    return placed_points


def decode(N: int, sym: Symmetry, emit_png: bool):
    """Read SAT-model from stdin, check if proper solution, and optionally emit png"""

    # read SAT-model
    model_str: str = ''
    for line in sys.stdin:
        model_str += line.rstrip()
    sat_model = [int(x) for x in model_str.split(' ')]

    # decode
    placed_points: Lattice.PointList
    match sym:
       case Symmetry.IDEN: placed_points = decode_iden(N, list(sat_model))
       case Symmetry.ROT4: placed_points = decode_rot4(N, list(sat_model))
       case _:
           raise NotImplementedError('TODO: implement symmetry case')

    # verify
    lat = Lattice(N)
    is_valid = lat.verify(placed_points)
    print("valid" if is_valid else "not valid")

    # emit desired output formats
    if emit_png:    
        create_png(N, placed_points)

#endregion

# -------------------------------------------------- APPLICATION ------------------------------------------------------
#region

def create_options() -> argparse.ArgumentParser:
    """Setup the options that the application accepts"""
    parser = argparse.ArgumentParser(
        prog="n3ilsat.py",
        description="Solves the 'no-three-in-line' problem using SAT.\n"
                    "Encode writes problem as dimacs to stdout.\n"
                    "Decode reads solution in dimacs from stdin.\n",
        epilog="For more information, please consult README.md",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--png", help="generate png while decoding", action="store_true")
    parser.add_argument("verb", choices=['encode', 'decode'], help="encode/decode to/from dimacs")
    parser.add_argument("N", type=int, help="the size of the N*N square lattice")
    parser.add_argument("symmetry", type=str, help="flammenkamp's (. : / - o c x + *)")
    return parser


if __name__ == '__main__':
    try:
        # VSCode's ${command:pickArgs} is passed as one string, split it up
        argv = (
            shlex.split(" ".join(sys.argv[1:]))
            if "USED_VSCODE_COMMAND_PICKARGS" in os.environ
            else sys.argv[1:]
        )

        # Parse cli options and arguments
        parser = create_options()
        args = parser.parse_args(args=argv)
        N = vars(args)['N']
        sym = Symmetry(vars(args)["symmetry"])
        emit_png = args.png

        # Run
        match vars(args)['verb']:
            case 'encode': encode(N, sym)
            case 'decode': decode(N, sym, emit_png)
                
    except Exception as ex:
        print('error')
        sys.exit(FAIL)

#endregion