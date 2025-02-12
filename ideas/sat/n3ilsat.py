#!/usr/bin/env python

# ---------------------------------------------------- IMPORTS --------------------------------------------------------
#region

import sys, argparse, shlex, os, math
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


def calc_equiv_points(lattice: Lattice, config: Symmetry) -> Dict[Lattice.Point, Lattice.Point]:
    """Calculates a dictionary that maps a mirrored point to the point which value it takes over"""
    result: Dict[Lattice.Point, Lattice.Point] = {}
    parity = 1 if lattice.N % 2 == 1 else 0
    match config:
        case Symmetry.IDEN:
            pass
        case Symmetry.ROT4:
            for row in range(1, lattice.N // 2 + 1):
                for col in range(1, lattice.N // 2 + parity + 1):
                    result[lattice[lattice.N - col + 1, row]] = lattice[row, col]
                    result[lattice[lattice.N - row + 1, lattice.N - col + 1]] = lattice[row, col]
                    result[lattice[col, lattice.N - row + 1]] = lattice[row, col]
        case _:
            raise NotImplementedError('TODO: implement symmetry case')
    return result

#endregion

# ----------------------------------------------------- ENCODE --------------------------------------------------------
#region

def encode(N: int, symmetry: Symmetry, verbose: bool):
    """Encodes the no-three-in-line problem as a SAT problem"""
    
    # Containers to collect the necessary vars, comments and clauses
    nof_vars = 0
    comments: List[str] = []
    cls: List[List[int]] = []

    # We need constraints for both points and pairs, and we use these maps to know which pt/pair is which var
    point_to_var_map: Dict[Lattice.Point, int] = {}
    pair_to_var_map: Dict[Tuple[Lattice.Point, Lattice.Point], int] = {}

    # Construct the lattice and the point symmetry map.
    lattice = Lattice(N)
    mirror_map = calc_equiv_points(lattice, symmetry)
    
    # Collect all the vars for the points
    if verbose:
        comments.append("POINTS:")
    for pt in lattice.points:
        if pt not in mirror_map:
            nof_vars += 1
            point_to_var_map[pt] = nof_vars
            if verbose:
                comments.append("x{} := {}".format(nof_vars, pt))
    for pt in lattice.points:
        if pt in mirror_map:
            orig_pt = mirror_map[pt]
            orig_var = point_to_var_map[orig_pt]
            point_to_var_map[pt] = orig_var
            if verbose:
                comments.append("x{} ~= {} (symmetry)".format(orig_var, pt))

    # Collect all vars and clauses for the pairs and their relationship with points
    if verbose:
        comments.append("LINES:")
        comments.append("--------------------")
    for line in lattice.lines:
        points_on_line = lattice.expand_line(line)
        if verbose:
            cls.append("line {} = {}".format(line, points_on_line))

        # We can ignore slanted lines with only two points on it, since this is no 3 in line
        if line.is_slanted and len(points_on_line) == 2:
            if verbose:
                cls.append("--------------------")
            continue

        # Create the vars for the pairs
        vars_for_line: List[int] = []
        for pair in combinations(points_on_line, 2):
            nof_vars += 1
            pair_to_var_map[pair] = nof_vars
            vars_for_line.append(nof_vars)
            if verbose:
                pt1_var = point_to_var_map[pair[0]]
                pt2_var = point_to_var_map[pair[1]]
                cls.append("x{} := {} <=> x{} && x{}".format(nof_vars, pair, pt1_var, pt2_var))

        # at least one of the pairs must be true on the hor/vert lines
        if line.is_vertical or line.is_horizontal :
            cls.append([pair_to_var_map[pair] for pair in combinations(points_on_line, 2)])
        
        # no pair of pairs can be true on any line, in particular on the current line
        if len(vars_for_line) > 1:
            for pair_of_pair in combinations(vars_for_line, 2):
                cls.append([-pair_of_pair[0], -pair_of_pair[1]])
        
        # x_n = 1 iff its two points are set too (hence if x_n = -1, then at least one of the points is not set)
        # note: pair_n <=> pt_i && pt_j SAME AS (-pair_n || pt_i) && (-pair_n || pt_j) && (-pt_i || -pt_j || pair_n)
        for pair in combinations(points_on_line, 2):
            pair_var = pair_to_var_map[pair]
            pt1_var = point_to_var_map[pair[0]]
            pt2_var = point_to_var_map[pair[1]]
            cls.append([-pair_var, pt1_var])
            cls.append([-pair_var, pt2_var])
            cls.append([-pt1_var, -pt2_var, pair_var])
        
        if verbose:
            cls.append("--------------------")

    # Output the DIMACS CNF representation
    symm_str = "{}".format(symmetry).replace('Symmetry.', '')
    print("c SAT-encoding for the no-three-in-line problem of N=%d (%s)" % (N, symm_str))
    print("p cnf %d %d" % (nof_vars, len(cls)))
    for msg in comments:
        print("c {}".format(msg))
    for c in cls:
        if isinstance(c, str):
            print("c {}".format(c))
        else:
            print(" ".join([str(l) for l in c])+" 0")

#endregion

# -------------------------------------------------- APPLICATION ------------------------------------------------------
#region

if __name__ == '__main__':
    try:
        # VSCode's ${command:pickArgs} is passed as one string, split it up
        argv = (
            shlex.split(" ".join(sys.argv[1:]))
            if "USED_VSCODE_COMMAND_PICKARGS" in os.environ
            else sys.argv[1:]
        )

        # Parse cli options and arguments
        parser = argparse.ArgumentParser(
            prog="n3il-encode.py",
            description="Encodes the 'no-three-in-line' problem as a SAT problem in DIMACS format",
            epilog="For more information, please consult README.md")
        parser.add_argument("-v", "--verbose", help="add comments to output", action="store_true")
        parser.add_argument("N", type=int, help="the size of the N*N square lattice")
        parser.add_argument("symmetry", type=str, help="flammenkamp's symmetry encoding (. : / - o c x + *)")
        args = parser.parse_args(args=argv)

        # Run encoder
        encode(vars(args)["N"], 
               Symmetry(vars(args)["symmetry"]),
               args.verbose)
    except Exception as ex:
        print('error')
        sys.exit(FAIL)

#endregion