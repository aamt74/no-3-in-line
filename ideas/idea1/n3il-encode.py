#!/usr/bin/env python

# Origin down left, for example (lattice 2*2):
#
#         col
#       ---.---
#       |21|22|
#   row ---:---
#       |11|12|
#       ---.---
#

import sys, argparse, math
from dataclasses import dataclass
from itertools import combinations
from typing import List, Set, Dict, Tuple, TypeAlias


FAIL = 1
EPSILON = 1e-10


@dataclass(frozen=True)
class LatticePoint:
    """A point in a lattice of size N*N, with 1 <= row,col <= N"""
    row: int
    col: int
    def __repr__(self):
        return '({0}, {1})'.format(self.row, self.col)


@dataclass(frozen=True)
class LatticeLine:
    """A line in a lattice, determined by its extremes"""
    end1: LatticePoint
    end2: LatticePoint
    slope: float
    def __repr__(self):
        return '{2:5.2f} : {0} --> {1}'.format(self.end1, self.end2, self.slope)
    

PointList: TypeAlias = list[LatticePoint]
PointPair: TypeAlias = tuple[LatticePoint, LatticePoint]


def calc_slope(pt_from: LatticePoint, pt_to: LatticePoint) -> float:
    """Calculates the slope of the line span by the given points"""
    rise = pt_to.row - pt_from.row
    run = pt_to.col - pt_from.col
    if run == 0:
        return math.inf
    gcd = math.gcd(rise, run)
    rise = rise / gcd
    run = run / gcd
    return 1.0 * rise / run


def calc_extremes(N: int, pt1: LatticePoint, pt2: LatticePoint, slope: float) -> PointPair:
    """Calculates the extreme lattice points of a line span by the given point(s) and slope"""
    # make sure column is going up
    if pt1.col > pt2.col:
        pt1, pt2 = pt2, pt1 # trick to swap
    # calc the slope, dealing with vert, hor and sloped lines
    if slope == math.inf:
        return (LatticePoint(row=1, col=pt1.col),
                LatticePoint(row=N, col=pt1.col))
    elif math.isclose(a=slope, b=0, abs_tol=EPSILON):
        return (LatticePoint(row=pt1.row, col=1),
                LatticePoint(row=pt1.row, col=N))
    else:
        end1 = LatticePoint(**pt1.__dict__) # clone
        end2 = LatticePoint(**pt2.__dict__) # clone
        frac_row = float(pt1.row)
        for col in reversed(range(1, pt1.col)):
            frac_row -= slope
            row = round(frac_row)
            if math.isclose(a=frac_row-row, b=0, abs_tol=EPSILON) and 1 <= row <= N:
                end1 = LatticePoint(row, col)
        frac_row = float(pt2.row)
        for col in range(pt2.col+1, N+1):
            frac_row += slope
            row = round(frac_row)
            if math.isclose(a=frac_row-row, b=0, abs_tol=EPSILON) and 1 <= row <= N:
                end2 = LatticePoint(row, col)
        return (end1, end2)


def expand_line(N: int, lattice: PointList, line: LatticeLine) -> PointList:
    """Enumerates all points on a line within the lattice"""
    points_on_line: PointList = []
    if line.slope == math.inf:
        for row in range(N):
            pt = lattice[row * N + (line.end1.col - 1)]
            points_on_line.append(pt)
    elif math.isclose(a=line.slope, b=0, abs_tol=EPSILON):
        for col in range(N):
            pt = lattice[(line.end1.row - 1) * N + col]
            points_on_line.append(pt)
    else:
        points_on_line.append(line.end1)
        pt = line.end1
        frac_row = float(pt.row)
        col = pt.col
        while pt != line.end2:
            col += 1 # make use of columns from calc_extremes go up
            frac_row += line.slope
            row = round(frac_row)
            if math.isclose(a=frac_row-row, b=0, abs_tol=EPSILON):
                pt = lattice[(row - 1) * N + (col - 1)]
                points_on_line.append(pt)
    return points_on_line


def create_lines(N: int, lattice: PointList) -> List[LatticeLine]:
    """Enumerates all (unique) lines in the lattice"""
    # note: extremes in a set uniquely determines the lines!
    extremes: Set[LatticeLine] = set() 
    for i in range(len(lattice)):
        for j in range(i+1, len(lattice)):
            slope = calc_slope(lattice[i], lattice[j])
            ext = calc_extremes(N, lattice[i], lattice[j], slope)
            extremes.add(LatticeLine(end1=ext[0], end2=ext[1], slope=slope))
    sorted = list(extremes)
    sorted.sort(key=lambda x: x.slope)
    return sorted


def create_lattice(N: int) -> PointList:
    """Creates the lattice consisting of N*N points"""
    lattice: PointList = []
    for row in range(1, N+1):
        for col in range(1, N+1):
            lattice.append(LatticePoint(row, col))
    return lattice


def encode(N: int, verbose: bool):
    """Encodes the no-three-in-line problem as a SAT problem"""
    
    # Containers to collect the necessary vars and clauses
    nof_vars = 0
    cls: List[List[int]] = []

    # We need constraints for both points and pairs, and we use these maps to know which pt/pair is which var
    point_to_var_map: Dict[LatticePoint, int] = {}
    pair_to_var_map: Dict[Tuple[LatticePoint, LatticePoint], int] = {}

    # Collect all the vars for the points
    if verbose:
        print("vars:")
    lattice = create_lattice(N)
    for pt in lattice:
        nof_vars += 1
        point_to_var_map[pt] = nof_vars
        if verbose:
            print("  {} |--> x{}".format(pt, nof_vars))

    # Collect all vars and clauses for the pairs and their relationship with points
    if verbose:
        print("lines:")
    lines = create_lines(N, lattice)
    for line in lines:
        if verbose:
            print("  l := {}".format(line))
        
        points_on_line = expand_line(N, lattice, line)
        is_horizontal = math.isclose(a=line.slope, b=0, abs_tol=EPSILON)
        is_vertical = line.slope == math.inf
        is_slanted = not is_horizontal and not is_vertical

        # We can ignore slanted lines with only two points on it, since this is no 3 in line
        if is_slanted and len(points_on_line) == 2:
            continue

        # Create the vars for the pairs
        if verbose:
            print("  pairs:")
        vars_for_line: List[int] = []
        for pair in combinations(points_on_line, 2):
            nof_vars += 1
            pair_to_var_map[pair] = nof_vars
            vars_for_line.append(nof_vars)
            if verbose:
                print("    {} |--> x{}".format(pair, nof_vars))

        # at least one of the pairs must be true on the hor/vert lines
        if is_vertical or is_horizontal :
            cls.append([pair_to_var_map[pair] for pair in combinations(points_on_line, 2)])
        
        # WORKS BUT IS QUITE INEFFICIENT
        # 
        # # no pair of pairs can be true on any line, in particular on the current line
        # for pair_of_pair in combinations(vars_for_line, 2):
        #     cls.append([-pair_of_pair[0], -pair_of_pair[1]])
        # 
        # INSTEAD:
        # 
        if len(points_on_line) > 2: # take N=2 into account
            if verbose:
                print("  pair complements:")
            for pair in combinations(points_on_line, 2):
                # collect all points on the line that are not in pair_var
                other_points_on_line = points_on_line.copy()
                other_points_on_line.remove(pair[0])
                other_points_on_line.remove(pair[1])
            
                # introduce pair_compl_var <=> other_pt_1 || other_pt_1 || ... | other_pt_k
                nof_vars += 1 
                pair_compl_var = nof_vars 
                pair_var = pair_to_var_map[pair]
                cls.append([-pair_var, -pair_compl_var])
                # encode the equivalence of pair_compl_var
                tmp_cls = [point_to_var_map[pt] for pt in other_points_on_line]
                tmp_cls.append(-pair_compl_var)
                cls.append(tmp_cls)
                for pt in other_points_on_line:
                    cls.append([-point_to_var_map[pt], pair_compl_var])
                if verbose:
                    tmp_str = ' || '.join(["x{}".format(point_to_var_map[pt]) for pt in other_points_on_line])
                    print("    x{} |--> {}".format(pair_compl_var, tmp_str))

        # x_n = 1 iff its two points are set too (hence if x_n = -1, then at least one of the points is not set)
        # note: pair_n <=> pt_i && pt_j SAME AS (-pair_n || pt_i) && (-pair_n || pt_j) && (-pt_i || -pt_j || pair_n)
        for pair in combinations(points_on_line, 2):
            pair_var = pair_to_var_map[pair]
            pt1_var = point_to_var_map[pair[0]]
            pt2_var = point_to_var_map[pair[1]]
            cls.append([-pair_var, pt1_var])
            cls.append([-pair_var, pt2_var])
            cls.append([-pt1_var, -pt2_var, pair_var])


    # Output the DIMACS CNF representation    
    print("c SAT-encoding for the no-three-in-line problem of N=%d" % N)
    print("p cnf %d %d" % (nof_vars, len(cls)))
    for c in cls:
        print(" ".join([str(l) for l in c])+" 0")


if __name__ == '__main__':
    try:
        # Parse cli options and arguments
        parser = argparse.ArgumentParser(
            prog="n3il-encode.py",
            description="Encodes the 'no-three-in-line' problem as a SAT problem in DIMACS format",
            epilog="For more information, please consult README.md")
        parser.add_argument("-v", "--verbose", help="add comments to output", action="store_true")
        parser.add_argument("N", type=int, help="the size of the N*N square lattice")
        args = parser.parse_args()
        # Run encoder
        encode(vars(args)["N"], args.verbose)
    except:
        print('error')
        sys.exit(FAIL)
