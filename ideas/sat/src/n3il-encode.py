#!/usr/bin/env python

import sys, argparse, math
from itertools import combinations
from typing import List, Dict, Tuple
from constants import EPSILON, FAIL
from lattice import Lattice


def encode(N: int, verbose: bool):
    """Encodes the no-three-in-line problem as a SAT problem"""
    
    # Containers to collect the necessary vars and clauses
    nof_vars = 0
    cls: List[List[int]] = []

    # We need constraints for both points and pairs, and we use these maps to know which pt/pair is which var
    point_to_var_map: Dict[Lattice.Point, int] = {}
    pair_to_var_map: Dict[Tuple[Lattice.Point, Lattice.Point], int] = {}

    # Construct the lattice. This is heavy if N is large. (TODO: multithreading)
    lattice = Lattice(N)

    # Collect all the vars for the points
    if verbose:
        print("vars:")
    for pt in lattice.points:
        nof_vars += 1
        point_to_var_map[pt] = nof_vars
        if verbose:
            print("  {} |--> x{}".format(pt, nof_vars))

    # Collect all vars and clauses for the pairs and their relationship with points
    if verbose:
        print("lines:")
    for line in lattice.lines:
        if verbose:
            print("  l := {}".format(line))
        
        points_on_line = lattice.expand_line(line)
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
        # no pair of pairs can be true on any line, in particular on the current line
        for pair_of_pair in combinations(vars_for_line, 2):
            cls.append([-pair_of_pair[0], -pair_of_pair[1]])
        # 
        # INSTEAD:
        # 
        # if len(points_on_line) > 2: # take N=2 into account
        #     if verbose:
        #         print("  pair complements:")
        #     for pair in combinations(points_on_line, 2):
        #         # collect all points on the line that are not in pair_var
        #         other_points_on_line = points_on_line.copy()
        #         other_points_on_line.remove(pair[0])
        #         other_points_on_line.remove(pair[1])
        #         # introduce pair_compl_var <=> other_pt_1 || other_pt_1 || ... | other_pt_k
        #         nof_vars += 1 
        #         pair_compl_var = nof_vars 
        #         pair_var = pair_to_var_map[pair]
        #         cls.append([-pair_var, -pair_compl_var])
        #         # encode the equivalence of pair_compl_var
        #         tmp_cls = [point_to_var_map[pt] for pt in other_points_on_line]
        #         tmp_cls.append(-pair_compl_var)
        #         cls.append(tmp_cls)
        #         for pt in other_points_on_line:
        #             cls.append([-point_to_var_map[pt], pair_compl_var])
        #         if verbose:
        #             tmp_str = ' || '.join(["x{}".format(point_to_var_map[pt]) for pt in other_points_on_line])
        #             print("    x{} |--> {}".format(pair_compl_var, tmp_str))

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
    except Exception as ex:
        print('error')
        sys.exit(FAIL)
