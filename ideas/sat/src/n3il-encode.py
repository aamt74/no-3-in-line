#!/usr/bin/env python

import sys, argparse, shlex, os
from itertools import combinations
from typing import List, Dict, Tuple
from constants import FAIL
from lattice import Lattice
from symmetry import Symmetry, calc_equiv_points


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
