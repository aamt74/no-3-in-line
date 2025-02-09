from enum import Enum
from lattice import Lattice
from typing import Dict


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
