import math
from dataclasses import dataclass
from typing import Set, TypeAlias, Tuple

# Origin down left, for example (lattice 2*2):
#
#         col
#       ---.---
#       |21|22|
#   row ---:---
#       |11|12|
#       ---.---
#

class Lattice:
    """N*N lattice with all of its lines"""

    #region inner types

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

    #endregion

    #region construction/destruction

    def __init__(self, N: int):
        """Initialize the N*N lattice"""
        self._N = N
        self._points = self._create_points()
        self._lines = self._create_lines()

    #endregion

    #region public

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
        
    #endregion

    #region private

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
