import math
from dataclasses import dataclass
from typing import Set, TypeAlias, Tuple
from constants import EPSILON

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
        """A line in a lattice, determined by its extremes"""
        end1: "Lattice.Point"
        end2: "Lattice.Point"
        slope: float
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
        (row, col) = row_col
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
        if line.slope == math.inf:
            for row in range(self._N):
                pt = self[row, line.end1.col] #self._points[row * self._N + (line.end1.col - 1)]
                points_on_line.append(pt)
        elif math.isclose(a=line.slope, b=0, abs_tol=EPSILON):
            for col in range(self._N):
                pt = self[line.end1.row, col] #lattice[(line.end1.row - 1) * N + col]
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
                    pt = self[row, col] #lattice[(row - 1) * N + (col - 1)]
                    points_on_line.append(pt)
        return points_on_line
    
    #endregion

    #region private

    def _create_points(self) -> PointList:
        points: Lattice.PointList = []
        for row in range(1, self._N + 1):
            for col in range(1, self._N + 1):
                points.append(Lattice.Point(row, col))
        return points
    
    def _create_lines(self) -> LineList:
        extremes: Set[Lattice.Line] = set()  # use set so that the extremes uniquely determine a lines!
        for i in range(len(self._points)):
            for j in range(i+1, len(self._points)):
                slope = self._calc_slope(self._points[i], self._points[j])
                ext = self._calc_extremes(self._points[i], self._points[j], slope)
                extremes.add(Lattice.Line(end1=ext[0], end2=ext[1], slope=slope))
        sorted = list(extremes)
        sorted.sort(key=lambda x: x.slope)
        return sorted
    
    def _calc_slope(self, pt_from: "Lattice.Point", pt_to: "Lattice.Point") -> float:
        rise = pt_to.row - pt_from.row
        run = pt_to.col - pt_from.col
        if run == 0:
            return math.inf
        gcd = math.gcd(rise, run)
        rise = rise / gcd
        run = run / gcd
        return 1.0 * rise / run

    def _calc_extremes(self, pt1: "Lattice.Point", pt2: "Lattice.Point", slope: float) -> PointPair:
        # make sure column is going up
        if pt1.col > pt2.col:
            pt1, pt2 = pt2, pt1 # trick to swap
        # calc the slope, dealing with vert, hor and sloped lines
        if slope == math.inf:
            return (Lattice.Point(row=1, col=pt1.col),
                    Lattice.Point(row=self._N, col=pt1.col))
        elif math.isclose(a=slope, b=0, abs_tol=EPSILON):
            return (Lattice.Point(row=pt1.row, col=1),
                    Lattice.Point(row=pt1.row, col=self._N))
        else:
            end1 = Lattice.Point(**pt1.__dict__) # clone
            end2 = Lattice.Point(**pt2.__dict__) # clone
            frac_row = float(pt1.row)
            for col in reversed(range(1, pt1.col)):
                frac_row -= slope
                row = round(frac_row)
                if math.isclose(a=frac_row-row, b=0, abs_tol=EPSILON) and 1 <= row <= self._N:
                    end1 = Lattice.Point(row, col)
            frac_row = float(pt2.row)
            for col in range(pt2.col+1, self._N+1):
                frac_row += slope
                row = round(frac_row)
                if math.isclose(a=frac_row-row, b=0, abs_tol=EPSILON) and 1 <= row <= self._N:
                    end2 = Lattice.Point(row, col)
            return (end1, end2)
    
    #endregion
