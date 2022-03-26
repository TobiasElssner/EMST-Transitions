#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Boundary data structure to compute OrientedVoronoi diagrams.
@author Tobias Elßner
"""
import numpy as np
from Coordinate import Coordinate


class Boundary:

    left_site = None
    right_site = None

    start = Coordinate(0, 0)
    end = Coordinate(0, 0)

    # Indicates if Segment expands to infinity from its starting point
    is_infinite = True

    def __init__(self, start, direction, ls, rs):
        """
        Initializes a Boundary; by default, it is assumed to run into infinity
        :param start: Starting Coordinate
        :param direction: direction, encoded as Coordinate
        :param ls: Left Site, encoded as Coordinate (used for OrientedVoronoi computation)
        :param rs: Right Site, encoded as Coordinate  (used for OrientedVoronoi computation)
        """
        self.start = start

        # End coordinate is the starting coordinate + direction
        # Note that is_infinite is set to True
        self.end = Coordinate(start.x + direction.x, start.y + direction.y)

        self.left_site = ls
        self.right_site = rs

        # Accuracy used for comparisons
        self.eps = 0.01

    def intersects_boundary(self, boundary):
        """
        Following Franklin Antonio,

        Geometric Gems III.
        Code uses the orientation property of the determinant.
        :param boundary: other Boundary
        :return: Boolean whether both boundaries intersect, and their intersection.
                 Note: Intersection is None in case no intersection exists.
        """
        if boundary is None:
            return False, None

        if not self.is_infinite and not boundary.is_infinite:
            if not self.box_test(self.start.x, self.end.x,
                                 boundary.start.x, boundary.end.x):

                return False, None

            if not self.box_test(self.start.y, self.end.y,
                                 boundary.start.y, boundary.end.y):

                return False, None

        ax = self.end.x - self.start.x
        ay = self.end.y - self.start.y

        bx = boundary.start.x - boundary.end.x
        by = boundary.start.y - boundary.end.y

        cx = self.start.x - boundary.start.x
        cy = self.start.y - boundary.start.y

        common_denominator = (ay * bx) - (ax * by)

        # Segments are parallel
        if -self.eps < common_denominator < self.eps:

            return False, None

        alpha_numerator = ((by * cx) - (bx * cy))

        if not self.is_valid_coefficient(alpha_numerator, common_denominator) and not self.is_infinite:

            return False, None

        beta_numerator = ((ax * cy) - (ay * cx)) # np.round(((ax * cy) - (ay * cx)), 2)

        if not self.is_valid_coefficient(beta_numerator, common_denominator) and not boundary.is_infinite:
            return False, None

        alpha = alpha_numerator / common_denominator
        beta = beta_numerator / common_denominator

        # If alpha or beta are below 0, the intersection lies beyond their starting point
        # In this case, the the intersection is also not considered to be valid
        if alpha < -self.eps or beta < -self.eps:

            return False, None

        # If alpha is close to 0, also return None
        if self.eps > alpha > -self.eps and self.eps > beta > -self.eps:

            return False, None

        x = self.start.x + alpha * (self.end.x - self.start.x)
        y = self.start.y + alpha * (self.end.y - self.start.y)

        return True, Coordinate(x, y)

    def is_valid_coefficient(self, numerator, denominator):
        """
        Checks if the coefficient is in range [0, 1]
        :param numerator:
        :param denominator:
        :return: Boolean if coefficient is valid
        """

        if denominator > 0:
            if numerator < 0 or numerator > denominator:
                return False
        else:
            if numerator > 0 or numerator < denominator:
                return False

        return True

    def box_test(self, s1_start, s1_end, s2_start, s2_end):
        """
        Checks if the intersection is within a bounding box (following Franklin Antonio, Geometric Gems III)
        :param s1_start: x- or y-coordinate of the starting point boundary
        :param s1_end: x- or y-coordinate of the end point boundary
        :param s2_start: x- or y-coordinate of the starting point of the second boundary
        :param s2_end: x- or y-coordinate of the end point of the second boundary
        :return: Boolean whether intersection falls into the bounding box
        """
        lo = s1_end
        hi = s1_start

        a = s1_end - s1_start
        b = s2_start - s2_end

        if a >= 0:
            lo = s1_start
            hi = s1_end

        if b > 0:
            if hi < s2_end or s2_start < lo:
                return False
        else:
            if hi < s2_start or s2_end < lo:
                return False

        return True

    def lies_on_segment(self, point):
        """
        Following Allan W. Paeth,
        "A FAST 2-D POINT-ON-LINE TEST", p. 49,
        GRAPHICS GEMS I Edited by ANDREW S. GLASSNER
        Included for completeness, currently not used.
        :param point: 2-D numpy array
        :return: Boolean if point lies on segment
        """

        # point is not on the infinite line formed by start and end
        if np.round(np.abs(((self.end.y - self.start.y) * (point.x - self.start.x)) -
                           ((point.y - self.start.y) * (self.end.x - self.start.x))), 2) \
                >= \
                np.max([np.round(np.abs(self.end.x - self.start.x), 2),
                        np.round(np.abs(self.end.y - self.start.y), 2)]):

            return False

        # point lies on the infinite ray through origin and start
        if ((self.end.x < self.start.x) and (self.start.x < point.x)) or \
                (self.end.y < self.start.y) and (self.start.y < point.y):

            return False

        # point lies on the infinite ray through origin and start
        if ((point.x < self.start.x) and (self.start.x < self.end.x)) or \
                (point.y < self.start.y) and (self.start.y < self.end.y):

            return False

        # point lies on the infinite ray through origin and end
        if ((self.start.x < self.end.x) and (self.end.x < point.x)) or \
                (self.start.y < self.end.y) and (self.end.y < point.y):

            return False

        # point lies on the infinite ray through origin and end
        if ((point.x < self.end.x) and (self.end.x < self.start.x)) or \
                (point.y < self.end.y) and (self.end.y < self.start.y):

            return False

        # Otherwise, point lies on segment <start, end>
        return True

    def to_string(self):
        """
        Summarizes the boundary in a readable string.
        :return: String of the starting- and end point of the boundary
        """
        return "[(" + str(self.start.x) + ", " + str(self.start.y) + ")," \
               + "(" + str(self.end.x) + ", " + str(self.end.y) + ")]"
