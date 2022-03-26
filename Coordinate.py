#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Coordinate data structure to compute OrientedVoronoi diagrams.
@author Tobias Elßner
"""
import numpy as np


class Coordinate:
    x = 0
    y = 0

    def __init__(self, x, y):
        """
        Initializes the coordinate.
        :param x: float x-entry
        :param y: float y-entry
        """

        # Shapely has sometimes problems with too many post decimal digits
        # However, rounding leads more often to problems in the calculation of the boundaries and their intersections
        # In the OrientedVoronoi diagram.
        # Therefore, rounding is disabled.
        self.x = x # np.round(x, 4)
        self.y = y # np.round(y, 4)

        # Accuracy, needed for comparisons
        self.eps = 0.01

    def lies_right(self, boundary):
        """
        Checks if the Coordinate lies right to Boundary
        :param boundary: Boundary
        :return: if Coordinate lies right to boundary.
        """

        # If the determinant is below zero (-self.eps) the coordinate lies to the right of the boundary
        return self.determinant(boundary) > self.eps

    def lies_on_boundary(self, boundary):
        """
        Checks if Coordinate lies on boundary
        :param boundary: Boundary
        :return: if Coordinate lies on boundary
        """

        # If determinant is close to zero (between +/- eps)
        return -self.eps > self.determinant(boundary) > self.eps

    def determinant(self, boundary):
        """
        Calculates the determinant/ crossproduct between the Coordinate and the boundary.
        See chapter 33.1, Introductions to Algorithms, 3rd Edition by Cormen et al.
        :param boundary: Boundary
        :return: Determinant
        """

        # positive if point lies to the left, 0 if on, and negative if to the right of boundary
        return (((self.x - boundary.start.x) * (boundary.end.y - boundary.start.y)) -\
               ((boundary.end.x - boundary.start.x) * (self.y - boundary.start.y)))

    def euclidean_distance(self, other):
        """
        Computes the euclidean distance between the Coordinate and another.
        :param other: other Coordinate
        :return: euclidean distance
        """
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def to_string(self):
        """
        Summarizes the coordinate in a readable string
        :return: String containing x- and y-entry of the Coordinate
        """
        return "[" + str(self.x) + ", " + str(self.y) + "]"

    def __eq__(self, other):
        """
        Checks for equality of two coordinates. Needed for hashing.
        :param other: Coordinate
        :return: Boolean if Coordinate is equal to other
        """
        return (self.x, self.y) == (other.x, other.y)

    def __ne__(self, other):
        """
        Checks for unequality of two coordinates. Needed for hashing.
        :param other: Coordinate
        :return: Boolean if Coordinate is unequal to other
        """
        return (self.x, self.y) != (other.x, other.y)

    def __hash__(self):
        """
        Hashes the coordinate.
        :return: hash of x- and y-entry of the Coordinate.
        """
        return hash((self.x, self.y))
