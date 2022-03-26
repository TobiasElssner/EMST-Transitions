#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author Tobias Elßner
Based on Monma & Suri (1992)
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean as eu
from itertools import chain, combinations
from shapely.geometry import Point, Polygon, LineString, JOIN_STYLE, LinearRing
from shapely.ops import split, polygonize, unary_union, polygonize_full
from shapely.ops import split, polygonize
import pylab as pl
from scipy.spatial.distance import euclidean as eu
from Coordinate import Coordinate
from Boundary import Boundary
from matplotlib import pyplot
from shapely.geometry.polygon import LinearRing, Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import collections as mc
import numpy as np
import matplotlib.pyplot as plt

from Node import QueueNode
from OrientedVoronoi import OrientedVoronoi
from Boundary import Boundary
from Coordinate import Coordinate
from PrimMST import PrimMST
from PriorityQueue import PriorityQueue


class Subdivision:

    def __init__(self, dp):
        """
        Initializes the Subdivision.
        :param dp: 2-D Numpy Data-array containing the coordinates of the MST nodes
        """
        self.data_points = dp

        # Refactoring for better readability
        self.num_of_points = self.data_points.shape[0]

        # Number of subregions
        self.num_of_regions = 0

        # The bounding box is specified by the lower left and upper right coordinates
        self.bounding_box = []

        # Contains all four corners of the box
        self.box_corners = []

        # Contains the directions of the boxes' boundaries )
        self.box_directions = []

        self.num_of_erroneous_regions = 0
        self.erroneous_regions = []
        self.correct_regions = []

        self.area_of_erroneous_regions = 0.0
        self.total_area = 0.0

        # The universal environment 'site' in which the diagrams are constructed
        self.environment = Coordinate(-np.infty, -np.infty)

        self.mst = PrimMST(self.data_points)

        self.topology_to_regions = {}

        # Specify the inner angle of the cones for the oriented OrientedVoronoi Diagrams
        # Following Yao, they have to by smaller than 14.4775...° degrees
        self.degrees = 10.0

        # Precision used to round shapely polygons
        # https://gis.stackexchange.com/questions/277334/shapely-polygon-union-results-in-strange-artifacts-of-tiny-non-overlapping-area
        self.eps = 0.01

        # Arrange the polygons in the order of the data points / sites
        united_regions = self.compute_oriented_voronoi_regions()

        # Compute the overlay of the oriented OrientedVoronoi Diagrams
        overlay = self.overlay(united_regions)

        # Store the regions for each topology in a dictionary
        self.topology_to_regions = {}

        self.subdivide(overlay)

    def compute_oriented_voronoi_regions(self):
        """
        Computes all oriented OrientedVoronoi Diagrams.
        :return: [shapely-Polygon] List of united oriented OrientedVoronoi regions for each site
        """

        # Construct a rotation matrix with rotation angle *degrees*
        # Convert angle into radiant
        rad = (np.pi * self.degrees) / 90

        cos = np.cos(rad)
        sin = np.sin(rad)
        rotation_matrix = np.array([[cos, -sin], [sin, cos]])

        # Build the frame, a collection of basis vectors sharing a 10 degree angle
        basis_1 = np.array([1, 0])

        # Contains all regions
        diagrams = []

        # Contains only the open regions
        open_regions = []

        # Corners of the box which fits all diagrams
        lower_left_x = 0.0
        lower_left_y = 0.0
        upper_right_x = 0.0
        upper_right_y = 0.0

        for i in range(int(360 / self.degrees)):
            basis_2 = np.matmul(rotation_matrix, basis_1)
            basis_2 /= np.linalg.norm(basis_2)

            # basis = np.round(np.array([basis_2, basis_1.transpose()]).transpose(), 2)
            basis = np.array([basis_2, basis_1.transpose()]).transpose()

            ovd = OrientedVoronoi(self.data_points, basis, self.environment)

            if lower_left_x > ovd.lower_left_x:
                lower_left_x = ovd.lower_left_x
            if lower_left_y > ovd.lower_left_y:
                lower_left_y = ovd.lower_left_y
            if upper_right_x < ovd.upper_right_x:
                upper_right_x = ovd.upper_right_x
            if upper_right_y < ovd.upper_right_y:
                upper_right_y = ovd.upper_right_y

            diagrams.append(ovd.regions_to_boundaries)

            # Store the open regions for each oriented voronoi diagram
            # The open regions can be closed, when the bounding box has been computed
            open_regions.append(ovd.open_regions)

            # Move on to the next sector
            basis_1 = basis_2

        # Compute a bounding box from the global extrema
        self.bounding_box, self.box_corners, self.box_directions = \
            self.compute_bounding_box(lower_left_x, lower_left_y, upper_right_x, upper_right_y)

        # Arrange the polygons in the order of the data points / sites
        # This way, it can be determined which intersected polygon has which sites as geographic neighbors
        united_regions = []

        # Loop over all vertices
        for j in range(len(self.data_points)):

            site = Coordinate(self.data_points[j][0], self.data_points[j][1])

            polygon = Polygon()

            # And loop over all diagrams
            for i in range(int(360 / self.degrees)):

                o_r = open_regions[i]
                diag = diagrams[i]

                edges = diag[site]

                # Close any open polygon at the calculated border
                if site in o_r:
                    edges = self.close_polygon(site, edges)

                # Turn the edges into a shapely polygon
                geometry_collection = list(polygonize_full(edges))[0]

                # The result can possibly be multiple geometric objects
                for geometry in geometry_collection.geoms:

                    # Filter only the non-empty valid polygons
                    if geometry.geom_type == 'Polygon' and not geometry.is_empty and geometry.is_valid:

                        # Add small buffer to avoid self-intersections around the vertex
                        # I.e. that the polygon contains only a single coordinate at the data point
                        polygon = polygon.union(geometry.buffer(self.eps))

                        # Expand and shrink the composed polygon to avoid artifacts along former borders
                        polygon = polygon.buffer(self.eps, 1, join_style=JOIN_STYLE.mitre). \
                            buffer(-self.eps, 1, join_style=JOIN_STYLE.mitre)

            # Undo the buffer to remove remaining borders regions
            polygon = polygon.buffer(-self.eps)

            # Expand and shrink the composed polygon to avoid artifacts along former borders
            polygon = polygon.buffer(self.eps, 1, join_style=JOIN_STYLE.mitre). \
                buffer(-self.eps, 1, join_style=JOIN_STYLE.mitre)

            united_regions.append(polygon)

        return united_regions

    def overlay(self, united_sites):
        """
        Overlays the united regions of oriented OrientedVoronoi Diagrams.
        :param united_sites: List of  united shapely-polygons for every site
        :return: A list of (Shapely-objects, tuple(integer)) tuples,
                 containing the indices of datapoints whose OVD regions overlap at Shapely-object (and are therefore
                 their nearest neighbors)
        """

        """
        geopandas Overlay proves to be too slow in practice:
        https://github.com/geopandas/geopandas/issues/706

        Shapely distinguishes critically between geometrical objects, such as LineStrings, Polygons, and Multipolygons.
        Although all oriented OrientedVoronoi region are of type Polygon, intersections and differences may result in other
        objects. This can be worsened by internal rounding errors. Therefore, after each operation, all resulting
        objects need to be examined. However, these loops slow the program down, already for a minimum number of two
        vertices.

        The solution instead is to compute every possible overlap of *united* sites (i.e., the union of all oriented
        OrientedVoronoi regions of one vertex) individually. Although this takes O(n³), because for each intersection of
        polygons p1 and p2, all other possibly intersecting polygons need to be subtracted, it runs in practice much
        faster than the above pseudo-code.
        Therefore, all intersections unfortunately need to be calculated individually.
        """
        # As noted above, the overlap of voronoi regions need to be computed manually.
        # Unfortunately, this increases the runtime to O(n³). On the other hand, it produces consistent results.
        overlay = []

        # Dictionaries contain the polygons for (partial) intersections and unions for each combination
        # This way, not every time the (Multi)Polygon for a specific combination needs to be calculated from the start
        subset_to_intersections = {(i,): united_sites[i] for i in range(len(united_sites))}
        subset_to_intersections.update({(): self.convert_box_to_polygon()})

        # Maximally, only that much regions can overlap as there are diagrams
        max_num_of_overlaps = int(360 / self.degrees)
        combinations = self.powerset([i for i in range(self.num_of_points)], max_num_of_overlaps)

        # Then, calculate for each subset of sites the overlapping polygon
        # By subtracting those sites not being in the subset from the intersection of sites in the subset
        # TODO:
        # This takes additionally n steps for each combination, thus resulting in a final runtime of O(n⁴ log(n)),
        # Compared to an optimal (worst case) runtime of O(n²)
        # An optimal Map-Overlay is explained in
        # de Berg, van Kreveld, Overmars and Sehwarzkopf:
        # Computational Geometry - Algorithms and Applications, Second Revised Edition
        # Chapter 2.3: Computing the Overlay of Two Subdivisions
        # And is left for future work.
        for combination in combinations:

            complement = Polygon()

            intersection = self.convert_box_to_polygon()

            # Compute the overlap for all sites in combination
            for elem in combination:
                intersection = intersection.intersection(united_sites[elem])

            # This is the step which costs additional n calculations per loop:
            # All other sites, which are not in the combination, are first united
            for elem in range(self.num_of_points):
                if elem not in combination:
                    complement = complement.union(united_sites[elem])

            # And afterwards subtracted
            # Buffering is only necessary for the intersection, because the complement has geom_type Polygon
            intersection = intersection.buffer(self.eps)

            polygon = intersection.difference(complement)

            # The subset now contains the indices of the data points whose OrientedVoronoi-regions overlap at the polygon
            overlay.append((polygon, combination))

        return overlay


    def subdivide(self, overlay):
        """
        Computes the refined subdivision by Monma & Suri.
        :param overlay: List of (Shapely-object, tuple(integer))-tuples of shapely objects and
                        indices of their nearest neighbors
        """

        # Possible function call to cluster regions (disabled)
        # overlay = self.cluster_regions(overlay)

        for sub_region, nearest_neighbors in overlay:

            # Important, otherwise some regions cannot be identified
            sub_region = sub_region.buffer(-self.eps).buffer(self.eps)

            # To avoid errors, only objects of type Polygon are used for later computations
            list_of_regions = []

            # Filter out polygons of the sub_region:
            if not sub_region.is_empty:
                if sub_region.geom_type == 'MultiPolygon' or sub_region.geom_type == 'GeometryCollection':
                    for geometry in sub_region.geoms:
                        if not geometry.is_empty and geometry.geom_type == 'Polygon' and geometry.is_valid:
                            list_of_regions.append(geometry)

                if sub_region.geom_type == 'Polygon' and sub_region.is_valid:
                    list_of_regions.append(sub_region)

                # A subset has maximally 5 nodes, as the maximum degree in an euclidean MST is wlog 5.
                powerset = self.powerset(nearest_neighbors, len(nearest_neighbors))

                # Each subset consists a set of neighbors of data points
                for combination in powerset:

                    # Go over all polygons with the same
                    for region in list_of_regions:

                        for element in combination:
                            refined_region = self.compute_region(region, element, combination, nearest_neighbors)

                            # Shrink and expand the refined region to remove artifacts
                            refined_region = refined_region.buffer(-self.eps).buffer(self.eps)

                            # Again, filter possible non-polygonal artifacts, such as LineStrings and Points
                            # And Polygons which remain due to rounding errors
                            if refined_region.geom_type == 'Polygon' and \
                                    not refined_region.is_empty and \
                                    refined_region.is_valid and \
                                    np.round(refined_region.area, 2) >= self.eps:

                                self.total_area += refined_region.area

                                is_optimal, optimal_subset = self.is_optimal_tree(refined_region,
                                                                                  combination,
                                                                                  powerset)

                                if not is_optimal:
                                    # Store both subsets to plot both the correct and erroneously identified tree
                                    self.erroneous_regions.append(
                                        (refined_region, tuple(combination), tuple(optimal_subset)))

                                    self.num_of_erroneous_regions += 1
                                    self.area_of_erroneous_regions += refined_region.area

                                else:
                                    self.correct_regions.append((refined_region, combination))

                                # Store the cell under its topology
                                if tuple(optimal_subset) not in self.topology_to_regions:
                                    self.topology_to_regions[tuple(optimal_subset)] = [refined_region]
                                else:
                                    self.topology_to_regions[tuple(optimal_subset)].append(refined_region)

                                self.num_of_regions += 1

                            if not refined_region.is_empty and \
                                    (refined_region.geom_type == 'MultiPolygon' or
                                     refined_region.geom_type == 'GeometryCollection'):

                                for geometry in refined_region.geoms:
                                    if geometry.geom_type == 'Polygon' and \
                                            not geometry.is_empty and \
                                            geometry.is_valid and \
                                            np.round(geometry.area, 2) >= self.eps:

                                        self.total_area += geometry.area

                                        is_optimal, optimal_subset = self.is_optimal_tree(geometry,
                                                                                          combination,
                                                                                          powerset)

                                        if not is_optimal:
                                            self.erroneous_regions.append(
                                                (geometry, tuple(combination), tuple(optimal_subset)))

                                            self.num_of_erroneous_regions += 1
                                            self.area_of_erroneous_regions += geometry.area
                                        else:
                                            self.correct_regions.append((geometry, combination))

                                        if tuple(optimal_subset) not in self.topology_to_regions:
                                            self.topology_to_regions[tuple(optimal_subset)] = [geometry]
                                        else:
                                            self.topology_to_regions[tuple(optimal_subset)].append(geometry)

                                        self.num_of_regions += 1

        # Possible function call to cluster cells (disabled)
        # self.cluster_cells()

    def compute_bounding_box(self, lower_left_x, lower_left_y, upper_right_x, upper_right_y):
        """
        :param lower_left_x: x-Coordinate of lower left corner
        :param lower_left_y: y-Coordinate of lower left corner
        :param upper_right_x: x-Coordinate of upper left corner
        :param upper_right_y: y-Coordinate of upper left corner
        :return: (bounding_box, box_corners, box_directions)-Triple of lists containing box boundaries, corners,
                 and directions of boundaries based on the lower-left and upper-right corner.
        """
        # Additional margins to prevent boundaries from ending up in corners
        # Idea from https://www.cs.hmc.edu/~mbrubeck/voronoi.html
        x_margin = (upper_right_x - lower_left_x) / 10
        y_margin = (upper_right_y - lower_left_y) / 10

        # Subtract/ Add margins from/ to lower left/ upper right corner
        lower_left_x -= x_margin
        upper_right_x += x_margin
        lower_left_y -= y_margin
        upper_right_y += y_margin

        # The four edges of the box and their projections onto the sweeping direction
        bottom_left_corner = Coordinate(lower_left_x, lower_left_y)
        top_left_corner = Coordinate(lower_left_x, upper_right_y)
        top_right_corner = Coordinate(upper_right_x, upper_right_y)
        bottom_right_corner = Coordinate(upper_right_x, lower_left_y)

        # Since boundaries are constructed based on a starting point and a direction,
        # The left-right-/ right-left-direction and top-bottom-/ bottom-top-direction are calculated beforehand
        bottom_top_dir = Coordinate(top_left_corner.x - bottom_left_corner.x,
                                    top_left_corner.y - bottom_left_corner.y)
        left_right_dir = Coordinate(top_right_corner.x - top_left_corner.x,
                                    top_right_corner.y - top_left_corner.y)
        top_bottom_dir = Coordinate(bottom_right_corner.x - top_right_corner.x,
                                    bottom_right_corner.y - top_right_corner.y)
        right_left_dir = Coordinate(bottom_left_corner.x - bottom_right_corner.x,
                                    bottom_left_corner.y - bottom_right_corner.y)

        # Boundaries in clockwise order, starting bottom_left
        # Left/ right regions cannot be determined, thus are None
        left_boundary = Boundary(bottom_left_corner, bottom_top_dir, None, None)
        left_boundary.is_infinite = False

        top_boundary = Boundary(top_left_corner, left_right_dir, None, None)
        top_boundary.is_infinite = False

        right_boundary = Boundary(top_right_corner, top_bottom_dir, None, None)
        right_boundary.is_infinite = False

        bottom_boundary = Boundary(bottom_right_corner, right_left_dir, None, None)
        bottom_boundary.is_infinite = False

        # Edges and boundaries in clockwise arrangement, arbitrarily starting with the bottom_left_corner/ left_boundary
        box_corners = [bottom_left_corner, top_left_corner, top_right_corner, bottom_right_corner]
        box_directions = [bottom_top_dir, left_right_dir, top_bottom_dir, right_left_dir]
        bounding_box = [left_boundary, top_boundary, right_boundary, bottom_boundary]

        return bounding_box, box_corners, box_directions

    def close_polygon(self, site, edges):
        """
        Clips an open voronoi polygon with the bounding box.
        :param site: The site of type Coordinate the voronoi region corresponds to
        :param edges: The edges of the open region; either type Boundary, or in the shapely tuple format
                      ((start_x, start.y), (end.x, end.y)) (all float)
        :return: [((float, float)(float, float))] list of clipped edges in shapely-format,
                 which form a valid shapely polygon
        """

        # The list of clipped edges
        clipped = []

        # Find the left-most open boundary:
        left_most_boundary = None
        i = -1
        is_boundary = False

        while i < len(edges) - 1 and not is_boundary:
            i += 1
            left_most_boundary = edges[i]
            is_boundary = isinstance(edges[i], Boundary)

            if not is_boundary:
                # If the edge is not of type boundary, i.e. it is closed, append it to the other edges
                clipped.append(left_most_boundary)

        # Move to the upcoming edge
        i += 1

        # Find the index of the box boundary that intersects this left-most edge
        j = -1
        intersects = False
        intersection = None

        while j < len(self.bounding_box) - 1 and not intersects:
            j += 1

            # In rare cases, the left most boundary does not intersect with any box boundary
            # Reasons are unknown
            intersects, intersection = self.bounding_box[j].intersects_boundary(left_most_boundary)

        # Copy the list of box corners to be able to replace the corner at index i with the intersection
        box_corner_copy = self.box_corners.copy()

        # Clip the left-most boundary at its intersection with the bounding box
        # If the the intersection is not None
        if intersection is not None:
            clipped.append(((left_most_boundary.start.x, left_most_boundary.start.y), (intersection.x, intersection.y)))

        if site.__eq__(self.environment):
            # If site is the environment, the bounding box must encapsulate the environment
            # Thus, a line is drawn from the last corner (in clockwise direction)

            if intersection is not None:
                clipped.append(((box_corner_copy[j].x, box_corner_copy[j].y), (intersection.x, intersection.y)))
        else:
            # If site is not the environment, the first corner to begin with is the left-most intersection
            box_corner_copy[j] = intersection

        for k in range(len(self.bounding_box)):

            # Move around the bounding box in clockwise fashion
            cur_index = (k + j) % len(self.bounding_box)

            cur_box_boundary = self.bounding_box[cur_index]

            last_intersection = box_corner_copy[cur_index]

            move_to_next_box_boundary = False

            # While there are unprocessed edges and no indication to move on to the next box boundary
            while i < len(edges) and not move_to_next_box_boundary:

                cur_edge = edges[i]

                if isinstance(cur_edge, Boundary):

                    intersects, intersection = cur_box_boundary.intersects_boundary(cur_edge)

                    if intersects:

                        # Clip the cur_edge at the boundary
                        clipped.append(((cur_edge.start.x, cur_edge.start.y), (intersection.x, intersection.y)))

                        # Check if the boundary has the actual region to its left
                        # Necessary for skipping "nested" cones of other sites
                        if cur_edge.left_site.__eq__(site):

                            # Add the box boundary between the last and current intersection
                            # If the last intersection is not None
                            if last_intersection is not None:
                                clipped.append(((last_intersection.x, last_intersection.y),
                                                (intersection.x, intersection.y)))

                            last_intersection = intersection

                            i += 1
                        else:
                            last_intersection = intersection
                            i += 1
                    else:
                        if cur_edge.left_site.__eq__(site):
                            # If the enclosed site coincides with the current site
                            # Close the gap between the last intersection and the next corner,
                            # Which is the starting point for the next clipped edge
                            next_corner = box_corner_copy[(k + j + 1) % len(self.box_corners)]

                            if last_intersection is not None:
                                clipped.append(((last_intersection.x, last_intersection.y),
                                                (next_corner.x, next_corner.y)))

                            last_intersection = next_corner

                        # Move in any case on to the upcoming box boundary in clockwise direction
                        move_to_next_box_boundary = True
                else:
                    # If the edge is not of type Boundary, i.e. is closed, move on to the next edge
                    clipped.append(cur_edge)

                    i += 1

            if i == len(edges) and site.__eq__(self.environment):
                # Close the gap between the last box_point and the next corner, which is the starting point
                # For the next partial_box_boundary
                next_corner = box_corner_copy[(j + k + 1) % len(self.box_corners)]

                if last_intersection is not None:
                    clipped.append(((last_intersection.x, last_intersection.y), (next_corner.x, next_corner.y)))

                last_intersection = next_corner

        return clipped

    def powerset(self, elements, max_length, min_length=1):
        """
        from https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
        :param min_length: default = 1 (to avoid the empty set)
        :param max_length: The maximal number of elements in a subset
        :param elements: List of arbitrary integers
        :return: List containing all possible subsets of ranging size between min_length and max_length
        """

        # Example:
        # powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

        # Neither should the empty set be part of the powerset, nor are sets with more than 6 elements not necessary
        return list(chain.from_iterable(combinations(elements, length) for length in range(min_length, max_length + 1)))

    def get_voronoi_region(self, vertex, neighbors):
        """
        Computes a standard (non-oriented) euclidean OrientedVoronoi region for a vertex with respect to other vertices.
        :param vertex: The integer index of the data point for which the OrientedVoronoi region is computed
        :param neighbors: The other data point indices, to which the region is adjacent

        :return: Shapely Polygon of the euclidean OrientedVoronoi region for vertex
        """

        # Shapely cannot handle infinite lines
        # Therefore, the intersections with the bounding box are handled with the self-defined
        # Coordinate- and Boundary-class
        site = Coordinate(self.data_points[vertex, 0], self.data_points[vertex, 1])

        # Fixed bounding box (as shapely polygon) which is split along the bisectors between vertex and neighbors
        box = self.convert_box_to_polygon()

        # At the beginning, the OrientedVoronoi-region for the vertex consists of the bounding box
        # If neighbors contains only a single item, i.e. the vertex, the bounding box is returned
        region = self.convert_box_to_polygon()

        # Then, for each (vertex, neighbor) pair, the bounding box is split along their bisector into two polygons
        # Subsequently, all those polygons containing vertex are intersected
        # The result is the OrientedVoronoi-region for the given vertex
        for neighbor in neighbors:

            # Any neighbor which does not coincide with the vertex
            if neighbor != vertex:
                neighbor_site = Coordinate(self.data_points[neighbor, 0], self.data_points[neighbor, 1])

                # Straight connecting the vertex and its neighbor
                connector = Coordinate(neighbor_site.x - site.x, neighbor_site.y - site.y)

                mid = Coordinate((site.x + neighbor_site.x) / 2, (site.y + neighbor_site.y) / 2)

                # Normal vector of the straight connecting the vertex and its neighbor
                # Points in right direction
                normal_1 = Coordinate(connector.y, -connector.x)

                # Points in left direction
                normal_2 = Coordinate(-connector.y, connector.x)

                bisector_1 = Boundary(mid, normal_1, site, neighbor_site)
                bisector_2 = Boundary(mid, normal_2, neighbor_site, site)

                intersection_1 = None
                intersection_2 = None

                # Loop over the box boundaries and check which ones are intersected by the two bisectors
                for boundary in self.bounding_box:
                    intersects, intersection = boundary.intersects_boundary(bisector_1)

                    if intersects:
                        intersection_1 = intersection

                    intersects, intersection = boundary.intersects_boundary(bisector_2)
                    if intersects:
                        intersection_2 = intersection

                splitting_line = LineString([(intersection_1.x, intersection_1.y),
                                             (intersection_2.x, intersection_2.y)])

                # Split the box along the bisector into two Polygons
                parts = split(box, splitting_line)

                # Determine in which of the two polygons the vertex is and intersect the region with the one in question
                if Point(site.x, site.y).within(parts.geoms[0]):
                    region = region.intersection(parts.geoms[0])
                elif Point(site.x, site.y).within(parts.geoms[1]):
                    region = region.intersection(parts.geoms[1])

        return region

    def compute_region(self, polygon, element, subset, nearest_neighbors):
        """
        Follows Procedure Compute-Region of Monma & Suri
        :param polygon: Shapely Polygonal cell in the subdivision of the plane
        :param element: Integer index of a vertex "v" of the subset "V" of the set of nearest neighbors
        :param subset: Tuple representing a subset of nearest neighbors
        :param nearest_neighbors: Tuple containing all nearest neighbors of the polygon
        :return: A Shapely object of the refined region which denotes the location of all points x
                 being attached to vertex v of the MST
        """

        # In this line, the code differs from Monma & Suri.
        # They suggest to take the OrientedVoronoi region of element with respect to the other members in the subset:
        # voronoi_region = self.get_voronoi_region(element, subset)
        # However, this does not lead to a proper subdivision of the plane:
        # In case, the subset contains only one vertex (i.e., element), the resulting OrientedVoronoi region would span
        # The whole diagram and would overlap with other subdivisions.
        # For instance, assume the neighbor-set {1, 2, 3} for an arbitrary cell. Then, the eligible singleton-subsets
        # {1}, {2}, and {3} would all cover the whole cell, because their OrientedVoronoi regions with respect to themselves
        # Would span the infinitely.
        # Also, if the cell intersects with a OrientedVoronoi region of a neighbor, this means that this neighbor is closer
        # To the points in the intersection than any other vertex, such that it has to be in the set of MST nodes the
        # Any new point there is connected to.
        voronoi_region = self.get_voronoi_region(element, nearest_neighbors)

        refined_cell = polygon.intersection(voronoi_region)

        # Intersect circles with the radii being of the longest common edges between subset members
        for v in subset:
            if v != element:
                center = (self.data_points[v, 0], self.data_points[v, 1])

                radius = self.mst.longest_common_edges[v][element][1]

                circle = Point(center).buffer(radius)

                refined_cell = refined_cell.intersection(circle)

        # Intersect circle complements with the radii being of the longest common edges
        # Between subset and non-subset members
        for neighbor in nearest_neighbors:

            if neighbor not in subset:
                center = (self.data_points[neighbor, 0], self.data_points[neighbor, 1])

                radius = self.mst.longest_common_edges[neighbor][element][1]

                circle = Point(center).buffer(radius)

                refined_cell = refined_cell.difference(circle)

        return refined_cell

    def convert_box_to_polygon(self):
        """
        Helper Method that converts the bounding box to a shapely polygon.
        :return: Shapely Polygon of the bounding box
        """
        return Polygon([(self.box_corners[0].x, self.box_corners[0].y),
                        (self.box_corners[1].x, self.box_corners[1].y),
                        (self.box_corners[2].x, self.box_corners[2].y),
                        (self.box_corners[3].x, self.box_corners[3].y)])

    def is_optimal_tree(self, cell, subset, powerset):
        """
        Checks in O(1) if a topology in a region results in an optimal tree.
        :param cell: Shapely Polygon representing the region

        :param subset: Tuple of integers, corresponding to the topology:
                       Each integer represents the index of a data point,
                       with which every point in the cell is connected

        :param powerset: List of tuples of integers corresponding to all possible topologies

        :return: Tuple (is_optimal, optimal_subset) containing a boolean if the given topology is optimal
                 and the optimal subset, that contains the indices of the actual data points with which an
                 arbitrary point in the cell is connected
        """
        # Any point x in the cell is sufficient as representative for the whole cell, because
        # "[t]he geographic neighbor set is invariant over all x in the cell",
        # See Monma & Suri, chapter 4, "Classification of Topologies"
        x = cell.centroid

        optimal_subset = subset
        optimal_weight = self.mst.weight
        is_optimal = True

        # The same edge(s) can be multiply deleted from the original MST
        # But its weight should only be subtracted once
        deleted_edges = set()

        # Compute the weight of the determined MST for the cell
        for elem in subset:
            # Add the distance from each node of the subset to the centroid of the cell x
            optimal_weight += eu(self.data_points[elem, :], [x.x, x.y])

            # Find the longest common edges between the nodes in the subset which are deleted
            for other_elem in subset:
                if elem != other_elem:
                    deleted_edges.add(self.mst.longest_common_edges[elem][other_elem])

        # Subtract the weights of the deleted edges
        for (deleted_edge, weight) in deleted_edges:
            optimal_weight -= weight

        # Loop over all other possible combinations of vertices, to which x could be connected
        for combination in powerset:
            if combination != subset:

                tmp_weight = self.mst.weight
                deleted_edges = set()

                # Proceed as above
                for elem in combination:
                    tmp_weight += eu(self.data_points[elem, :], [x.x, x.y])

                    for other_elem in combination:
                        if elem != other_elem:
                            deleted_edges.add(self.mst.longest_common_edges[elem][other_elem])

                for (deleted_edge, weight) in deleted_edges:
                    tmp_weight -= weight

                # If the weight of the tree for the combination is lower than the one produced by Monma & Suri:
                # Store the combination and its associated weight
                if tmp_weight < optimal_weight:
                    is_optimal = False
                    optimal_weight = tmp_weight
                    optimal_subset = combination

        return is_optimal, optimal_subset

    def cluster_regions(self, subdivision):
        """
        Clusters regions with the same set of nearest neighbors.
        Not used, as it does not always work properly; close, but non-adjacent regions are sometimes clustered,
        while adjacent regions sometimes are not.
        This might be due to two reasons:
        1) Finding an appropriate buffer-size to check if adjacent regions overlap/ touch/ intersect is difficult.
        If chosen too large, close, but non-adjacent regions are combined, whereas if too small, adjacent regions are
        not clustered.
        2) Shapely's own overlap/ touch/ intersect functions do not always return correct values, especially when dealing
        with more complex-shaped polygons.

        :param subdivision: Original list of (Shapely-object, tuple(int)) with shapely-objects and the indices of
                            nearest neighbors

        :return: List of (Shapely-object, tuple(int)) tuples containing the clustered shapely objects which share the
                 same nearest neighbors and are adjacent
        """
        clustered_division = []

        for region, nearest_neighbors in subdivision:

            if region.geom_type == 'Polygon' and not region.is_empty and region.is_valid:
                clustered_division.append((region, nearest_neighbors))

            elif not region.is_empty and \
                (region.geom_type == 'MultiPolygon' or
                region.geom_type == 'GeometryCollection'):

                subregions = []

                for subregion in region.geoms:
                    if subregion.geom_type == 'Polygon' and not subregion.is_empty and subregion.is_valid:
                        subregions.append(subregion)

                is_clustered = set()

                # Loop over all (so far un-clustered) regions
                for i in range(len(subregions)):
                    if i not in is_clustered:

                        is_clustered.add(i)
                        polygon = subregions[i]

                        # Only look for possible regions to union with in the remaining list
                        # Regions *before* the current region would have already been subsumed with the current one
                        for j in range(i + 1, len(subregions)):

                            if j is not is_clustered:
                                # Slightly scale the possible neighbor
                                possibly_adjacent_region = subregions[j].buffer(self.eps)

                                # If regions i and j do now overlap, touch or intersect, they are combined
                                if possibly_adjacent_region.overlaps(polygon) or \
                                        possibly_adjacent_region.touches(polygon) or \
                                        possibly_adjacent_region.intersects(polygon):
                                    polygon = polygon.union(possibly_adjacent_region)

                                    # Region j is thereby clustered
                                    is_clustered.add(j)

                        polygon = polygon.buffer(self.eps, 1, join_style=JOIN_STYLE.mitre).buffer(-self.eps, 1,
                                                 join_style=JOIN_STYLE.mitre)

                        # Re-scale the union of i and j and add it to the list of clustered regions
                        clustered_division.append((polygon.buffer(- self.eps), nearest_neighbors))

        return clustered_division

    def cluster_cells(self):
        """
        Clusters cells with the same topology, which are then stored in topology_to_regions.
        Not used, as it does not always work properly; close, but non-adjacent regions are sometimes clustered,
        while adjacent regions sometimes are not.
        This might be due to two reasons:
        1) Finding an appropriate buffer-size to check if adjacent regions overlap/ touch/ intersect is difficult.
        If chosen too large, close, but non-adjacent regions are combined, whereas if too small, adjacent regions are
        not clustered.
        2) Shapely's own overlap/ touch/ intersect functions do not always return correct values, especially when dealing
        with more complex-shaped polygons.
        """

        # Reset the number of regions
        self.num_of_regions = 0

        # For each topology, union (so far unclustered) regions, if they overlap/ touch/ intersect
        # after a small increase in size through buffering
        for topology in self.topology_to_regions:

            is_clustered = set()

            clustered_cells = []

            for i in range(len(self.topology_to_regions[topology])):
                if i not in is_clustered:

                    is_clustered.add(i)

                    polygon = self.topology_to_regions[topology][i].buffer(self.eps)

                    # Only look for possible regions to union with in the remaining list
                    # Regions *before* the current region would have already been subsumed with the current one
                    for j in range(i + 1, len(self.topology_to_regions[topology])):

                        # Region j is only combined with region i if both have not been clustered yet
                        if j not in is_clustered:

                            # Slightly scale the possible neighbor
                            possibly_adjacent_cell = self.topology_to_regions[topology][j].buffer(self.eps)

                            # If regions i and j do now overlap, touch or intersect, they are combined
                            if possibly_adjacent_cell.overlaps(polygon) or \
                                    possibly_adjacent_cell.touches(polygon) or \
                                    possibly_adjacent_cell.intersects(polygon):
                                polygon = polygon.union(possibly_adjacent_cell)

                                # Region j is thereby clustered
                                is_clustered.add(j)

                    polygon = polygon.buffer(self.eps, 1,
                                             join_style=JOIN_STYLE.mitre).buffer(-self.eps, 1,
                                                                                 join_style=JOIN_STYLE.mitre)

                    # Re-scale the union of i and j and add it to the list of clustered cells
                    clustered_cells.append(polygon.buffer(- self.eps))

            self.num_of_regions += len(clustered_cells)
            self.topology_to_regions[topology] = clustered_cells

    def get_cmap(self, n, name='hsv'):
        """
        Returns a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color;
        the keyword argument name must be a standard mpl colormap name.
        Based on:
        https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib

        :param n: Integer indicating the number of distinct colors
        :param name: matplotlib color map
        :return: matplotlib color map containing n distinct colors
        """

        return plt.cm.get_cmap(name, n)
