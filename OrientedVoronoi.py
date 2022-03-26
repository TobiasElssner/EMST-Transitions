#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Computes an oriented OrientedVoronoi diagram following the algorithm by Chang & Tang (1990)
@author Tobias Elßner
"""

import numpy as np
from scipy.spatial.distance import euclidean as eu
from blist import blist
from typing import Union
from Coordinate import Coordinate
from Boundary import Boundary
from Node import QueueNode
from PriorityQueue import PriorityQueue


class OrientedVoronoi:

    def __init__(self, sites, basis, env):
        """
        Initialize a OrientedVoronoi diagram and its attributes.
        :param sites: 2-d numpy array containing the data points ('sites' in OrientedVoronoi terms)
        :param basis: 2x2 numpy array indicating the two basis vectors forming the cone.
        :param env: Coordinate specifying the environment in which the OrientedVoronoi regions for the sites are constructed.
        """

        # Accuracy
        self.eps = 0.01

        self.data_points = sites

        # Stores the sites and their boundaries from left to right
        # blist ensures logarithmic insertion/ deletion from the list, s.t. O(n log n) can theoretically achieved
        self.active = blist()  # type: Union[blist[Boundary], blist[Coordinate]]
        self.boundaries = set()

        self.regions_to_boundaries = {}

        self.open_regions = set()

        num_of_sites = len(sites)

        self.site_pq = PriorityQueue(num_of_sites, comp=self.compare, dist=self.euclidean_distance)

        self.environment = env

        # Maximum number of intersections is twice the squared number of sites
        # Two sites can intersect at most once
        # Therefore, the upper bound for intersections is the number of different combinations of two sites:
        # (n**2 - n) / 2
        # In fact, the number of OrientedVoronoi points (of which the intersections are a subset) is in O(n)
        # But to avoid errors, the supremum is chosen
        self.intersection_pq = PriorityQueue(int((num_of_sites ** 2 - num_of_sites) / 2),
                                             comp=self.compare,
                                             dist=self.euclidean_distance)


        self.basis_1 = Coordinate(basis[0, 0], basis[1, 0])
        self.basis_2 = Coordinate(basis[0, 1], basis[1, 1])
        angle = np.dot(basis[:, 0], basis[:, 1]) / (np.linalg.norm(basis[:, 0]) * np.linalg.norm(basis[:, 1]))

        self.regions_to_boundaries[self.environment] = []

        # It is assumed that basis_1 lies to the right, and basis_2 to the left of the sweeping direction
        # The sites are projected onto the sweeping direction
        # The order in which those projections are handled is from bottom to top, right to left
        sweeping_direction_x = (self.basis_1.x + self.basis_2.x) / 2
        sweeping_direction_y = (self.basis_1.y + self.basis_2.y) / 2

        self.sweeping_direction = Coordinate(sweeping_direction_x, sweeping_direction_y)

        # Length-normalization to facilitate the projection of points later on
        length = np.sqrt(self.sweeping_direction.x**2 + self.sweeping_direction.y**2)

        self.sweeping_direction.x /= length
        self.sweeping_direction.y /= length

        self.lower_left_x = 0
        self.lower_left_y = 0
        self.upper_right_x = 0
        self.upper_right_y = 0

        for coordinates in sites:

            x = coordinates[0]
            y = coordinates[1]

            if self.lower_left_x > x:
                self.lower_left_x = x
            if self.lower_left_y > y:
                self.lower_left_y = y

            if self.upper_right_x < x:
                self.upper_right_x = x
            if self.upper_right_y < y:
                self.upper_right_y = y

            site = Coordinate(x, y)

            # Orthogonal Projection onto the sweeping direction
            # Assuming the sweeping direction being a unit vector through the origin
            site_key = self.project_onto_sweeping_direction(site)

            site_node = QueueNode(site_key, site)

            self.site_pq.insert(site_node)

            self.regions_to_boundaries[site] = []

        self.active.append(self.environment)

        # Scan through the queues in order of the projections onto the sweeping direction
        while not self.site_pq.is_empty() or not self.intersection_pq.is_empty():

            if not self.site_pq.is_empty() and not self.intersection_pq.is_empty():
                site_node = self.site_pq.peek()
                intersection_node = self.intersection_pq.peek()

                # If a site coincides with an intersection, the intersection is ignored
                if site_node.key.__eq__(intersection_node.key):
                    self.intersection_pq.pop()
                    self.handle_site()

                # If the next site comes before an upcoming intersection
                if self.compare(site_node.key, intersection_node.key):
                    self.handle_intersection()

                # If an intersection comes before the next site
                else:
                    self.handle_site()
            # Otherwise, handle the remaining sites/ intersections
            elif not self.site_pq.is_empty() and self.intersection_pq.is_empty():
                self.handle_site()

            elif self.site_pq.is_empty() and not self.intersection_pq.is_empty():
                self.handle_intersection()

        # Loop over all active elements to find any open boundaries
        # And detect the size of the diagram by the left-bottom-most and right-top-most point
        for elem in self.active:
            if isinstance(elem, Boundary):
                self.boundaries.add(elem)

                self.open_regions.add(elem.right_site)
                self.open_regions.add(elem.left_site)

                # Type the remaining open edges as Boundary
                # This way, they can be identified and intersected with the final bounding box
                # Note: The open edges are ordered from left to right, which is important for the computation of the
                # Bounding box in the next step
                self.regions_to_boundaries[elem.right_site].append(elem)
                self.regions_to_boundaries[elem.left_site].append(elem)

                if self.upper_right_x < elem.start.x:
                    self.upper_right_x = elem.start.x
                if self.upper_right_x < elem.end.x:
                    self.upper_right_x = elem.end.x
                if self.upper_right_y < elem.start.y:
                    self.upper_right_y = elem.start.y
                if self.upper_right_y < elem.end.y:
                    self.upper_right_y = elem.end.y

                if self.lower_left_x > elem.start.x:
                    self.lower_left_x = elem.start.x
                if self.lower_left_x > elem.end.x:
                    self.lower_left_x = elem.end.x
                if self.lower_left_y > elem.start.y:
                    self.lower_left_y = elem.start.y
                if self.lower_left_y > elem.end.y:
                    self.lower_left_y = elem.end.y


    def handle_site(self):
        """
        Handles an upcoming site, and inserts/ deletes future intersections along its boundaries.
        """

        site_node = self.site_pq.pop()

        # Refactoring to improve code readability
        site_coordinates = site_node.value

        index = self.place_region(site_node.value)

        active_site = self.active[index]

        # Add the basis vectors as boundary segments
        # By default, it is assumed that the new site lies completely in another
        # Exceptions (i.e., new site lies on one or two boundaries) are handled later on
        left_cone_boundary = Boundary(site_coordinates, self.basis_1, active_site, site_node.value)
        right_cone_boundary = Boundary(site_coordinates, self.basis_2, site_node.value, active_site)

        left_boundary = None
        right_boundary = None

        coincides_with_intersection = False

        # Delete any succeeding intersections between the left and right boundary
        # And check whether the new site coincides with an intersection
        if index - 1 > 0 and index + 1 < len(self.active):
            left_boundary = self.active[index - 1]
            right_boundary = self.active[index + 1]

            intersects, intersection = left_boundary.intersects_boundary(right_boundary)

            if intersects:
                self.delete_intersection(intersection)

                # Make sure that the intersection is sufficiently away from the new site
                distance_to_intersection = intersection.euclidean_distance(site_coordinates)

                if distance_to_intersection < self.eps:
                    coincides_with_intersection = True

        if not coincides_with_intersection:

            # By default, it is assumed that the new site lies within one region
            lies_within_region = True

            # And that no intersections take place
            intersects_left, left_intersection, intersects_right, right_intersection = False, None, False, None

            if index - 1 > 0:
                left_boundary = self.active[index - 1]

                # Check if the new site lies directly on the left boundary or is very close to the left intersection
                # Then, the site in which it lies does not have to be duplicated
                intersects_left, left_intersection = left_cone_boundary.intersects_boundary(left_boundary)
                distance_to_intersection = self.eps

                if intersects_left:
                    distance_to_intersection = left_intersection.euclidean_distance(site_coordinates)

                # If the site lies directly on the left boundary
                if site_coordinates.lies_on_boundary(left_boundary) or distance_to_intersection < self.eps:

                    # The possible left intersection takes place with the boundary at index - 3
                    if index - 3 > 0:

                        intersects_left, left_intersection = \
                            self.active[index - 3].intersects_boundary(left_cone_boundary)

                        # Delete any intersections with the 'old' left boundary
                        intersected_previously, previous_intersection = \
                            self.active[index - 3].intersects_boundary(left_boundary)

                        if intersected_previously:
                            self.delete_intersection(previous_intersection)

                    # Likewise, a possible intersection to the right takes place at index + 1
                    if index + 1 < len(self.active):

                        intersects_right, right_intersection = \
                            self.active[index + 1].intersects_boundary(right_cone_boundary)

                        intersected_previously, previous_intersection = \
                            self.active[index + 1].intersects_boundary(right_boundary)

                        if intersected_previously:
                            self.delete_intersection(previous_intersection)

                    # Finish the left boundary at the new site
                    left_boundary.is_infinite = False
                    left_boundary.end = site_coordinates
                    self.boundaries.add(left_boundary)

                    # Store the edges of the "closed" edges in the shapely format (used in later computations)
                    # For the left and right hand side region
                    # As a list of tuples:
                    # [((start.x, start.y), (end.x, end.y)), ...]
                    self.regions_to_boundaries[left_boundary.right_site]. \
                        append(((left_boundary.start.x, left_boundary.start.y), (left_boundary.end.x, left_boundary.end.y)))

                    self.regions_to_boundaries[left_boundary.left_site]. \
                        append(((left_boundary.start.x, left_boundary.start.y), (left_boundary.end.x, left_boundary.end.y)))

                    left_cone_boundary.left_site = left_boundary.left_site

                    # Lies left to site
                    self.active.pop(index - 1)
                    self.active.insert(index - 1, right_cone_boundary)
                    self.active.insert(index - 1, site_node.value)
                    self.active.insert(index - 1, left_cone_boundary)

                    lies_within_region = False
                else:
                    intersects_left, left_intersection = left_boundary.intersects_boundary(left_cone_boundary)

            if index + 1 < len(self.active):
                right_boundary = self.active[index + 1]

                # Avoid errors by checking if the new site lies directly on the right boundary
                # Additional check with "lies_within_region", because otherwise it would intersect with its own cone,
                # Which was inserted after its intersection with the left_boundary was detected
                intersects_right, right_intersection = right_cone_boundary.intersects_boundary(right_boundary)
                distance_to_intersection = self.eps

                if intersects_right:
                    distance_to_intersection = right_intersection.euclidean_distance(site_coordinates)

                # If the site lies directly on the right boundary
                if (site_coordinates.lies_on_boundary(right_boundary)) and \
                        lies_within_region or \
                        distance_to_intersection < self.eps:

                    # Check for future intersections and delete intersections with the 'old' right boundary
                    if index - 1 > 0:
                        intersects_left, left_intersection = \
                            self.active[index - 1].intersects_boundary(left_cone_boundary)

                        intersected_previously, previous_intersection = \
                            self.active[index - 1].intersects_boundary(right_boundary)

                        if intersected_previously:
                            self.delete_intersection(previous_intersection)

                    if index + 3 < len(self.active):
                        intersects_right, right_intersection = \
                            self.active[index + 3].intersects_boundary(right_cone_boundary)

                        intersected_previously, previous_intersection = \
                            self.active[index + 3].intersects_boundary(right_boundary)

                        if intersected_previously:
                            self.delete_intersection(previous_intersection)

                    # Finish the left boundary at the new site
                    right_boundary.is_infinite = False
                    right_boundary.end = site_coordinates
                    self.boundaries.add(right_boundary)

                    self.regions_to_boundaries[right_boundary.right_site]. \
                        append(((right_boundary.start.x, right_boundary.start.y),
                                (right_boundary.end.x, right_boundary.end.y)))

                    self.regions_to_boundaries[right_boundary.left_site]. \
                        append(((right_boundary.start.x, right_boundary.start.y),
                                (right_boundary.end.x, right_boundary.end.y)))

                    right_cone_boundary.right_site = right_boundary.right_site

                    # Insert boundaries and the site into the list of active elements
                    self.active.pop(index + 1)
                    self.active.insert(index + 1, right_cone_boundary)
                    self.active.insert(index + 1, site_node.value)
                    self.active.insert(index + 1, left_cone_boundary)

                    lies_within_region = False
                else:
                    intersects_right, right_intersection = right_boundary.intersects_boundary(right_cone_boundary)

            # Insert detected left and right intersections
            if intersects_left:
                self.insert_intersection(left_intersection)

            if intersects_right:
                self.insert_intersection(right_intersection)

            # If a site does not collide with a boundary
            if lies_within_region:
                # Insert the new site and its boundaries
                self.active.insert(index + 1, active_site)
                self.active.insert(index + 1, right_cone_boundary)
                self.active.insert(index + 1, site_node.value)
                self.active.insert(index + 1, left_cone_boundary)

    def handle_intersection(self):
        """
        Handles an upcoming intersection, and inserts/ deletes future boundaries, bisectors, and possible intersections.
        """

        intersection_node = self.intersection_pq.pop()

        # Refactoring for better code readability
        intersection_coordinates = intersection_node.value

        left_boundary, left_index, right_boundary, right_index = self.place_intersection(intersection_coordinates)

        # If the intersection belongs to a bisector between two sites, the right boundary has None as index
        # (See place_intersection)
        if right_index is None:

            # Delete any remaining intersections between the first part and its adjacent left and right boundaries
            if left_index - 2 > 0 and isinstance(self.active[left_index - 2], Boundary):

                intersects, intersection = self.active[left_index - 2].intersects_boundary(left_boundary)
                if intersects:

                    self.delete_intersection(intersection)

            if left_index + 2 < len(self.active) and isinstance(self.active[left_index + 2], Boundary):

                intersects, intersection = self.active[left_index + 2].intersects_boundary(left_boundary)
                if intersects:

                    self.delete_intersection(intersection)

            # Mark the end of the first part of the bisector between the left and right site, set it to finite length,
            # And add it to the left and right site's boundaries
            left_boundary.is_infinite = False
            left_boundary.end = intersection_coordinates

            self.boundaries.add(left_boundary)

            self.regions_to_boundaries[left_boundary.left_site]. \
                append(((left_boundary.start.x, left_boundary.start.y), (left_boundary.end.x, left_boundary.end.y)))

            self.regions_to_boundaries[left_boundary.right_site]. \
                append(((left_boundary.start.x, left_boundary.start.y), (left_boundary.end.x, left_boundary.end.y)))

            # Now overwrite the current boundary with the second part of the bisector
            # All other sites and boundaries remain unaffected
            right_boundary_dir = Coordinate(right_boundary.end.x - right_boundary.start.x,
                                            right_boundary.end.y - right_boundary.start.y)

            common_boundary = Boundary(intersection_coordinates, right_boundary_dir,
                                       left_boundary.left_site, left_boundary.right_site)
            self.active[left_index] = common_boundary

            # Insert any intersections with its adjacent left and right boundaries
            if left_index - 2 > 0 and isinstance(self.active[left_index - 2], Boundary):

                intersects, intersection = self.active[left_index - 2].intersects_boundary(common_boundary)

                if intersects and intersection_coordinates.euclidean_distance(intersection) > self.eps:
                    self.insert_intersection(intersection)

            if left_index + 2 < len(self.active) and isinstance(self.active[left_index + 2], Boundary):
                intersects, intersection = self.active[left_index + 2].intersects_boundary(common_boundary)
                if intersects and intersection_coordinates.euclidean_distance(intersection) > self.eps:

                    self.insert_intersection(intersection)
        else:
            # The site in which the intersection takes place lies between the left and right boundary
            cur_site_index = int((left_index + right_index) / 2)

            # The left / right site are left_index -1 / right_index +1
            left_site = self.active[left_index - 1]  # type: Coordinate
            left_site_projection = self.project_onto_sweeping_direction(left_site)

            right_site = self.active[right_index + 1]  # type: Coordinate
            right_site_projection = self.project_onto_sweeping_direction(right_site)

            # The straight connecting the left and right site
            # Needed to calculate the bisector between them
            left_right_straight = Coordinate(right_site.x - left_site.x,
                                             right_site.y - left_site.y)

            # The bisector sits perpendicularly on the straight between left_site and right_site
            # I.e. it is the normal vector to that line
            # It should point to the left hand side in walking direction from left_site towards right_site
            normal_vector = Coordinate(-left_right_straight.y, left_right_straight.x)

            # The bisector which points to infinity
            common_boundary = None

            # Special case: left_site and right_site are at the same height
            # Their shared bisector thus consists of one piece
            # Corresponds to case (c) in Huang & Chang pseudo-code
            if -self.eps < (left_site_projection.y - right_site_projection.y) < self.eps:

                # The bisector starts at the intersection coordinates
                bisector = Boundary(intersection_coordinates, normal_vector, left_site, right_site)

                # Not yet added to the set boundaries, since no endpoint exists and thus still active
                common_boundary = bisector

                # Insert any intersections of the common_boundary with its neighbors
                if left_index - 2 > 0 and isinstance(self.active[left_index - 2], Boundary):

                    # Check for an intersection with the active_common_bisector
                    intersects, intersection = \
                        self.active[left_index - 2].intersects_boundary(common_boundary)

                    if intersects and intersection_coordinates.euclidean_distance(intersection) > self.eps:
                        self.insert_intersection(intersection)

                if right_index + 2 < len(self.active) and isinstance(self.active[right_index + 2], Boundary):

                    # Check for an intersection with the active_common_bisector
                    intersects, intersection = \
                        self.active[right_index + 2].intersects_boundary(common_boundary)

                    if intersects and intersection_coordinates.euclidean_distance(intersection) > self.eps:
                        self.insert_intersection(intersection)

            else:
                # Otherwise, construct an ordinary boundary between the two sites
                # And check for an additional intersection with their bisector
                # By default, it is assumed that the sweepline meets the left_site first
                # Both boundaries end at this intersection

                # Starting from the intersection, the first piece of the common boundary consists of
                # Either the left or right boundary
                right_dir = Coordinate(left_boundary.end.x - left_boundary.start.x,
                                       left_boundary.end.y - left_boundary.start.y)

                left_dir = Coordinate(right_boundary.end.x - right_boundary.start.x,
                                      right_boundary.end.y - right_boundary.start.y)

                # Assuming that the left site is met first by the sweepline, the left_dir continues from the intersection
                common_boundary = Boundary(intersection_coordinates, left_dir, left_site, right_site)

                # If the right_site is met first by the sweepline
                if left_site_projection.y > right_site_projection.y:

                    # If right_site comes first, the first piece of the common boundary has to be the right_dir
                    common_boundary = Boundary(intersection_coordinates, right_dir, left_site, right_site)

                if not left_site.__eq__(self.environment) and not right_site.__eq__(self.environment):
                    bisector_start_x = (left_site.x + right_site.x) / 2
                    bisector_start_y = (left_site.y + right_site.y) / 2
                    bisector_start = Coordinate(bisector_start_x, bisector_start_y)

                    # The general bisector between the left and right site,
                    # Sitting perpendicular on the straight connecting both sites
                    left_right_bisector = Boundary(bisector_start, normal_vector, left_site, right_site)

                    # Determine whether the left_right_bisector intersects at all with the left_right_bisector
                    intersects_common_boundary, intersection_of_common_boundary = \
                        left_right_bisector.intersects_boundary(common_boundary)

                    # The common bisector consists of two pieces, if the first shared bisector is intersected by the
                    # left_right_bisector between the left and right site
                    # Corresponds to case (b) in the pseudo-code of Chang & Huang
                    if intersects_common_boundary and intersection_coordinates.euclidean_distance(intersection_of_common_boundary) > self.eps:
                        self.insert_intersection(intersection_of_common_boundary)

                # Insert any intersections of the common_boundary with its neighbors
                if left_index - 2 > 0 and isinstance(self.active[left_index - 2], Boundary):

                    # Check for an intersection with the common_boundary
                    intersects, intersection = \
                        self.active[left_index - 2].intersects_boundary(common_boundary)

                    if intersects and intersection_coordinates.euclidean_distance(intersection) > self.eps:
                        self.insert_intersection(intersection)

                if right_index + 2 < len(self.active) and isinstance(self.active[right_index + 2], Boundary):

                    # Check for an intersection with the active_common_bisector
                    intersects, intersection = \
                        self.active[right_index + 2].intersects_boundary(common_boundary)

                    if intersects and intersection_coordinates.euclidean_distance(intersection) > self.eps:
                        self.insert_intersection(intersection)

            # Delete any intersection between left_boundary and a boundary to its left
            # And any intersection between right_boundary and a boundary to its right
            if left_index - 2 > 0 and isinstance(self.active[left_index - 2], Boundary):

                intersects, intersection = self.active[left_index - 2].intersects_boundary(left_boundary)

                if intersects:
                    self.delete_intersection(intersection)

            if right_index + 2 < len(self.active) and isinstance(self.active[right_index + 2], Boundary):

                intersects, intersection = self.active[right_index + 2].intersects_boundary(right_boundary)

                if intersects:
                    self.delete_intersection(intersection)

            # Delete any intersection between the left boundary and the bisector between its left and right site
            # And any intersection between the right boundary and the bisector between its left and right site
            if not left_boundary.left_site.__eq__(self.environment) and \
                    not left_boundary.right_site.__eq__(self.environment):

                left_right_straight = Coordinate(left_boundary.right_site.x - left_boundary.left_site.x,
                                                 left_boundary.right_site.y - left_boundary.left_site.y)

                # The bisector sits perpendicularly on the straight between left_site and right_site
                # I.e. it is the normal vector to that line
                # It should point to the left hand side in walking direction from left_site towards right_site
                normal_vector = Coordinate(-left_right_straight.y, left_right_straight.x)

                bisector_start_x = (left_boundary.left_site.x + left_boundary.right_site.x) / 2
                bisector_start_y = (left_boundary.left_site.y + left_boundary.right_site.y) / 2
                bisector_start = Coordinate(bisector_start_x, bisector_start_y)

                # The general bisector between the left and right site,
                # Sitting perpendicular on the straight connecting both sites
                left_right_bisector = Boundary(bisector_start, normal_vector,
                                               left_boundary.left_site, left_boundary.right_site)

                intersects_left_boundary, intersection_of_left_boundary = \
                    left_right_bisector.intersects_boundary(left_boundary)

                if intersects_left_boundary:
                    self.delete_intersection(intersection_of_left_boundary)

            if not right_boundary.left_site.__eq__(self.environment) and \
                    not right_boundary.right_site.__eq__(self.environment):

                left_right_straight = Coordinate(right_boundary.right_site.x - right_boundary.left_site.x,
                                                 right_boundary.right_site.y - right_boundary.left_site.y)

                # The bisector sits perpendicularly on the straight between left_site and right_site
                # I.e. it is the normal vector to that line
                # It should point to the left hand side in walking direction from left_site towards right_site
                normal_vector = Coordinate(-left_right_straight.y, left_right_straight.x)

                bisector_start_x = (right_boundary.left_site.x + right_boundary.right_site.x) / 2
                bisector_start_y = (right_boundary.left_site.y + right_boundary.right_site.y) / 2
                bisector_start = Coordinate(bisector_start_x, bisector_start_y)

                # The general bisector between the left and right site,
                # Sitting perpendicular on the straight connecting both sites
                left_right_bisector = Boundary(bisector_start, normal_vector,
                                               right_boundary.left_site, right_boundary.right_site)

                intersects_right_boundary, intersection_of_right_boundary = \
                    left_right_bisector.intersects_boundary(right_boundary)

                if intersects_right_boundary:
                    self.delete_intersection(intersection_of_right_boundary)

            # Replace left_boundary, cur_site, and right_boundary by active_bisector
            self.active.pop(left_index)
            self.active.pop(left_index)
            self.active.pop(left_index)
            self.active.insert(left_index, common_boundary)

            # Set the intersection as endpoint to the left_boundary and right_boundary
            left_boundary.end = intersection_coordinates
            left_boundary.is_infinite = False

            # Add boundary
            self.boundaries.add(left_boundary)

            self.regions_to_boundaries[left_boundary.left_site]. \
                append(((left_boundary.start.x, left_boundary.start.y), (left_boundary.end.x, left_boundary.end.y)))

            self.regions_to_boundaries[left_boundary.right_site]. \
                append(((left_boundary.start.x, left_boundary.start.y), (left_boundary.end.x, left_boundary.end.y)))

            right_boundary.end = intersection_coordinates
            right_boundary.is_infinite = False

            # Add boundary
            self.boundaries.add(right_boundary)

            self.regions_to_boundaries[right_boundary.left_site]. \
                append(((right_boundary.start.x, right_boundary.start.y), (right_boundary.end.x, right_boundary.end.y)))

            self.regions_to_boundaries[right_boundary.right_site]. \
                append(((right_boundary.start.x, right_boundary.start.y), (right_boundary.end.x, right_boundary.end.y)))

    def place_region_fast(self, lo, hi, site):
        """
        Places a site in an active region in O(log n) by performing binary search on the list of active elements
        :param lo: lower limit
        :param hi: upper limit
        :param site: Coordinate of the site
        :return: Integer Index of the region where site is placed.
        Note: In rare cases, when points lie close to each other, this method cannot find the region in question.
        Then, None is returned.
        """


        mid = int((lo + hi) / 2)

        # If the mid element is a site, check the boundaries to its left and right (if they exist)
        if isinstance(self.active[mid], Coordinate):

            if (mid == 0) or (mid == len(self.active) - 1):
                return mid

            left_boundary = self.active[mid - 1]
            lies_on_left_boundary = site.lies_on_boundary(left_boundary)
            lies_right_to_left_boundary = site.lies_right(left_boundary)

            # If the site lies on a boundary, return always the site to the left of the boundary in question
            # In case of the left boundary (at index mid - 1), it is the site at position mid - 2
            if lies_on_left_boundary:
                return mid - 2

            right_boundary = self.active[mid + 1]
            lies_on_right_boundary = site.lies_on_boundary(right_boundary)
            lies_right_to_right_boundary = site.lies_right(right_boundary)

            # In case the site lies on the right boundary (at index mid), the site is the one to which mid points
            if lies_on_right_boundary:
                return mid

            # Region lies to the right of the right boundary, i.e. somewhere between mid + 1 and hi
            if lies_right_to_left_boundary and lies_right_to_right_boundary:
                return self.place_region_fast(mid, hi, site)

            # Region lies in between the left and right boundary, i.e. is the current one
            elif lies_right_to_left_boundary and not lies_right_to_right_boundary:
                return mid

            # Impossible: a Region cannot lie to the left of the left boundary and to the right of the right boundary
            # elif not lies_right_to_left_boundary and lies_right_to_right_boundary:

            # Region lies to the right of the right boundary, i.e. somewhere between lo and mid - 1
            elif not lies_right_to_left_boundary and not lies_right_to_right_boundary:
                return self.place_region_fast(lo, mid, site)
            else:
                return None
        else:
            if site.lies_right(self.active[mid]):
                return self.place_region_fast(mid, hi, site)
            else:
                return self.place_region_fast(lo, mid, site)

    def place_region(self, site):
        """
        Places the site in an active region in O(n)
        :param site: Coordinate of the site
        :return: Integer index of the region in the list of active elements
        """

        # Try to place the region fast in O(log n)
        region = self.place_region_fast(0, len(self.active), site)

        if region is not None:

            return region

        else:
            # Otherwise:
            # Find the region whose boundaries have the lowest absolute sum of their determinants regarding the site.
            # If the site falls into a borderline region (self.active[0] or self.active[-1]),
            # The sum of determinants (both summands are either positive or negative) only gets larger with increasing
            # Distance to the site.
            # If the site lies somewhere in the middle, the absolute sum of determinants is close to zero
            # (since the determinant of one border is <= 0, and the other one is >= 0).
            # In rare cases, this is necessary, especially when data points lie very closely to each other
            closest_region = None
            closest_non_zero_det = np.infty

            # In case only one element is active, return 0
            if len(self.active) == 1:
                return 0

            det = np.abs(site.determinant(self.active[1]))

            if det < closest_non_zero_det:
                closest_region = 0
                closest_non_zero_det = det

            for i in range(2, len(self.active) - 2, 2):
                left_boundary = self.active[i - 1]


                right_boundary = self.active[i + 1]

                det = (np.abs(site.determinant(left_boundary) + site.determinant(right_boundary))) / 2

                if site.determinant(left_boundary) < 0 < site.determinant(right_boundary):
                    return i

                if det < closest_non_zero_det:
                    closest_region = i
                    closest_non_zero_det = det

            det = np.abs(site.determinant(self.active[-2]))
            if det < closest_non_zero_det:
                closest_region = len(self.active) - 1
                closest_non_zero_det = det

            return closest_region

    def place_intersection_fast(self, lo, hi, intersection):
        """
        Places an intersection between two boundaries or bisectors in O(log n) by performing binary search on the list
        of active elements.
        :param lo: Integer lower limit
        :param hi: Integer upper limit
        :param intersection: Intersection coordinates
        :return: Quadruple (left boundary, Index of left boundary, right boundary, Index of right boundary)
        Note: If boundary is a bisector, its index is None.
        Also, in rare cases, when points lie close to each other, this method cannot find the region in question.
        Then, None is returned.
        """
        mid = int((lo + hi) / 2)

        # If the mid element corresponds to a site, which has a left and a right boundary
        if isinstance(self.active[mid], Coordinate) and 0 < mid < len(self.active) - 1:

            if mid == 0:
                return self.place_intersection_fast(mid, hi, intersection)
            if mid == len(self.active) - 1:
                return self.place_intersection_fast(lo, mid, intersection)

            # Retrieve left and right boundaries from the list and check whether the intersection lies on them or
            # To their right
            left_boundary = self.active[mid - 1]
            lies_on_left_boundary = intersection.lies_on_boundary(left_boundary)
            lies_right_to_left_boundary = intersection.lies_right(left_boundary)

            right_boundary = self.active[mid + 1]
            lies_on_right_boundary = intersection.lies_on_boundary(right_boundary)
            lies_right_to_right_boundary = intersection.lies_right(right_boundary)

            if lies_on_left_boundary:
                # The order of if-statements is crucial here:
                # First, it needs to be checked if two boundaries meet at the intersection
                # Then, it is checked if possibly a bisector between two sites is involved
                # Otherwise, if the two sites are at the same height, their boundaries AND
                # Their bisector intersect at the same coordinate - but this case is handled separately
                if lies_on_right_boundary:
                    return left_boundary, mid - 1, right_boundary, mid + 1

                if mid - 3 > 0:

                    if intersection.lies_on_boundary(self.active[mid - 3]):

                        return self.active[mid - 3], mid - 3, left_boundary, mid - 1

                if not left_boundary.left_site.__eq__(self.environment) and \
                        not left_boundary.right_site.__eq__(self.environment):
                    left_right_straight = Coordinate(left_boundary.right_site.x - left_boundary.left_site.x,
                                                     left_boundary.right_site.y - left_boundary.left_site.y)

                    # The bisector sits perpendicularly on the straight between left_site and right_site
                    # I.e. it is the normal vector to that line
                    # It should point to the left hand side in walking direction from left_site towards right_site
                    normal_vector = Coordinate(-left_right_straight.y, left_right_straight.x)

                    bisector_start_x = (left_boundary.left_site.x + left_boundary.right_site.x) / 2
                    bisector_start_y = (left_boundary.left_site.y + left_boundary.right_site.y) / 2
                    bisector_start = Coordinate(bisector_start_x, bisector_start_y)

                    # The general bisector between the left and right site,
                    # Sitting perpendicular on the straight connecting both sites
                    left_right_bisector = Boundary(bisector_start, normal_vector,
                                                   left_boundary.left_site, left_boundary.right_site)

                    if intersection.lies_on_boundary(left_right_bisector):

                        return left_boundary, mid - 1, left_right_bisector, None

            elif lies_on_right_boundary:
                # Note that it has already been checked in the previous if-statement whether the intersection
                # Lies both on the left and the right boundary

                # The only boundary it can possibly intersect lies to its right
                if mid + 3 < len(self.active):

                    if intersection.lies_on_boundary(self.active[mid + 3]):

                        return right_boundary, mid + 1, self.active[mid + 3], mid + 3

                # Check for an intersection with the bisector between the left and right site
                if not right_boundary.left_site.__eq__(self.environment) and \
                        not right_boundary.right_site.__eq__(self.environment):
                    left_right_straight = Coordinate(right_boundary.right_site.x - right_boundary.left_site.x,
                                                     right_boundary.right_site.y - right_boundary.left_site.y)

                    # The bisector sits perpendicularly on the straight between left_site and right_site
                    # I.e. it is the normal vector to that line
                    # It should point to the left hand side in walking direction from left_site towards right_site
                    normal_vector = Coordinate(-left_right_straight.y, left_right_straight.x)

                    bisector_start_x = (right_boundary.left_site.x + right_boundary.right_site.x) / 2
                    bisector_start_y = (right_boundary.left_site.y + right_boundary.right_site.y) / 2
                    bisector_start = Coordinate(bisector_start_x, bisector_start_y)

                    # The general bisector between the left and right site,
                    # Sitting perpendicular on the straight connecting both sites
                    left_right_bisector = Boundary(bisector_start, normal_vector,
                                                   right_boundary.left_site, right_boundary.right_site)

                    if intersection.lies_on_boundary(left_right_bisector):

                        return right_boundary, mid + 1, left_right_bisector, None

            # If the intersection lies on none of both boundaries
            if not lies_on_left_boundary and not lies_on_right_boundary:

                # Intersection lies to the right of the right boundary, i.e. somewhere between mid + 1 and hi
                if lies_right_to_left_boundary and lies_right_to_right_boundary:

                    return self.place_intersection_fast(mid, hi, intersection)

                # Impossible: An Intersection cannot lie to the left of the left boundary
                # And to the right of the right boundary at the same time
                # elif not lies_right_to_left_boundary and lies_right_to_right_boundary:

                # Intersection lies to the left of the left boundary, i.e. somewhere between lo and mid - 1
                if not lies_right_to_left_boundary and not lies_right_to_right_boundary:
                    return self.place_intersection_fast(lo, mid, intersection)

        # If the mid element is a boundary
        else:

            # If the intersection lies on that boundary
            if intersection.lies_on_boundary(self.active[mid]):

                # Check for possible intersections with the left and right neighboring boundaries
                if mid - 2 > 0:

                    if intersection.lies_on_boundary(self.active[mid - 2]):

                        return self.active[mid - 2], mid - 2, self.active[mid], mid

                if mid + 2 < len(self.active):

                    if intersection.lies_on_boundary(self.active[mid + 2]):

                        return self.active[mid], mid, self.active[mid + 2], mid + 2

                # Check for an intersection with the bisector between its left and right site
                if not self.active[mid].left_site.__eq__(self.environment) and \
                        not self.active[mid].right_site.__eq__(self.environment):

                    left_right_straight = Coordinate(self.active[mid].right_site.x - self.active[mid].left_site.x,
                                                     self.active[mid].right_site.y - self.active[mid].left_site.y)

                    # The bisector sits perpendicularly on the straight between left_site and right_site
                    # I.e. it is the normal vector to that line
                    # It should point to the left hand side in walking direction from left_site towards right_site
                    normal_vector = Coordinate(-left_right_straight.y, left_right_straight.x)

                    bisector_start_x = (self.active[mid].left_site.x + self.active[mid].right_site.x) / 2
                    bisector_start_y = (self.active[mid].left_site.y + self.active[mid].right_site.y) / 2
                    bisector_start = Coordinate(bisector_start_x, bisector_start_y)

                    # The general bisector between the left and right site,
                    # Sitting perpendicular on the straight connecting both sites
                    left_right_bisector = Boundary(bisector_start, normal_vector,
                                                   self.active[mid].left_site, self.active[mid].right_site)

                    if intersection.lies_on_boundary(left_right_bisector):

                        return self.active[mid], mid, left_right_bisector, None

            # If the intersection does not lie on the boundary
            else:
                # Check if the intersection lies to its right or left
                # In case the intersection lies right, search the upper half
                if intersection.lies_right(self.active[mid]):

                    return self.place_intersection_fast(mid, hi, intersection)

                # Otherwise, search the lower half
                else:

                    return self.place_intersection_fast(lo, mid, intersection)

    def place_intersection(self, intersection):
        """
        Places an intersection in the list of active sites in O(n)
        :param intersection: Intersection Coordinates
        :return: Quadruple (left boundary, index of left boundary, right boundary, Index of right boundary)
        Note: If boundary is a bisector, its Index is None.
        """

        # Try to place the Intersection in O(log n)
        region = self.place_intersection_fast(0, len(self.active), intersection)

        if region is not None:
            return region
        else:
            # Otherwise, go over all active Elements and choose the Site to whose boundaries/ bisectors
            # The given Intersection is closest
            # In rare cases, this is necessary, especially when data points lie very closely to each other
            closest = None
            closest_num = np.infty

            # First check if the Intersection is between two ordinary boundaries
            for i in range(2, len(self.active) - 2, 2):

                cur_site = self.active[i]

                left_boundary = self.active[i - 1]
                left_site = self.active[i - 2]

                right_boundary = self.active[i + 1]
                right_site = self.active[i + 2]

                if np.abs(intersection.determinant(left_boundary) + intersection.determinant(right_boundary)) < closest_num:
                    closest_num = np.abs(intersection.determinant(left_boundary) + intersection.determinant(right_boundary))
                    closest = left_boundary, i - 1, right_boundary, i + 1

            # Secondly, check if the Intersection lies between a bisector and a boundary
            for i in range(2, len(self.active) - 2, 2):

                cur_site = self.active[i]

                if not cur_site.__eq__(self.environment):

                    left_boundary = self.active[i - 1]
                    left_site = self.active[i - 2]

                    right_boundary = self.active[i + 1]
                    right_site = self.active[i + 2]

                    if not left_site.__eq__(self.environment):
                        # The straight connecting the left and right site
                        # Needed to calculate the bisector between them
                        left_right_straight = Coordinate(cur_site.x - left_site.x,
                                                         cur_site.y - left_site.y)

                        # The bisector sits perpendicularly on the straight between left_site and right_site
                        # I.e. it is the normal vector to that line
                        # It should point to the left hand side in walking direction from left_site towards right_site
                        normal_vector = Coordinate(-left_right_straight.y, left_right_straight.x)

                        bisector_start_x = (left_site.x + cur_site.x) / 2
                        bisector_start_y = (left_site.y + cur_site.y) / 2
                        bisector_start = Coordinate(bisector_start_x, bisector_start_y)

                        # The general bisector between the left and right site,
                        # Sitting perpendicular on the straight connecting both sites
                        left_right_bisector = Boundary(bisector_start, normal_vector, left_site, cur_site)

                        if np.abs(intersection.determinant(left_boundary) + intersection.determinant(left_right_bisector)) < closest_num:
                            closest_num = np.abs(intersection.determinant(left_boundary) + intersection.determinant(left_right_bisector))
                            closest = left_boundary, i - 1, left_right_bisector, None

                    if not right_site.__eq__(self.environment):
                        # The straight connecting the left and right site
                        # Needed to calculate the bisector between them
                        left_right_straight = Coordinate(right_site.x - cur_site.x,
                                                         right_site.y - cur_site.y)

                        # The bisector sits perpendicularly on the straight between left_site and right_site
                        # I.e. it is the normal vector to that line
                        # It should point to the left hand side in walking direction from left_site towards right_site
                        normal_vector = Coordinate(-left_right_straight.y, left_right_straight.x)

                        bisector_start_x = (cur_site.x + right_site.x) / 2
                        bisector_start_y = (cur_site.y + right_site.y) / 2
                        bisector_start = Coordinate(bisector_start_x, bisector_start_y)

                        # The general bisector between the left and right site,
                        # Sitting perpendicular on the straight connecting both sites
                        left_right_bisector = Boundary(bisector_start, normal_vector, cur_site, right_site)

                        if np.abs(intersection.determinant(right_boundary) + intersection.determinant(left_right_bisector)) < closest_num:
                            closest_num = np.abs(intersection.determinant(right_boundary) + intersection.determinant(left_right_bisector))
                            closest = right_boundary, i + 1, left_right_bisector, None

            return closest

    def insert_intersection(self, intersection):
        """
        Inserts an intersection into the PriorityQueue
        :param intersection: Intersection coordinates
        """
        intersection_key = self.project_onto_sweeping_direction(intersection)

        q_node = QueueNode(intersection_key, intersection)

        self.intersection_pq.insert(q_node)

    def delete_intersection(self, intersection):
        """
        Deletes an intersection from the Priority-Queue
        :param intersection: coordinate
        """
        intersection_key = self.project_onto_sweeping_direction(intersection)

        # Delete any intersection of the left and right boundary beyond the new site
        self.intersection_pq.delete_key(intersection_key)

    def to_string(self):
        """
        Puts the list of active Elements into a readable string.
        :return: String of the list of active Elements
        """

        string = "["

        for elem in self.active:
            if isinstance(elem, Boundary):
                string += " B: " + elem.to_string()
            else:
                string += " R: " + elem.to_string()
        string += " ]"

        return string

    def project_onto_sweeping_direction(self, coordinate):
        """
        Calculates the orthogonal Projection and Rejection of a coordinate onto the sweepline
        :param coordinate: Coordinate to be projected/ rejected
        :return: Coordinate of (rejection.x, projection.y)
        """

        projection = (coordinate.x * self.sweeping_direction.x) + (coordinate.y * self.sweeping_direction.y)
        rejection = (self.sweeping_direction.y * coordinate.x) - (self.sweeping_direction.x * coordinate.y)

        return Coordinate(rejection, projection)

    def get_all_boundaries(self):
        """
        :return: List of all boundaries
        """
        return self.boundaries

    def get_regions2boundaries(self):
        """
        :return: Dictionary mapping coordinates (i.e., sites) to the list of boundaries enclosing their OrientedVoronoi region
        """
        return self.regions_to_boundaries

    def get_open_regions(self):
        """
        :return: List of OrientedVoronoi-Regions which are not closed
        """
        return self.open_regions

    def get_corners(self):
        """
        :return: quadruple of lower-left x- and y coordinates and upper-right x- and y-coordinates
        """
        return self.lower_left_x, self.lower_left_y, self.upper_right_x, self.upper_right_y

    def compare(self, point_1, point_2):
        """
        Compares the coordinates of two given sites in first y- and then x-dimension
        :param point_1: Point
        :param point_2: Point
        :return: point_1 is less or equal than point_2
        """

        if point_1.y == point_2.y:
            return point_1.x > point_2.x
        else:
            return point_1.y > point_2.y

    def euclidean_distance(self, key1, key2):
        """
        Computes the euclidean distance between two coordinates
        :param key1: Coordinate
        :param key2: Coordinate
        :return: Euclidean distance between key1 and key2.
        """
        return np.sqrt((key1.x - key2.x)**2 + (key1.y - key2.y)**2)
