#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Computes an MST based on Prim's algorithm.
Also, it calculates the longest edge between two nodes using Tarjans least common ancestor technique.
@author Tobias Elßner
"""


import numpy as np
from scipy.spatial.distance import euclidean as eu

from Node import QueueNode, TreeNode, SetNode
from PriorityQueue import PriorityQueue


class PrimMST:

    def __init__(self, dp):
        """
        Initializes an MST and its attributes.
        :param dp: 2-D numpy-array containing the coordinates for the MST nodes.
        """
        self.data_points = dp
        self.distance_matrix = self.get_adjacency_matrix(self.data_points)

        self.num_of_points = self.data_points.shape[0]

        # Initialize the data structures for computing the MST, its ordered edge tree, and the lowest common ancestors
        # The MST itself
        self.mst = None
        self.weight = 0

        # Minimum difference between two edges in the MST
        self.min_difference = np.infty

        # Number of nodes in the tree between two vertices
        self.nodes_between = [[0*i for i in range(self.num_of_points)] for j in range(self.num_of_points)]

        # Maps each data point index to its TreeNode
        self.mst_nodes = [TreeNode(i) for i in range(self.num_of_points)]

        # The edge tree is a binary tree, where each leaf node represents a node in the MST
        # And each internal node is the longest edge connecting its two subtrees
        self.edge_tree = None
        self.edge_nodes = [TreeNode(i) for i in range(self.num_of_points)]

        # Initialize (n + n - 1) SetNodes to compute the least common ancestors
        self.set_nodes = [SetNode(i) for i in range(self.num_of_points + self.num_of_points - 1)]

        # Initialize the 2-D List storing th ancestor edges consisting of parent key, child key, and edge length
        self.longest_common_edges = [[0 * j for j in range(self.num_of_points)]
                                     for i in range(self.num_of_points)]

        # Build the MST
        self.build_mst()

        # Compute the lowest common ancestors of the edge tree
        self.lca(self.edge_tree)

    def get_adjacency_matrix(self, dp):
        """
        Computes the adjacency matrix of the data points.
        :param dp: 2-D numpy-array containing coordinates
        :return:
        """

        num_of_points = dp.shape[0]

        # Create an empty quadratic matrix to store the distances between the points
        distance_matrix = np.zeros((num_of_points, num_of_points))

        # Calculate for each data point the euclidean distance to all others
        for u in range(num_of_points):

            u_coordinates = dp[u, :]

            for v in range(num_of_points):
                v_coordinates = dp[v, :]
                distance_matrix[u, v] = eu(u_coordinates, v_coordinates)

        return distance_matrix

    def build_mst(self):
        """
        Builds an MST following Prim's Algorithm in O(n²)
        The MST is stored in a tree structure.
        """

        self.weight = 0

        # Begin with a random vertex
        start_vertex = np.random.randint(self.num_of_points)

        # Tree structure of the MST
        tmp_mst = [TreeNode(i) for i in range(self.num_of_points)]

        # Maximum priority queue to sort the (n-1) edges in descending order for the edge tree
        max_pq = PriorityQueue(self.num_of_points - 1, comp=self.is_smaller, dist=eu)

        # Depth-first search style implementation of Prim's algorithm for fully connected graphs
        unvisited = {}

        for i in range(self.num_of_points):

            if i != start_vertex:
                unvisited.update({i: (self.distance_matrix[start_vertex, i], (start_vertex, i))})

        while len(unvisited.keys()) != 0:

            min_weight = np.infty
            min_vertex = None
            min_edge = None

            # Find the unvisited vertex with the lightest connection
            for vertex in unvisited.keys():
                weight, edge = unvisited[vertex]

                if weight < min_weight:
                    min_weight = weight
                    min_edge = edge
                    min_vertex = vertex

            # Delete this vertex
            unvisited.pop(min_vertex)

            # Set the new vertex' parent to that node in the subtree with which it is connected
            tmp_mst[min_vertex].parent = tmp_mst[min_edge[0]]

            # Update weights for the remaining unvisited vertices accordingly
            for vertex in unvisited.keys():

                # If an connection from the newly added min_vertex to some unvisited node is lighter
                # Than the existing one, update both the weight and the edge
                if self.distance_matrix[min_vertex, vertex] < unvisited[vertex][0]:
                    unvisited[vertex] = (self.distance_matrix[min_vertex, vertex], (min_vertex, vertex))

        """
        # Priority Queue Implementation of Prim's algorithm for graphs of lower density
        # Minimum priority queue to sort the (n-1) edges in ascending order for Prim's algorithm
        min_pq = PriorityQueue(self.num_of_points, comp=self.is_greater, dist=eu)

        for i in range(self.num_of_points):
            if i != start_vertex:

                # Initialize the queue with the vertices and infinite weights
                # Key is the minimum distance to the tree built so far, value is the index of the data point
                node = QueueNode(np.infty, i)
                min_pq.insert(node)

            else:
                # Give the start vertex zero weight to begin with
                node = QueueNode(0, i)
                min_pq.insert(node)

        # As long as there are vertices which are not yet included into the MST
        while not min_pq.is_empty():

            # Get the vertex with the lowest weight from the queue
            u = min_pq.pop().value

            for node in min_pq.get_nodes():

                # Insert here additional check if an edge (u, node) exists;
                # Otherwise, a fully-connected graph is assumed
                # And the algorithm additionally performs n-times log(n) insertions in the priority queue

                # If the distance from any node to the current vertex u is less than before:
                if min_pq.compare(node.key, self.distance_matrix[u, node.value]):

                    # Set the node's parent to u
                    tmp_mst[node.value].parent = tmp_mst[u]

                    # And update its priority
                    min_pq.update(node, self.distance_matrix[u, node.value])

        """
        # Put the MST together
        for i in range(self.num_of_points):

            # Root node has itself as parent
            if tmp_mst[i].parent.key != i:

                edge_length = self.distance_matrix[i, tmp_mst[i].parent.key]

                # Insert the length of the edge into the queue
                # Key is the edge length, value is the index of the parent and the index of the child
                max_pq.insert(QueueNode(edge_length, (tmp_mst[i].parent.key, i)))

                # Add the length to the MST's overall weight
                self.weight += edge_length

                # Add the child to the parent's node
                self.mst_nodes[tmp_mst[i].parent.key].add_child(self.mst_nodes[tmp_mst[i].key])

            else:
                # Set the mst to root node; this can be any node
                self.mst = tmp_mst[i]

        # Build the edge tree
        self.build_edge_tree(max_pq)

    def build_edge_tree(self, max_pq):
        """
        Builds an edge tree following Monma & Suri: Leaves are MST nodes, intermediate nodes denote the longest edges
        connecting two subtrees. Those edges are sorted in descending order; root represents the longest edge.
        :param max_pq: maximum PriorityQueue containing the descendingly sorted edges
        """

        # Copy the tree nodes
        for i in range(self.num_of_points):
            edge_node = self.edge_nodes[i]
            edge_node.parent = self.edge_nodes[self.mst_nodes[i].parent.key]

            for child in self.mst_nodes[i].children:

                edge_node.add_child(self.edge_nodes[child.key])

        # The additional (n-1) edges for the n vertices {0, 1, ..., n-1} in the mst get extra keys starting with n
        # This way, edge tree nodes can be distinguished by being >= self.num_of_points
        edge_number = self.num_of_points

        self.edge_tree = TreeNode(edge_number)

        self.edge_nodes.append(self.edge_tree)

        prev_length = None

        while not max_pq.is_empty():

            q_node = max_pq.pop()
            edge_length = q_node.key

            if prev_length is None:
                prev_length = edge_length

            else:
                difference = prev_length - edge_length

                if difference < self.min_difference:
                    self.min_difference = difference
                    prev_length = edge_length

            parent_key, child_key = q_node.value

            parent_node, child_node = self.edge_nodes[parent_key], self.edge_nodes[child_key]

            # Cut the ties between the parent and the child node
            parent_node.remove_edge_to_child(child_node)

            # Find root node of parent tree (the child_node is automatically the root node of the child tree)
            # The root has to be part of the MST, and must not be an edge_node (which have keys >= num_of_points)
            while parent_node.parent.key < self.num_of_points and parent_node.parent is not parent_node:
                parent_node = parent_node.parent

            # By default, the edge node is the root
            edge_node = self.edge_tree

            # If the root has already children, a new edge node is initialized,
            # And its parent is the edge node above the parent node
            if self.edge_tree.has_children():
                edge_number += 1

                edge_node = TreeNode(edge_number)
                parent_node.parent.add_child(edge_node)
                parent_node.parent.remove_edge_to_child(parent_node)
                self.edge_nodes.append(edge_node)

            # Value comprises of all information: parent, child, and edge length
            # Primitive types (int, int, float) allow simple comparison and hashing
            edge_node.value = ((parent_key, child_key), edge_length)

            # Add the root node of the parental tree and the child node to the newly introduced edge node
            edge_node.add_child(parent_node)
            edge_node.add_child(child_node)

    def lca(self, u):
        """
        Following the pseudo-code of Introduction to Algorithms, Third Edition, p. 584.
        Calculates the lowest common ancestors in the edge tree to find the longest edge between two MST nodes.
        :param u: TreeNode in the edge tree.
        """

        self.set_nodes[u.key].make_set()
        self.set_nodes[u.key].find_set().ancestor = self.set_nodes[u.key]

        for child in u.children:

            self.lca(child)

            self.set_nodes[u.key].union(self.set_nodes[child.key])

            self.set_nodes[u.key].find_set().ancestor = self.set_nodes[u.key]


        self.set_nodes[u.key].visited = True

        for v in range(self.num_of_points):

            if self.set_nodes[v].visited and u.key < self.num_of_points:

                # The distance of the edge (i.e., the value of the edge_tree_node with the same key as the set_node)
                # Is stored in the numpy matrix under the keys of the nodes (which correspond to the data  points)
                self.longest_common_edges[u.key][v] = self.longest_common_edges[v][u.key] = \
                    self.edge_nodes[self.set_nodes[v].find_set().ancestor.key].value

    def is_greater(self, key1, key2):
        """
        Compares the float keys key1 and key2
        :param key1: Float
        :param key2: Float
        :return: key1 is greater than key2.
        Thus, smaller keys "swim", and larger ones "sink"; the result is a MIN-PQ.
        """

        return key1 > key2

    def is_smaller(self, key1, key2):
        """
        Compares the float keys key1 and key2
        :param key1: Float
        :param key2: Float
        :return: key1 is less than key2.
        Thus, larger keys "swim", and smaller ones "sink"; the result is a MAX-PQ.
        """
        return key1 < key2
