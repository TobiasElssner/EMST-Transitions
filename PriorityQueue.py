#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implements a priority queue on the basis of a binary heap.
See Algorithms, Chapter 2.4, by Sedgewick and Wayne, fourth edition.
@author Tobias Elßner
"""

from blist import blist
from Node import  QueueNode
import numpy as np
from Coordinate import Coordinate


class PriorityQueue:

    def __init__(self, max_size, comp, dist):
        """
        Initializes a PriorityQueue.
        :param max_size: Maximal size
        :param comp: Method to compare keys
        :param dist: Method to calculate the distance between keys
        """
        self.heap = [0 * j for j in range(max_size + 1)]

        self.compare = comp
        self.distance = dist

        # Set of nodes to check if values are already present
        self.nodes = set()

        # Dictionary to map keys to nodes
        # Allows multiple nodes under the same key
        self.key_to_node = {}

        # Current size
        self.size = 0

        # Accuracy
        self.eps = 0.01

    def insert(self, node):
        """
        Inserts a QueueNode into the PriorityQueue.
        :param node: QueueNode
        """

        # Insert node only if it is not already in the queue
        if node.key not in self.key_to_node or node not in self.key_to_node[node.key]:

            # Increase the number of current elements in the queue
            self.size += 1

            # Insert the new element at the end of the queue
            node.pos = self.size
            self.heap[self.size] = node

            # Add the node to the set of nodes
            self.nodes.add(node)

            # Insert the node into the dictionary
            self.insert_into_dic(node)

            # 'Swim' the new node as far as its key allows
            self.swim(self.size)

    def delete_node(self, node):
        """
        Deletes a QueueNode from the PriorityQueue.
        :param node: QueueNode
        :return: Boolean whether deletion was successful
        """

        # If the Node is no longer present, it cannot be deleted
        if (node.pos > self.size) or (self.heap[node.pos] != node) or (node.key not in self.key_to_node):
            return False
        else:
            # Exchange node with the last node in the queue
            self.exchange(node.pos, self.size)

            # Decrease the number of current elements in the queue
            self.size -= 1

            # Overwrite the node's position in the queue
            self.heap[self.size + 1] = 0

            # 'Sink' the exchanged node at its new position as far as its key allows
            self.sink(node.pos)

            # Remove then the node also from the set of nodes
            if node in self.nodes:
                self.nodes.remove(node)

            # Remove the node from the keys-to-nodes dictionary
            self.remove_from_dic(node)

            # Finally, delete the QueueNode object
            del node

            return True

    def delete_pos(self, pos):
        """
        Deletes QueueNode at a certain position in the PriorityQueue.
        :param pos: Integer position
        :return: Boolean whether deletion was successful
        """

        # If the position is out of bounce, its QueueNode cannot be deleted
        if pos > self.size:
            return False
        else:
            # Retrieve the node at the given position
            node = self.heap[pos]

            # Delete it
            self.delete_node(node)


    def delete_key(self, key):
        """
        Deletes nodes with the same key.
        :param key: Generic key
        :return: Boolean whether deletion was successful
        """

        # If the key is in the dictionary, delete all associated nodes
        if key in self.key_to_node:

            for node in self.key_to_node[key]:
                self.delete_node(node)

            return True
        else:
            # Additional check to handle rounding errors
            # Degrades deletion runtime to O(n)
            # (And thus overall OrientedVoronoi computation to O(n²))
            # Looks for all keys in the dictionary with distance less than eps and selects the closest one
            if len(self.key_to_node) > 0:
                found = False
                alternative_key = None
                min_dist = np.infty

                for other in self.key_to_node.keys():

                    if self.distance(key, other) < min_dist:
                        alternative_key = other
                        min_dist = self.distance(key, other)

                if min_dist < self.eps and alternative_key in self.key_to_node:
                    for node in self.key_to_node[alternative_key]:
                        self.delete_node(node)
                    return True

        return False

    def update(self, node, new_key):
        """
        Gives a node a new priority.
        :param node: QueueNode
        :param new_key: Generic new key
        :return: Boolean whether update was successful.
        """

        # If node is not in queue, its key cannot be updated
        if node not in self.nodes:
            return False

        # Remove the node from the dictionary
        cur_key = node.key

        self.remove_from_dic(node)

        # Overwrite its old key with the new one
        node.key = new_key

        # Store it again in the dictionary
        self.insert_into_dic(node)

        # Store it again in the queue at its old position
        # TODO: Necessary??
        self.heap[node.pos] = node

        # Sink or swim the node, depending on the comparison between its old and new key
        if self.compare(new_key, cur_key):
            self.sink(node.pos)
        else:
            self.swim(node.pos)

        return True

    def pop(self):
        """
        Pop the QueueNode with the highest priority.
        :return: QueueNode with the highest priority in the queue.
        """

        # Get the node which is front-most
        # Note that the queue starts at index 1 for the ease of computation
        node = self.heap[1]

        # Exchange it with the last node
        self.exchange(1, self.size)

        # Decrease the number of current elements in the queue
        self.size -= 1

        # Overwrite the popped node in the queue
        self.heap[self.size + 1] = 0

        # Remove the popped node from the set and the dictionary
        if node in self.nodes:
            self.nodes.remove(node)

        self.remove_from_dic(node)

        # Sink the node which has been put at front as far as its key allows
        self.sink(1)

        return node

    def peek(self):
        """
        Peeks at the node with the highest priority without popping it.
        :return: QueueNode with highest priority.
        """
        return self.heap[1]

    def sink(self, index):
        """
        Sink the node at given index.
        :param index: Integer index
        """

        # Exchange node with nodes from lower levels of the binary heap (having higher indices in the array)
        # The two children below index are at (index * 2) and (index * 2) + 1
        while index * 2 <= self.size:
            j = 2 * index
            if j < self.size and self.compare(self.heap[j].key, self.heap[j + 1].key):
                j += 1

            if not self.compare(self.heap[index].key, self.heap[j].key):
                break

            self.exchange(index, j)
            index = j

    def swim(self, index):
        """
        Swims the node at given index.
        :param index: Integer index.
        """

        # Exchange node with nodes from higher levels of the binary heap (having lower indices in the array)
        # The parent pf node at index are at int((index / 2)
        while int(index) > 1 and self.compare(self.heap[int(index / 2)].key, self.heap[int(index)].key):
            self.exchange(int(index / 2), int(index))
            index /= 2

    def exchange(self, pos1, pos2):
        """
        Exchanges the nodes from two given positions.
        :param pos1: Integer position
        :param pos2: Integer position
        :return:
        """

        # Exchange the position pointers of the nodes at those positions
        self.heap[pos1].pos = pos2
        self.heap[pos2].pos = pos1

        # Exchange the actual positions of those nodes
        tmp_node = self.heap[pos1]
        self.heap[pos1] = self.heap[pos2]
        self.heap[pos2] = tmp_node

    def remove_from_dic(self, node):
        """
        Removes QueueNode from dictionary.
        :param node: QueueNode
        """

        # Sanity check that the node's key is in fact in the dictionary
        if node.key in self.key_to_node:

            # Either delete the whole entry (if node is the only member of the set)
            if len(self.key_to_node[node.key]) == 1:
                self.key_to_node.pop(node.key)
            # Or remove node from the set, when there are other nodes having the same key
            else:
                self.key_to_node[node.key].remove(node)

    def insert_into_dic(self, node):
        """
        Inserts QueueNode into the dictionary.
        :param node: QueueNode
        """

        # If its key is unique in the queue, initialize a new entry
        if node.key not in self.key_to_node:
            self.key_to_node[node.key] = {node}

        # If there is already another node with the same key, add the new node to the existing set
        else:
            self.key_to_node[node.key].add(node)

    def get_size(self):
        """
        Returns the number of QueueNodes in the PriorityQueue
        :return: Integer
        """
        return self.size

    def is_empty(self):
        """
        Checks if the PriorityQueue is empty.
        :return: Boolean whether the PriorityQueue is empty
        """
        return self.size == 0

    def get_nodes(self):
        """
        Returns all nodes in the PriorityQueue.
        :return: Set of nodes
        """
        return self.nodes
