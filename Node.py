#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Data structure to represent a node in the Priority Queue
@author Tobias Elßner
"""

class QueueNode:

    def __init__(self, key, val=None):
        """
        Initializes a QueueNode.
        :param key: Generic key by which the QueueNode is ranked in priority
        :param val: Generic value which is stored under the key
        """
        self.key = key
        self.value = val

        # The position at which the QueueNode is stored in the queue
        # Necessary for swim/ sink operations
        self.pos = 0

    def __eq__(self, other):
        """
        Checks if QueueNode is equal to another one. Needed for hashing.
        :param other: QueueNode
        :return: Boolean whether QueueNode is equal to other.
        """

        # Compare only values, as keys can change (see Prim's Algorithm)
        return self.value == other.value

    def __ne__(self, other):
        """
        Checks if QueueNode is unequal to another one.
        :param other: QueueNode
        :return: Boolean whether QueueNode is unequal to other.
        """
        return self.value != other.value

    def __hash__(self):
        """
        Computes hash-code of QueueNode based on its value.
        The same value should not have different keys, however, different values can have the same key.
        Therefore, the value is used as basis for the hash-code.
        :return: hash-code of QueueNode
        """
        return hash(self.value)


"""
SetNode data structure to compute least common ancestors based on union-find.
Following Chapter 21.3, p. 571, in Introduction to Algorithms, Third Edition (2009), by 
Thomas H. Cormen, Charles E. Leiserson. Ronald L. Rivest & Clifford Stein
@author Tobias Elßner
"""


class SetNode:

    def __init__(self, num):
        """
        Initializes a SetNode.
        :param num: Integer identifyer
        """

        # Each SetNode has an integer identifier
        # A None-parent (the set it belongs to) by default
        # A None-ancestor (the value of the set) by default
        # A rank to indicate the size of its set
        # And a boolean to check if it has been processed (i.e., if it already belongs to a set)
        self.key = num
        self.parent = None
        self.ancestor = None
        self.rank = 0
        self.visited = False


    def make_set(self):
        """
        Make a new set with the Node as only member.
        """

        # The current SetNode is the only member in this Set, thus is its own parent
        self.parent = self
        self.rank = 0

    def union(self, other):
        """
        Union the set with another.
        :param other: Other set.
        """

        # Find the set to which the Node belongs and link it to the other node's set
        self.find_set().link(other.find_set())

    def link(self, other):
        """
        Links to sets according to their ranks
        :param other: other SetNode
        :return:
        """

        # Link the set with the smaller rank to the one with the larger rank
        if self.rank > other.rank:
            other.parent = self
        else:
            self.parent = other

            # When both ranks are equal, self is linked to the other set, whose rank is increased by one
            if self.rank == other.rank:
                other.rank += 1

    def find_set(self):
        """
        Finds the set the current SetNode belongs to.
        :return: The parental SetNode the current Node belongs to.
        """

        # Recursively call the next parent, until the root of the set is found
        # That is, the SetNode which is its own parent (cf. make_set())
        if self is not self.parent:
            self.parent = self.parent.find_set()

        return self.parent


"""
TreeNode data structure for the MST.
@author Tobias Elßner
"""


class TreeNode:

    def __init__(self, num):
        """
        Initializes a tree node.
        :param num: Integer identifier
        """

        # Each node has an integer as unique identifier
        # A parent node (which is by default itself)
        # An empty set of child nodes
        # And an arbitrary integer value
        self.key = num
        self.parent = self
        self.children = set()
        self.value = 0

    def add_child(self, node):
        """
        Adds node as a child.
        :param node: A new child Node
        """

        self.children.add(node)

        # Set the node's parent to the one to which it is added to as child
        node.parent = self

    def remove_edge_to_child(self, node):
        """
        Removes node from the set of children.
        :param node: The node which is removed
        """

        if node in self.children:
            self.children.remove(node)

            # Set the node's parent to itself
            node.parent = self

    def has_children(self):
        """
        Checks if the node has children.
        :return: Boolean whether the node has children
        """
        return len(self.children) != 0

    def __eq__(self, other):
        """
        Checks if two nodes are equal. Needed for hashing.
        :param other: other Node
        :return: Boolean whether node and other are equal.
        """
        return (self.key, self.value) == (other.key, other.value)

    def __ne__(self, other):
        """
        Checks if two nodes are unequal. Needed for hashing.
        :param other: other Node
        :return: Boolean whether node and other are unequal.
        """
        return (self.key, self.value) != (other.key, other.value)

    def __hash__(self):
        """
        Computes the hash-code of the node. Needed for hashing.
        :return: Hash-Code
        """
        return hash((self.key, self.value))
