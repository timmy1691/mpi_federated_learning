from mpi4py import MPI
from phe import paillier

import numpy as np

# what do we need to do?
# create the encryption keys
# send the encrypted gradients and the encrypted hessian
# perform the split, all of the data will need to be allocated to the left or the right
# then need to query 
# only need to store the condition for split in the node
# the nodes contain the id of the data points that are in the node.
# so need to track where we end up.

# we have the tree, then we have the node, then we have the leaves

# a leaf is a node that does not have any next nodes.
# what is the valuation of each node?
# node valuation is known upon construct of the node


data = np.random.uniform(0, 2, size = (1000, 2))
target = np.random.randint(0, 2, size=(1000,1))

combined_data = np.concatenate((data, target), axis=1)

class Node():

    def __init__(self, previous_Node, feature, condition, thresh, value):
        self.previous = previous_Node
        self.feature = feature
        self.condition = condition
        self.threshold = thresh
        self.value = value

    def value_assignment(self, value):
        if self.left_Node == None and self.right_Node == None:
            self.value = value
    
    def value_calc(self, )

    def extend_new_Node(self, nextNode):
        self.left_Node = 
        self.right_Node = 


class Tree():
    def __init__(self, root):
        self.leaf = root


