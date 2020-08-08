import random as rand
import numpy as np
import math


class Node:

    # Constructor
    def __init__(self, val):
        self.value = val
        self.weights = []
        self.value = 0

    # Set self.weights to param w
    def setWeights(self, w):
        self.weights = w
    
    # Set self.value to param v
    def setValue(self, v):
        self.value = v

    # Sigmoids input x
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    # Set self.weights to random double (-1, 1)
    def setRandomWeights(self, input_layer):
        self.weights = np.random.uniform(
            low=-1, high=1, size=len(input_layer.nodes))

    # Sums a product of all last layer values with self.weights
    # and sets value to the sigmoided result
    def forward(self, input_layer):
        input_list = input_layer.gather()
        a = sum([a*b for a, b in zip(input_list, self.weights)])
        self.value = self.sigmoid(a)
        

    # Prints self.value
    def print_value(self):
        print(self.value)


class Layer:

    # Constructor
    def __init__(self, size):
        self.size = size
        self.nodes = [Node(0) for _ in range(self.size)]

    # Collects all nodes in the layer and returns them as a list
    def gather(self):
        return [self.nodes[i].value for i in range(self.size)]

    # Sets each node's weight values in the layer to a random
    # double (-1, 1)
    def setRandomLayerWeights(self, prev_layer):
        for i in range(self.size):
            self.nodes[i].setRandomWeights(prev_layer)

    # Computes new value for each node by suming prev
    # layers values with this layers node weight values
    def forwardLayer(self, prev_layer):
        for i in range(self.size):
            self.nodes[i].forward(prev_layer)

    # Sets layers nodes to be a value from l (list)
    def setLayerValues(self, l):
        for i in range(self.size):
            self.nodes[i].setValue(l[i])

    # Prints Layer Node Values
    def printValues(self):
        for i in range(self.size):
            self.nodes[i].print_value()

    # Print Layer Node Weights
    def printWeights(self):
        for i in range(self.size):
            print(self.nodes[i].weights)


class Mesh:

    # Constructor 
    def __init__(self, input_sizes):
        self.size = len(input_sizes)
        self.input_sizes = input_sizes
        self.input_layer = 0
        self.output = 0
        self.layers = [Layer(input_sizes[i]) for i in range(len(input_sizes))]

    # Performs a set for the input layer, random init if specified,
    # and then pushed all layers in the mesh
    def compute(self, input_layer, random):
        self.setInputLayer(input_layer)
        if(random == 1):
            self.initAllWeightsRandom()
        self.forwardMesh()

    # Sets self.input_layer to param input_layer
    def setInputLayer(self, input_layer):
        self.input_layer = input_layer

    # Pushes each layer of the mesh forward by suming the products
    # of the weights for the current layers node and the previous 
    # layers node values

    # [Driver Function]
    def forwardMesh(self):
        self.layers[0].forwardLayer(self.input_layer)
        for i in range(1, len(self.layers)):
            self.layers[i].forwardLayer(self.layers[i-1])
        self.output = self.layers[len(self.layers) - 1]

    # Sets all layers node weights to a random value (-1, 1)
    def initAllWeightsRandom(self):
        self.layers[0].setRandomLayerWeights(self.input_layer)
        for i in range(1, len(self.layers)):
            self.layers[i].setRandomLayerWeights(self.layers[i-1])

    # Prints Full stats on outcome of a pass through of the mesh
    def full_print(self):
        print("Input Layer: ")
        self.input_layer.printWeights()
        self.input_layer.printValues()
        print("")

        for i in range(len(self.layers)):
            print("Layer " + str(i) + ":")
            self.layers[i].printWeights()
            self.layers[i].printValues()
            print("")

    def print_layer(self, index):
        self.layers[index].printValues()


class Merger:

    def __init__(self):
        return

    def merge(self, mesh1, mesh2):
        child_mesh = Mesh(mesh1.input_sizes)
        child_layers = []
        for r1, r2 in zip(mesh1.layers, mesh2.layers):

            child_nodes = []

            for q1, q2 in zip(r1.nodes, r2.nodes):
                child_nodes.append(self.merge_weights(q1, q2))

            l = Layer(r1.size)
            l.nodes = child_nodes
            child_layers.append(l)

        child_mesh.layers = child_layers

        return child_mesh

    def merge_weights(self, partner1, partner2):
        merge_dec = np.random.randint(
            low=0, high=2, size=len(partner1.weights))
        child_weights = []
        for ind in range(len(merge_dec)):
            child_weights.append(
                partner2.weights[ind] if merge_dec[ind] == 1 else partner1.weights[ind])

        child_node = Node(0)
        child_node.setWeights(child_weights)

        return child_node

# Loopable


'''
input_values = [9, 54, 18, 745]
input_layer = Layer(len(input_values))
input_layer.setLayerValues(input_values)

print('\n///////////////////////////////////////\n     Mesh #1     \n///////////////////////////////////////\n')
m = Mesh([3, 5, 2, 1])
m.compute(input_layer, 1)
m.full_print()

print('\n///////////////////////////////////////\n     Mesh #2     \n///////////////////////////////////////\n')

g = Mesh([3, 5, 2, 1])
g.compute(input_layer, 1)
g.full_print()

print('\n///////////////////////////////////////\n     Mesh #3     \n///////////////////////////////////////\n')

mer = Merger()
t = mer.merge(m, g)

t.compute(input_layer, 0)
t.full_print()


'''