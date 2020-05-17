import random as rand
import numpy as np
import math 

class Node:
    
    weights = []
    value = 0
    
    def __init__(self, val):  
            self.value = val
                               
    def pass_weights_in(self, w):
        self.weights = w
    
    def set_value(self, v):
        self.value = v
    
    def sigmoid(self, x):        
        return 1/(1 + np.exp(-x)) 
    
    def push_forward(self, input_layer):
        self.weights = np.random.uniform(low = -1, high = 1, size = len(input_layer.nodes))
        input_list = input_layer.gather_node_values()  
        a = sum([a*b for a,b in zip(input_list, self.weights)])
        self.value = self.sigmoid(a)
        
    def alter_weight_at_index(self, index, val):
        self.weights[index] = val
        
    def print_value(self):
        print(self.value)
        
        
class Layer:
    
    nodes = []
    
    def __init__(self, size):
        self.size = size
        self.nodes = [Node(0) for _ in range(self.size)]
            
    def gather_node_values(self):
        l = [self.nodes[i].value for i in range(self.size)]
        return l
    
    def push_all_nodes(self, prev_layer): 
        for i in range(self.size):
            self.nodes[i].push_forward(prev_layer)
    
    def assign_node_values(self, l):
        for i in range(self.size):
            self.nodes[i].set_value(l[i])
        
    def print_layers_nodes_values(self):
        for i in range(self.size):
            self.nodes[i].print_value()
    
    def print_layers_nodes_weights(self):
        for i in range(self.size):
            print(self.nodes[i].weights)
        

class Mesh:
    
    layers = []
    input_layer = 0
    output = 0

    def __init__(self, input_layer, input_sizes):  
        self.size = len(input_sizes) 
        self.input_layer = input_layer
        self.layers = [Layer(input_sizes[i]) for i in range(len(input_sizes))]
        
    def full_push_forward(self):    
        self.layers[0].push_all_nodes(input_layer)   
        for i in range(1, len(self.layers)):
            self.layers[i].push_all_nodes(self.layers[i-1])
        self.output = self.layers[len(self.layers) - 1]

    def full_print(self):
        print("Input Layer: ")
        self.input_layer.print_layers_nodes_weights()
        self.input_layer.print_layers_nodes_values()
        print("")
        
        for i in range(len(self.layers)):
            print("Layer " + str(i) + ":")
            self.layers[i].print_layers_nodes_weights()
            self.layers[i].print_layers_nodes_values()
            print("")
    
    def print_layer(self, index):
        self.layers[index].print_layers_nodes_values()
        

expected_output = [.4]
computed_difference = 0        

input_values = [.5, .7, -.2, -.8, .1]
input_layer = Layer(len(input_values))
input_layer.assign_node_values(input_values)

m = Mesh(input_layer, [3, 5, 2, 1])
m.full_push_forward()
m.full_print()

computed_difference = [(expected_output[i] - m.output.nodes[i].value) for i in range(len(expected_output))]

print('Stats:')
print('Expected Output:\n', expected_output)
print('Mesh Output:')
print(end=' ')
m.print_layer(m.size - 1)
print('Computed Difference:\n', computed_difference)
