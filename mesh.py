
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

    def init_weights_random(self, input_layer):
        self.weights = np.random.uniform(low = -1, high = 1, size = len(input_layer.nodes))
    
    def push_forward(self, input_layer):
        
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

    def init_layer_node_weights_random(self, prev_layer):
        for i in range(self.size):
            self.nodes[i].init_weights_random(prev_layer)
    
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

    def __init__(self, input_sizes):  
        self.size = len(input_sizes) 
        self.input_sizes = input_sizes
        self.layers = [Layer(input_sizes[i]) for i in range(len(input_sizes))]
        
    def compute(self, input_layer, random): 
        self.set_input_layer(input_layer)
        if(random == 1):
            self.init_all_layer_node_weights_random()
        self.full_push_forward()
        
    
    def set_input_layer(self, input_layer):
        self.input_layer = input_layer

    def full_push_forward(self):    
        self.layers[0].push_all_nodes(self.input_layer)   
        for i in range(1, len(self.layers)):
            self.layers[i].push_all_nodes(self.layers[i-1])
        self.output = self.layers[len(self.layers) - 1]

    def init_all_layer_node_weights_random(self):    
        self.layers[0].init_layer_node_weights_random(self.input_layer)   
        for i in range(1, len(self.layers)):
            self.layers[i].init_layer_node_weights_random(self.layers[i-1])
        

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
        merge_dec = np.random.randint(low=0, high=2, size = len(partner1.weights))
        child_weights = []
        for ind in range(len(merge_dec)):
            child_weights.append(partner2.weights[ind] if merge_dec[ind] == 1 else partner1.weights[ind])
        
        child_node = Node(0)
        child_node.pass_weights_in(child_weights)

        return child_node
    
# Loopable

input_values = [.5, .7, -.2, -.8, .1]
input_layer = Layer(len(input_values))
input_layer.assign_node_values(input_values)

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
