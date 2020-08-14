import math
import random

class Node:
    def __init__(self, value):
        self.value = value

class WNode:

    def __init__(self, value, prev_, next_):
        self.value = 0
        self.prev = prev_
        self.next = next_

    def setValueRandom(self):
        self.value = (random.random() * 2) - 1

    def passForwardValue(self):
        if(self.prev != None):
            self.next.value += self.prev.value * self.value


class Layer:
    def __init__(self, size):
        self.size = size
        self.nodes = [Node(0) for _ in range(size)]

    def printLayer(self):
        for n in self.nodes:
            print(n.value, end='  ')
        print()

    def setNodeValues_list(self, nVals):
        for i, n in enumerate(self.nodes):
            n.value = nVals[i]
     
class WLayer:
    def __init__(self, prev_layer, next_layer):
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        
        self.weights = []

        for pNode in self.prev_layer.nodes:
            for nNode in self.next_layer.nodes:
                self.weights.append(WNode(0, pNode, nNode))
                
        for s in self.weights:
            s.setValueRandom()
    
    def printLayer(self):
        for w in self.weights:
            print(w.value)
        print()
    


class Net:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.layers = [Layer(self.layer_sizes[i]) for i in range(len(self.layer_sizes))]
        self.weighted_layers = []
        self.createWeightLayers()
              
              
    def setInputValues(self, arr):
        self.layers[0].setNodeValues_list(arr)
        
    def createWeightLayers(self):
        self.weighted_layers = []
        for l in range(len(self.layers) - 1):
            self.weighted_layers.append(WLayer(self.layers[l], self.layers[l + 1]))


    def pushForward(self):
        for w in self.weighted_layers:    
            for wN in w.weights:
                wN.passForwardValue()
            # Loop through current layer and sigmoid output
            
    def outputs(self):
        return_arr = []
        for n in self.layers[-1].nodes:
            return_arr.append(n.value)
        return return_arr


    def sig(self, n):
        return 1 / (1 + pow(math.e, -n))

    def printLayers(self):
        for i in range(len(self.weighted_layers)):
            self.layers[i].printLayer()
            self.weighted_layers[i].printLayer()
        self.layers[-1].printLayer()
            


layer_sizes = [4, 3, 1]
n = Net(layer_sizes)
n.setInputValues([.4, .6, .1, .7])

n.pushForward()
n.printLayers()

print(n.outputs())
