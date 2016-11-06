#!/usr/bin/env python
from SimpleNN import SimpleNN

def main():

    layer_dims = [
        3,  # 3 inputs
        3,  # single hidden layer has 3 nodes
        1   # 1 output
    ]
    nn = SimpleNN(layer_dims=layer_dims)
    nn.train([],[])
    # http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/


if __name__ == "__main__":
    main()
