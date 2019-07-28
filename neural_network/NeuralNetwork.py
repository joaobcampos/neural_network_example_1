import numpy as np
from neural_network import Layer

class NeuralNetwork:
    
    def __init__(self, list_objective_functions, list_of_dimensions, learning_rate):
        self.input_dim = list_of_dimensions[0]
        self.learning_rate = learning_rate
        if (list_objective_functions.shape[0] != (list_of_dimensions.shape[0] - 1)):
            raise ValueError('The number of objective functions must be equal to the list of dimensions')
            
        list_of_layers = []
        for i in range(list_of_dimensions.shape[0] - 1):
            input_dimension  = list_of_dimensions[i]
            output_dimension = list_of_dimensions[i + 1]
            
            new_layer = Layer.Layer_3(input_dimension, output_dimension, list_objective_functions[i])
            list_of_layers.append(new_layer)
        
        self.layer_list = np.array(list_of_layers)
    
    def initialize_all_layers_randomly(self):
        for layer in self.layer_list:
            layer.random_initializer()
    
    def _check_input_dimensions(self, x):
        if (x.shape[0] != self.input_dim) or (x.shape[1] != 1):
            return False
        return True
    
    def classify_input(self, x):
        # Check if sample has the required shape for input
        if self._check_input_dimensions(x) == False:
            print('The input doesn\'t have the required dimensions')
            return x
        
        vec = x.copy()
        for layer in self.layer_list:
            vec = layer.calculate_layer_output(vec)
        
        return vec
    
    # Eq 13
    def grad_output_function(self, x, y):
        return(x - y)
    
    #Used for training only
    def backpropagation_algorithm(self, sample, target):
        list_of_outputs = []
        # Forward pass
        list_of_outputs.append(sample)
        vec = sample.copy()
        for layer in self.layer_list:
            vec = layer.calculate_layer_output(vec)
            list_of_outputs.append(vec)
            
        external_vector = self.grad_output_function(list_of_outputs[-1], target)
        
        # Backward pass: In this 
        for i in range(-1, -(len(list_of_outputs)), -1):
            x_out = list_of_outputs[i]
            x_in  = list_of_outputs[i-1]
            self.layer_list[i].update_weights(external_vector, x_out, x_in, self.learning_rate)
            external_vector = self.layer_list[i].return_derivative_vector(external_vector)
    
    #Used to propagate an input to the output
    def propagation(self, sample):
        x = sample.copy()
        for layer in self.layer_list:
            x = layer.calculate_layer_output(x)
        return x
