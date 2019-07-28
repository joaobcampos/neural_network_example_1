import numpy as np

class Layer_3:
    
    def __init__(self, input_dimension, output_dimension, activation_function):
        #Weights/bias
        self.weights = np.random.rand(output_dimension, input_dimension)
        self.bias    = np.random.rand(output_dimension, 1)
        
        self.activation_function = activation_function
    
    def random_initializer(self):
        n_rows, n_cols = self.weights.shape
        self.weights = np.random.rand(n_rows, n_cols)
        self.bias = np.random.rand(n_rows,1)
    
    #Equation 1 in the paper
    def calculate_layer_output(self, input_vector):
        linear_vector = np.dot(self.weights, input_vector) + self.bias
        return self.activation_function.function_value(linear_vector)
        
    #Equations 21/22   
    def update_weights(self, external_vector, output_vector, input_vector, learning_rate):
        #Eq 19
        gradient_weights = self.activation_function.gradient_weights(external_vector,output_vector, input_vector)
        #Eq 20
        gradient_bias    = self.activation_function.gradient_bias(external_vector, output_vector)
        self.weights = self.weights - learning_rate * gradient_weights #equation 21
        self.bias    = self.bias    - learning_rate * gradient_bias #Equation 22
    
    #Eq 34
    def return_derivative_vector(self, external_vector):
        resulting_vector = np.dot(self.weights.T, external_vector) #equation 36 of the article
        return resulting_vector
   
