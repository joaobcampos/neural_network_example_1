import numpy as np

class ActivationFunction():
 
    def function_value(self, x):
        pass
    
    def gradient_weights(self, external_vector, x_out, x_in):
        pass
    
    def gradient_bias(self, external_vector, x_out):
        pass

#Eq 3
class sigmoid(ActivationFunction):
    
    def __init__(self):
        super().__init__()
    
    def function_value(self, x):
        return (1/(1 + np.exp(-x)))
    
    #equation 35 of the article
    def gradient_weights(self, external_vector, x_out, x_in):
        v = external_vector * (1-x_out) * x_out
        return np.dot(v, x_in.T)
    
    def gradient_bias(self, external_vector, x_out):
        return external_vector * (1-x_out) * x_out
    
#Eq 4
class hyperbolic_tangent(ActivationFunction):
    
    def __init__(self):
        super().__init__()
    
    def function_value(self, x):
        return np.tanh(x)
    
    def gradient_weights(self, external_vector, x_out, x_in):
        v = external_vector * (1 - (x_out * x_out))
        return np.dot(v, x_in.T)
    
    def gradient_bias(self, external_vector, x_out):
        return external_vector * (1 - (x_out * x_out))

class re_lu(ActivationFunction):
    
    def __init__(self):
        super().__init__()
    
    def function_value(self, x):
        x_ = x.copy()
        x_[x_ < 0] = 0
        return x_
    
    def gradient_weights(self, external_vector, x_out, x_in):
        x_ = x_out.copy()
        x_[x_ < 0] = 0
        v = external_vector * x_
        return np.dot(v, x_in.T)
    
    def gradient_bias(self, external_vector, x_out):
        x_ = x_out.copy()
        x_[x_ < 0] = 0
        return external_vector * x_
    
class experiment(ActivationFunction):
    def __init__(self):
        super().__init__()
    
    def function_value(self, x):
        return 2 * x
    
    def gradient_weights(self, external_vector, x_out, x_in):
        v = x_out * (1-x_in)
        return np.dot(v, external_vector.T)
    
    def gradient_bias(self, external_vector, x_out):
        return external_vector * x_out
