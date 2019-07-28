This article is reflects the process of understanding what are neural networks and how they work. It will be divided in 3 main parts:

Part I - What are feed forward neural network.\
Part II - Maths: the mathematical equations behind neural networks.  
Part III - How can the concepts behind it appear on code.

# Part I

## Introduction

I have always seen neural networks as black boxes and recently, I tried to uncover what is behind them. The theme is vast as there are several types of networks and different architectures. In this post, I will only focus in the fully connected neural network, since it seems to be the oldest and simpler type of neural network to begin with [1], [2], [3] and [4]. The reason why I am writing this post is that basically a huge amount of material in the web focus on the concept of perceptron [5]. Although a valid approach, this is obscures the fact that a neural network is basically this:

<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_1.png" alt="drawing" width="600" class="center"/>

Now it is easy to see that <MATH>x&#8407;<sub>n</sub></MATH>, <MATH>x&#8407;<sub>n-1</sub></MATH> and <MATH>b&#8407;<sub>n</sub></MATH> are vectors, <b>W<sub>n</sub></b> are matrices and <MATH>f<sub>n</sub>(.)</MATH> is a function applied to each element of a vector. So basically, a neural network is a succession of the same operation recursively, in which the output of layer l-1: <MATH>x&#8407;<sub>l-1</sub></MATH> is the input to the next layer l and <MATH>f<sub>l</sub></MATH> can vary between layers.
The purpose of this post is to make a bridge between the mathematical concepts and an implementation given in this repo.


## Activation functions

Looking again at (1), it is possible to see that, without the functions <MATH>f<sub>n</sub>(.)</MATH>, a neural network would be a succession of linear operations. Usually, these functions, called activation functions, are used to make the network capture non linear relationships between variables and eventually limit each value of the output vector. The functions usually used have another interesting property: their derivatives can be written as functions of themselves. Three functions normally considered are the sigmoid (4), the softmax (5) and the hyperbolic tangent (6):

<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_2_1.png" alt="drawing" width="600" class="center"/>
Performing some calculations we can see that their derivatives are:

<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_2_2.png" alt="drawing" width="600" class="center"/>

As you can see, you can obtain the derivative of each function as a function of itself (if you want to know how the derivatives were obtained, you can check the annex). You may have noticed the index in the softmax function. These activation functions will be applied to each element of a vector. While the sigmoid or the hyperbolic functions only take into consideration the vector itself, the softmax function takes into account all the elements of the vector. Although these are the activation functions considered in this post, we have many others: the ReLU [6], [7], the sinusoid, the softplus or the identity used in [8].

## How to train a network: Objective function
The two previous sections described the elements of a neural network. The knots and bolts of the device. The most obvious one is to learn. For that we need a set of input/output pairs. The vectors <MATH>x&#8407;<sub>0</sub></MATH> are the inputs and the vectors <MATH>y&#8407;</MATH> are the corresponding expected outputs. At the end, we want that, given the vectors <MATH>x&#8407;<sub>0</sub></MATH>, the network outputs the corresponding vector <MATH>y&#8407;</MATH>. Take this claim with a pinch of salt, since if your neural network outputs exactly what you want, it may mean that it has learned the training set too well and may not be able to accurately predict the output of a samples <MATH>x&#8407;<sub>0</sub></MATH> that it has never seen.
But if we carry on with our reasoning, we want our vectors <MATH>x&#8407;<sub>N</sub></MATH>, remember, the last vectors of our network to be as close as possible to <MATH>y&#8407;</MATH>.
The objective function is the way we measure the similarity between those vectors. One of the most common objective functions is:

<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_3.png" alt="drawing" width="600" class="center"/>

Assuming vectors <MATH>x&#8407;<sub>N</sub></MATH> and <MATH>y&#8407;</MATH> have the same dimensions: L x 1. There are other functions, but for simplicity we will stick with this objective function for training and the sigmoid function for the activation, whose derivatives may not be written as functions of themselves only.

# Part II

## How to train a network: Back-propagation

The back-propagation method was firstly proposed in [9], [10], [11] and [12]. It cost me a good deal of ink and paper trying to understand it. Perhaps my efforts will be useful to others, hence this post, but nothing replaces the self work of making the calculations oneself.\
The first thing one should have in mind by reading this section is that our objective function depends implicitly on all the weight matrices and biases of all layers. So we will need the chain rule to obtain the gradients for the parameters (weight matrices and biases in all layers). The purpose of this section is trying to uncover a generic rule behind the back-propagation algorithm so we can actually implement it in code. 

### The chain rule
The chain rule is the most important factor in the back-propagation algorithm. It is this rule that allows the replacement of a layer's the activation function or a network's objective function and the back propagation while the back-propagation method remains the same. This means it allows this method to be modular.
Basically this rule tells that, if y=f(x) and x=g(z), then we will have:

<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_4_1.png" alt="drawing" width="600" class="center"/>

In the next two sections, we will use this rule to show how the back-propagation algorithm works. And, in (36) it is possible to observe an example of why the chain rule is important: it will allow modularity, by making the gradient of a single layer dependent only on its input and output vectors and an external vector (the propagated error). Taking into account that only the output vector will depend on a layer's weights and coefficients, we can replace a layer's activation function without changing the rest of the network that the algorithm would work the same way. In the next subsections we will use this rule to calculate the gradient of the weights/biases for the last two layers.

### Last layer
We will first calculate the derivative of our objective function in order to the last layer weight matrix and bias (bear in mind <b>W<sub>n</sub></b> is a matrix, so it has two indexes).

<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_4_2_1.png" alt="drawing" width="600" class="center"/>


Now, looking at (13) and (14), we can calculate each element:

<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_4_2_2.png" alt="drawing" width="600" class="center"/>

Thus, putting it all together, we have:

<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_4_2_3.png" alt="drawing" width="600" class="center"/>

If we equation a global formula, we can rewrite (18) and (19) as:

<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_4_2_4.png" alt="drawing" width="600" class="center"/>

where o represents elementwise multiplication.
The conclusion of this subsection is that, considering a learning rate of $\eta$ we can write the update on the weight matrix and bias of the last layer as:

<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_4_2_5.png" alt="drawing" width="600" class="center"/>


In this section we ended up with an expression that allows us to calculate the gradients of the last layer's parameters as functions of it inputs/output vectors. Let's go one layer back.

### One layer before the last
As an effort to observe a general rule, we will calculate the derivative of gradients of the layer before the last.
Consider the derivative of just the element k in the sum:

<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_4_3_1.png" alt="drawing" width="600" class="center"/>

The summation expresses the fact that all the terms that appear in the sum of the objective function will depend on all the terms of the matrix <b>W<sub>N-1</sub></b>. In the previous the i<sup>th</sup> element of the sum would depend only on the i<sup>th</sup> row of <b>W<sub>N</sub></b>.
Developing the derivatives we have in expressions (32) and (33):


<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_4_3_2.png" alt="drawing" width="600" class="center"/>

Replacing equations (26)-(31) in the respective places, we have:

<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_4_3_3.png" alt="drawing" width="900" class="center"/>



Comparing this expression with the previous one, we can already draw one conclusion. For layer m, the gradients of the objective function will always contain [<MATH>x&#8407;<sub>m</sub></MATH> o (<MATH>1&#8407;</MATH>-<MATH>x&#8407;<sub>m</sub></MATH>)]<MATH>x&#8407;<sub>m-1</sub></MATH><sup>T</sup> for the weights and [<MATH>x&#8407;<sub>m</sub></MATH> o (<MATH>1&#8407;</MATH>-<MATH>x&#8407;<sub>m</sub></MATH>)] for the biases. In other words, if, for each layer, we know the vectors at the input and at the output, we already know something about the gradient concerning of the objective function regarding the parameters of that layer. 
We will explore the vector:

<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_4_3_4.png" alt="drawing" width="600" class="center"/>

If you do the maths, you will see that this vector can simply be written as:
In other words, this will will be the error vector that will be passed to layer N-1 by layer N in the propagation pass. Thus, the final gradient can be written as:

<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_4_3_5.png" alt="drawing" width="600" class="center"/>

<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_4_3_6.png" alt="drawing" width="600" class="center"/>


The best hint we can take from this section is that the gradients of the objective function in order to a specific layer will depend on the current input/output vectors at that layer and from an error vector passed on by the previous layer, hence the modularity. If you followed the calculations you will realize that is a consequence of the chain rule.

# Part III
## Filling the gap
This post's purpose is to present a simple point: The conceptual unit behind a Neural Network is the layer. In the section Material the data used will be presented. In the code section, there will be the correspondences between the lines of code and the equations presented in part II and in the conclusion section, the network's behavior will be presented.

### Material
The dataset used was the iris dataset [13] and the code was developped using the library numpy [14]. Is is important to note that this dataset has 4 features:

* sepal length in cm,
* sepal width in cm,
* petal length in cm,
* petal width in cm.

So we will have a table of variables whose dimension is 150 x 4 being the target the 5<sup>th</sup> column. In our example, the <MATH>x&#8407;<sub>0</sub></MATH> vectors will have a dimension of 4 x 1. That's why I reshaped the train and test sets will have a dimension of: number of features x number of vector columns (always 1) x number of samples. Beware that for simpler problems, the dimensions may not be a big issue, but when you tackle convolution neural networks [15], [16] in a framework such as keras/tensorflow [17], [18], understanding the dimensions of your inputs/outputs may not be a trivial issue.
Concerning classes, each class will be:
* Iris Setosa
* Iris Versicolour
* Iris Virginica

It is easy to see that we cannot represent the class using a single number, as it doesn't make sense to say that between numbers n<sub>1</sub> and n<sub>2</sub> the class is Setosa, between n<sub>2</sub> and n<sub>3</sub> the class is Versicolour and outside this interval is Virginica. It makes more sense in this case to build a 3 x 1 vector whose 3 elements mean:

* is Setosa: True/False
* is Versicolour: True/False
* is Virginica: True/False

So, if, for example, our example is an Iris Setosa, we want our network to give:

<img src="https://github.com/joaobcampos/neural_network_example_1/blob/master/images/sec_5_1.png" alt="drawing" width="600" class="center"/>

### The code
The code is structured to show the modular concepts behind a neural network. We have the packages:
* ActivationFunction: where we define the possible activation functions for a layer and its gradients.
* Layer: where we define the operations that a layer must perform: calculate its outputs using its activation function (\textbf{calculate\_layer\_output}) and update its weights/biases (\textbf{update\_weights}) using the gradients of its activation function and the propagation error (\textbf{return\_derivative\_vector}).
* NeuralNetwork: where we define the operations a network must have. Its main operations are: providing the implementation of the back-propagation algorithm \textbf{backpropagation\_algorithm} and propagating an input vector to the output \textbf{propagate}.

#### The activation function
Every activation function must provide the following:
```
class ActivationFunction():
 
    def function_value(self, x):
        pass
    
    def gradient_weights(self, external_vector, x_out, x_in):
        pass
    
    def gradient_bias(self, external_vector, x_out):
        pass
```
The ```function_value``` will represent the layer's output given and input vector <MATH>x&#8407;</MATH>. The ```gradient_weights``` and ```gradient_bias``` will calculate the gradients to update the layer's wights and biases. These last two functions will correspond to equations (20), (21), (32) and (33) in part II.

For layer m we will have:

* ```x_out```:  <MATH>x&#8407;<sub>m</sub></MATH>
* ```x_in```:  <MATH>x&#8407;<sub>m-1</sub></MATH>
* ```external_vector```:  <MATH>e&#8407;<sub>m</sub></MATH>

For the sigmoid function (4), the implementation is:
```
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
 ```
#### The Layer
The layer must contain its own activation function, its own weights matrix and bias vector. In the file Layer.py it is possible to see:
```
def __init__(self, input_dimension, output_dimension, activation_function):
        #Weights/bias
        self.weights = np.random.rand(output_dimension, input_dimension)
        self.bias    = np.random.rand(output_dimension, 1)
        
        self.activation_function = activation_function
```

In this case, the parameters are being initialized randomly. It it the layer's responsability to:
* Calculate its own output:
```
 def calculate_layer_output(self, input_vector):
        linear_vector = np.dot(self.weights, input_vector) + self.bias
        return self.activation_function.function_value(linear_vector)
```
* Update its weights and biases:
```
def update_weights(self, external_vector, output_vector, input_vector, learning_rate):
        #Eq 19
        gradient_weights = self.activation_function.gradient_weights(external_vector,output_vector, input_vector)
        #Eq 20
        gradient_bias    = self.activation_function.gradient_bias(external_vector, output_vector)
        self.weights = self.weights - learning_rate * gradient_weights #equation 21
        self.bias    = self.bias    - learning_rate * gradient_bias #Equation 22
 ```
 * Return the error vector for the previous layer (<MATH>e&#8407;<sub>m</sub></MATH>):
 ```
 def return_derivative_vector(self, external_vector):
        resulting_vector = np.dot(self.weights.T, external_vector) #equation 36 of the article
        return resulting_vector
 ```
 #### The network
 The last component of a neural network is the network itself, which will be simply a collection of layers:
 ```
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
        ```
