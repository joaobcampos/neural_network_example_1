This article is reflects the process of understanding what are neural networks and how they work. It will be divided in 3 main parts:

<p>Part I - What are feed forward neural network.</p>
<p>Part II - Maths: the mathematical equations behind neural networks.</p>  
<p>Part III - How can the concepts behind it appear on code.</p>
<p> Annex </p>

# Part I

## Introduction

I have always seen neural networks as black boxes and recently, I tried to uncover what is behind them. The theme is vast as there are several types of networks and different architectures. In this post, I will only focus in the fully connected neural network, since it seems to be the oldest and simplest type of neural network to begin with [1], [2], [3] and [4]. The reason why I am writing this post is that until I could separate what a neural network is from its behavior, from what is it for, I would not understand them and also there is a huge amount of material in the web that is focused on the concept of perceptron [5], which is a valid approach, but, for me, it is obscures the fact that a neural network is basically this:

![nn_equations](/assets/images/sec_1.png) 

Now it is easy to see that <MATH>x&#8407;<sub>n</sub></MATH>, <MATH>x&#8407;<sub>n-1</sub></MATH> and <MATH>b&#8407;<sub>n</sub></MATH> are vectors, <b>W<sub>n</sub></b> are matrices and <MATH>f<sub>n</sub>(.)</MATH> is a function applied to each element of a vector. So basically, a neural network is a succession of the same operation recursively, in which the output of layer l-1: <MATH>x&#8407;<sub>l-1</sub></MATH> is the input to the next layer l and <MATH>f<sub>l</sub></MATH> can vary between layers.
The purpose of this post is to make a bridge between the mathematical concepts and an implementation given in this repo.


## Activation functions

Looking again at (1), it is possible to see that, without the functions <MATH>f<sub>n</sub>(.)</MATH>, a neural network would be a succession of linear operations. Usually, these functions, called activation functions, are used to make the network capture non linear relationships between variables and eventually limit each value of the output vector. The functions can have another interesting property: their derivatives may be written as functions of themselves. Three functions normally considered are the sigmoid (4), the softmax (5) and the hyperbolic tangent (6):

![activation_functions](/assets/images/sec_2_1.png)

Performing some calculations (see the Annex) we can see that their derivatives are:

![derivative_activation_functions](/assets/images/sec_2_2.png)


As you can see, you obtain the derivative of each function as a function of itself. You may have noticed the index in the softmax function. These activation functions will be applied to each element of a vector. While the sigmoid or the hyperbolic functions only take into consideration the vector itself, the softmax function takes into account all the elements of the vector. Although these are the activation functions considered in this post, we have many others: the ReLU [6], [7], the sinusoid, the softplus or the identity used in [8].

## Purpose of a network: Learn from data
The two previous sections described the elements of a neural network. The knots and bolts of the device. However, this tells us nothing about its purpose, and its purpose is to learn from data. For that we need a set of input/output pairs. The vectors <MATH>x&#8407;<sub>0</sub></MATH> are the inputs and the vectors <MATH>y&#8407;</MATH> are the corresponding expected outputs. At the end, we want that, given the vectors <MATH>x&#8407;<sub>0</sub></MATH>, the network outputs the corresponding vector <MATH>y&#8407;</MATH>. Take this claim with a pinch of salt, since if your neural network outputs exactly what you want, it may mean that it has learned the training set too well and may not be able to accurately predict the output of a samples <MATH>x&#8407;<sub>0</sub></MATH> that it has never seen.
But if we carry on with our reasoning, we want our vectors <MATH>x&#8407;<sub>N</sub></MATH>, remember, the last vectors of our network to be as close as possible to <MATH>y&#8407;</MATH>.
The objective function is the way we measure the similarity between those vectors. One of the most common objective functions is:

![nn_objective_function](/assets/images/sec_3.png)

Assuming vectors <MATH>x&#8407;<sub>N</sub></MATH> and <MATH>y&#8407;</MATH> have the same dimensions: L x 1. There are other functions, but for simplicity we will stick with this objective function for training and the sigmoid function for the activation, whose derivatives may not be written as functions of themselves only.

# Part II

## How to make a neural network learn: Back-propagation

The back-propagation method was firstly proposed in [9], [10], [11] and [12]. It cost me a good deal of ink and paper trying to understand it. Perhaps my efforts will be useful to others, hence this post, but nothing replaces the self work of making the calculations oneself.</br>
The first thing one should have in mind by reading this section is that our objective function depends implicitly on all the weight matrices and biases of all layers. So we will need the chain rule to obtain the gradients for the parameters (weight matrices and biases in all layers). The purpose of this section is to uncover a generic rule behind the back-propagation algorithm so we can actually implement it in code. 

### The chain rule
The chain rule is the most important factor in the back-propagation algorithm. It is this rule that allows the replacement of a layer's the activation function or a network's objective function and the back propagation while the back-propagation method remains the same. This means it allows this method to be modular.
Basically this rule tells that, if y=f(x) and x=g(z), then we will have:

![chain_rule](/assets/images/sec_4_1.png)

In the next two sections, we will use this rule to show how the back-propagation algorithm works. And, through equation (36), it is possible to observe an example of why the chain rule is important: it will allow modularity, by making the gradient of a single layer dependent only on its input and output vectors and an external vector (the backwards propagated error). Taking into account that only the output vector will depend on a layer's weights and biases, we could replace a layer's activation function without changing the rest of the network and the algorithm would work the same way. In the next subsections we will use this rule to calculate the gradient of the weights/biases for the last two layers.

### Last layer
We will first calculate the derivative of our objective function in order to the last layer weight matrix and bias (bear in mind <b>W<sub>n</sub></b> is a matrix, so it has two indexes).

![sec_4_2_1](/assets/images/sec_4_2_1.png)


Now, for equations (12) and (13), we can calculate each element in the chain rule as:

![sec_4_2_2](/assets/images/sec_4_2_2.png)

Thus, putting it all together, we have:

![sec_4_2_3](/assets/images/sec_4_2_3.png)

Equations (18) and (19) can be written in a generay way as:

![sec_4_2_4](/assets/images/sec_4_2_4.png)

where o represents elementwise multiplication. The conclusion of this subsection is that, considering a learning rate of &eta; we can write the update on the weight matrix and bias of the last layer as:

![sec_4_2_5](/assets/images/sec_4_2_5.png)

In this section we ended up with an expression that allows us to calculate the gradients of the last layer's parameters as functions of it inputs/output vectors. For future reference, in this case, we can see vector <MATH>e&#8407;<sub>N+1</sub></MATH>=<MATH>x&#8407;<sub>N</sub></MATH> - <MATH>y&#8407;</MATH> as the first error vector that will be backpropagated. Let's go one layer back.

### One layer before the last
As an effort to observe a general rule, we will calculate the derivative of gradients of the layer before the last.
Consider the derivative of just the element k in the sum:

![sec_4_3_1](/assets/images/sec_4_3_1.png)

The summation expresses the fact that all the terms that appear in the sum of the objective function will depend on all the terms of the matrix <b>W<sub>N-1</sub></b>. In the previous the i<sup>th</sup> element of the sum would depend only on the i<sup>th</sup> row of <b>W<sub>N</sub></b>.
Developing the derivatives we have in expressions (32) and (33):

![sec_4_3_2](/assets/images/sec_4_3_2.png)

Replacing equations (26)-(31) in the respective places, we have:

![sec_4_3_3](/assets/images/sec_4_3_3.png)

Comparing this expression with the previous one, we can already draw one conclusion. For layer m, the gradients of the objective function will always contain [<MATH>x&#8407;<sub>m</sub></MATH> o (<MATH>1&#8407;</MATH>-<MATH>x&#8407;<sub>m</sub></MATH>)]<MATH>x&#8407;<sub>m-1</sub></MATH><sup>T</sup> for the weights and [<MATH>x&#8407;<sub>m</sub></MATH> o (<MATH>1&#8407;</MATH>-<MATH>x&#8407;<sub>m</sub></MATH>)] for the biases. In other words, if, for each layer, we know the vectors at the input and at the output, we already know something about the gradient concerning of the objective function regarding the parameters of that layer. 
We will explore the vector:

![sec_4_3_4](/assets/images/sec_4_3_4.png)

If you do the maths, you will see that this vector can simply be written as:
In other words, this will will be the error vector that will be passed to layer N-1 by layer N in the propagation pass. Thus, the final gradient can be written as (remember <MATH>e&#8407;<sub>N+1</sub></MATH>=<MATH>x&#8407;<sub>N</sub></MATH> - <MATH>y&#8407;</MATH>):

![sec_4_3_5](/assets/images/sec_4_3_5.png)

![sec_4_3_6](/assets/images/sec_4_3_6.png)

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

![sec_5_1](/assets/images/sec_5_1.png)

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
The network is a collection of layers: ``` self.layer_list ```. In order for the network to be created, in this case, we give it a learning rate, a list of dimensions for each layer and a list of objective functions (one for each layer). The list of dimensions contain the input dimension for each layer. The input dimension for layer m is the output dimension for layer m-1.

Has the backpropagation algorithm is the resposability of the network, has all layers' weights and biases must be updated. The propagation algorithm is:
```
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
```
As you can see, in the forward pass, the outputs of each layer are calculated, and stored in ```list_of_outputs```, but for each layer, the weights and biases are untouched. In the backward pass we calculate the <MATH>e&#8407;<sub>N</sub></MATH> vector in the line ``` external_vector = self.grad_output_function(list_of_outputs[-1], target) ``` which will correspond, in this case, to the difference between the last vector and the target, and, in the for cycle, we use the ``` update_weights ```, which will implement equations (22) and (23).

### Results
We train the network in batches. Firstly, in the notebook training.ipynb, the network is created:
```
#Creation of the network
dimensions = np.array([4, 8, 5, 3])
functions = np.array([ActivationFunction.sigmoid(), ActivationFunction.sigmoid(), ActivationFunction.sigmoid()])
ff_nn = NeuralNetwork.NeuralNetwork(functions, dimensions, 0.01)
```
Then, the data is loaded:
```
#Load training and testing data
DataLoader = LoadData.LoadData()
X_train, X_test, y_train, y_test = DataLoader.partition_dataset()
```
And finally, the network is trained:
```
epochs = 10000
batch_size = 5
training_error = np.zeros(shape=(epochs, X_train.shape[2])) # Container of the training error
testing_error = np.zeros(shape=(epochs, X_test.shape[2])) #Container for the testing error
for k in range(epochs):
    # Select withing the training set the samples that will be given for the batch
    indexes = np.random.randint(0, high=99, size=(batch_size,))
    X_train_ = X_train[:,:,indexes].copy()
    y_train_ = y_train[:,:,indexes].copy()
    
    #Bear in mind this is not the most efficient way to implement a nn. 
    #It was used merely to illustrate the mathematical concepts
    #At this point the weights and biases of the network were updated for this specific
    #sample
    for i in range(X_train_.shape[2]):
        X_in = X_train_[:,:,i]
        y_in = y_train_[:,:,i]
        ff_nn.backpropagation_algorithm(X_in, y_in)
        
    # It is interesting to see how the error evolves with the time
    # At this point we run the network as is for the test set
    training_error[k, :] = evaluate_error_datasets(X_train, y_train, ff_nn)
    testing_error[k, :]  = evaluate_error_datasets(X_test, y_test, ff_nn)
```
In this case, for 10000 times, 5 random train vectors are picked along with their targets:
```
X_train_ = X_train[:,:,indexes].copy()
y_train_ = y_train[:,:,indexes].copy()
```
Then, for each vector, the backpropagation algorithm is run:
```
for i in range(X_train_.shape[2]):
        X_in = X_train_[:,:,i]
        y_in = y_train_[:,:,i]
        ff_nn.backpropagation_algorithm(X_in, y_in)
```

In the lines:
```
training_error[k, :] = evaluate_error_datasets(X_train, y_train, ff_nn)
testing_error[k, :]  = evaluate_error_datasets(X_test, y_test, ff_nn)
```
The network's parameters calculated in a given iteration are applied to the train and test sets. And their error stored. The function used is:
```
def evaluate_error_datasets(X_dataset, y_dataset, network):
    error_list = np.zeros((X_dataset.shape[2], ))
    for i in range(X_dataset.shape[2]):
        X_in = X_dataset[:,:,i]
        y    = y_dataset[:,:,i]
        y_out = ff_nn.propagation(X_in)
        diff = y - y_out
        error = np.dot(diff.T, diff)
        error_list[i] = error[0,0]
    return error_list
```
The following plot shows how the error average decreases with each one of the 10000 iterations:

![sec_5](/assets/images/sec_5.png)


# References

[1] Frank Rosenblatt. The perceptron: a probabilistic model for information storage and organization in the brain.
Psychological review, 65(6):386, 1958.<br>
[2] F. Rosenblatt. Principles of Neurodynamics. Spartan, New York, 1962.<br>
[3] B. Widrow and M.E. Hoff. Associative storage and retrieval of digital information in networks of adaptive neurons.
Biological Prototypes and Synthetic Systems, 1:160, 1962.<br>
[4] J. Schmidhuber. Deep learning in neural networks: An overview. Neural Networks, 61:85–117, 2015. Published
online 2014; based on TR arXiv:1404.7828 [cs.NE].<br>
[5] Frank Rosenblatt. Principles of Neurodynamics. Spartan Books, 1962.<br>
[6] Vinod Nair and Geoffrey E. Hinton. Rectified linear units improve restricted boltzmann machines. In
Proceedings of the 27th International Conference on International Conference on Machine Learning, ICML’10, pages 807–814,
USA, 2010. Omnipress.<br>
[7] K. He, X. Zhang, S. Ren, and J. Sun.  Delving deep into rectifiers:  Surpassing human-level performance on imagenet classification. In 2015 IEEE International Conference on Computer Vision (ICCV), pages 1026–1034, Dec 2015.<br>
[8] Ashmore S.C. Gashler M.S. Training deep fourier neural networks to fit time-series data. In Proceedings of the
27th International Conference on International Conference on Machine Learning, International Conference on
Intelligent Computing, pages 48–55. Springer, Cham, 2014.<br>
[9] S. E. Dreyfus.  The computational solution of optimal control problems with time lag.
IEEE Transactions on Automatic Control, 18(4):383–385, 1973.<br>
[10] Seppo Linnainmaa. Taylor expansion of the accumulated rounding error. BIT Numerical Mathematics, 16(2):146–160, 1976.
[11] S. Linnainmaa. The representation of the cumulative rounding error of an algorithm as a Taylor expansion of the
local rounding errors. Master’s thesis, Univ. Helsinki, 1970.<br>
[12] David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. Learning representations by back-propagating
errors. Nature 323:533–, October 1986.<br>
[13]  R. A. Fisher. The use of multiple measurements in taxonomic problems. Annual Eugenics, 7:179–188, 1936.<br>
[14] Travis Oliphant. NumPy: A guide to NumPy. USA: Trelgol Publishing, 2006–. [Online; accessed <today>].<br>
[15] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation
applied to handwritten zip code recognition. Neural Comput., 1(4):541–551, December 1989.<br>
[16] Yelong Shen, Xiaodong He, Jianfeng Gao, Li Deng and Gregoire Mesnil. Learning semantic representations using convolutional neural networks for web search. WWW 2014, April 2014<br>
[17] François Chollet et al. Keras. https://keras.io, 2015.<br>
[18] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy
Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael
Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Rajat
Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever,
Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden,
Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on
heterogeneous systems, 2015. Software available from tensorflow.org.<br>
[19]  Sebastian Ruder. An overview of gradient descent optimization algorithms. CoRR, abs/1609.04747, 2016.<br>

# Annex

![sec_7_1](/assets/images/sec_7_1.png)


![sec_7_2](/assets/images/sec_7_2.png)


![sec_7_3](/assets/images/sec_7_3.png)
