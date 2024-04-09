"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
        
        print('shape of weights', [ v.shape for k, v in self.params.items() if 'W' in k])
        print('shape of all inputs', self.input_size, self.hidden_sizes, self.output_size, self.num_layers)

            # TODO: You may set parameters for Adam optimizer here
            # I will leave it blank, as the update function will handle this already
            
    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        return np.dot(X, W) + b
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, b: np.ndarray, de_dz: np.ndarray, reg, N) -> np.ndarray:
        """Gradient of linear layer
            z = WX + b
            returns de_dw, de_db, de_dx
        """
        # TODO: implement me
        # Seems like reg and N are used for regularization, but they're not important right now
        # We basically use the chain rule to calculate the gradient of the loss with respect to the weights, bias, and input
            # So basically for each variable, we need to multiply the other values not included, because we're calculating the gradient
        # For the weight gradient, we always do dotproduct of x and y. Here the y is z, or the derivative is de_dz
        dw = np.dot(X.T, de_dz)
        # For the bias gradient, we always calculate the sum of the y
        db = np.sum(de_dz, axis = 0)
        # We can calculate the gradient of the loss with respect to the input (z), 
        dx = np.dot(de_dz, W.T)

        return dw, db, dx

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(0, X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        return np.where(X > 0, 1, 0)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        return 1 / (1 + np.exp(-x))

    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        # TODO implement this
        # return np.exp(-X) / (1 + np.exp(-X))**2
        return self.sigmoid(X) * (1 - self.sigmoid(X))

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return np.mean( (y - p) ** 2 )
    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return 2 * (p - y) / len(y)
    
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return self.sigmoid_grad(p) * self.mse_grad(y, p)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.

        print(self.hidden_sizes)
        print(self.num_layers)

        for i in range(1, self.num_layers):
            W = self.params["W" + str(i)]
            b = self.params["b" + str(i)]

            print(W.shape, b.shape)
            print(self.params.keys())

            linear_output = self.linear(W, X, b)
            relu_output = self.relu(linear_output)

            self.outputs["linear_outputs_" + str(i)] = linear_output
            self.outputs["relu_outputs_" + str(i)] = relu_output

            # W = self.params["W" + str(i)]
            # b = self.params["b" + str(i)]

            # self.outputs["linear_outputs_" + str(i)] = self.linear(W, X, b)
            # self.outputs["relu_outputs_" + str(i)] = self.relu(self.outputs["linear_outputs_" + str(i)])
            print('one layer done')
            # self.outputs["linear_outputs_2_" + str(i)] = self.linear(W, self.outputs["relu_outputs_" + str(i)], b)
            # self.outputs["sigmoid_outputs_" + str(i)] = self.sigmoid(self.outputs["linear_outputs_2_" + str(i)])

        # Pass everyhting through last linear layer, using the final weights and biases (which is why we add num_layers to the key, as it is the final layer)
        # print(self.outputs.keys())
        # print(self.num_layers)
        # print(self.params.keys())
        print(self.outputs.keys())
        self.outputs["linear_outputs_outputs"] = self.linear(self.params["W" + str(self.num_layers)], self.outputs["relu_outputs_" + str(self.num_layers - 1)], self.params["b" + str(self.num_layers)])
        self.outputs["output_layer"] = self.sigmoid(self.outputs["linear_outputs_outputs"])
        
        return self.outputs["output_layer"]

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.

        # To calculate loss, we need to calculate the mean squared error

        self.loss = self.mse(y, self.outputs["output_layer"])

        # To calculate the loss gradient, we need to calculate the mean squared error gradient, and 
        # multiply this with the output activation layer's grad as well, which in this case is sigmoid
        dz = self.mse_sigmoid_grad(y, self.outputs["output_layer"])

        # Backpropagation through the layers
        for i in range(self.num_layers - 1, 0, -1):
            # Calculate the gradients for the current layer
            print(self.params["W" + str(i)].shape)
            print(dz.shape)
            dw, db, dx = self.linear_grad(
                self.params["W" + str(i)],
                self.outputs["linear_outputs_" + str(i)],
                self.params["b" + str(i)],
                dz,
                reg=0,
                N=len(y),
            )
            # Store the gradients in self.gradients
            self.gradients["W" + str(i)] = dw
            self.gradients["b" + str(i)] = db

            # Update de_dz for the next layer
            if i > 1:  # Exclude input layer
                dz = dx * self.relu_grad(self.outputs["linear_outputs_" + str(i - 1)])

        return self.loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt
        self.t = 0

        if opt == "SGD":
            for key, val in self.gradients.items():
                self.params[key] -= lr * val
        elif opt == "Adam":
            for param_key in self.gradients.keys():
                self.adam(lr, b1, b2, eps, param_key)


    # Implement Adam optimizer
    def adam(self, lr, b1, b2, eps, param_key):
        # For every parameter in gradients (weight, bias, whatever), we need to calculate the first and second moments

        # Initialize the first and second moments
        self.t += 1
        self.m[param_key] = np.zeros_like(self.params[param_key])
        self.v[param_key] = np.zeros_like(self.params[param_key])

        # Update the first and second moments
        self.m[param_key] = b1 * self.m[param_key] + (1 - b1) * self.gradients[param_key]
        self.v[param_key] = b2 * self.v[param_key] + (1 - b2) * self.gradients[param_key]**2

        # Bias correction
        m_hat = self.m[param_key] / (1 - b1**self.t)
        v_hat = self.v[param_key] / (1 - b2**self.t)

        # Update the parameters
        self.params[param_key] -= lr * m_hat / (np.sqrt(v_hat) + eps)