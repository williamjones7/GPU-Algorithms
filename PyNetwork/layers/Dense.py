import numpy as np

from PyNetwork import get_activation_function
from PyNetwork.layers import Layer
from PyNetwork.validation import check_layer
from PyNetwork import utils

import pyopencl.array as cl_array

class Dense(Layer):
    """ A fully connected layer

        Attributes
        ----------
        hidden_nodes : int
            The number of neurons in this layer
        g_name : str
            Name of the activation function
        built : bool
            Has the model been initialised
        output_shape : (n, ) tuple
            The shape of the output of this layer
        input_shape : (m, ) tuple
            The shape of the input of this layer
        W : (n, m) OpenCL array
            The weight matrix
        W_F : (n, m) OpenCL array
            The F-contiguous weight matrix
        b : (n, ) OpenCL array
            The bias unit

        Notes
        -----
        It is assumed that the input to this layer is a flattened vector. As such, when passing
        a multidimensional input, use a `flatten` layer first
    """
    def __init__(self, hidden_nodes, activation_function, l1=0.0, l2=0.0, trainable_mask=None, activation_kwargs=None, **kwargs):
        """ A fully connected layer

            Parameters
            ----------
            hidden_nodes : int
                The number of neurons in this layer
            activation_function : str
                The name of the activation function of this layer
            activation_kwargs : dict of str - :obj:, optional
                The keyword arguments for the activation function if it has hyper-parameters
        """
        self.hidden_nodes = hidden_nodes

        self.activation_function = activation_function
        self.activation_kwargs = {} if activation_kwargs is None else activation_kwargs

        self.output_shape = None
        self.input_shape = None
        
        self.W = None
        self.b = None
        
        if trainable_mask is not None:
            assert isinstance(trainable_mask, np.ndarray)
            self.trainable_mask = trainable_mask.astype(bool) 
        else:
            self.trainable_mask = None

        self.basis = None
        self.coeffs = None
        
        self.l1 = l1
        self.l2 = l2

        self.built = False

    def build(self, device_context, device_queue, previous_output_shape):
        """ Initialises the weight and bias units

            Parameters
            ----------
            previous_output_shape : 1 tuple of int
                The output shape of the previous layer. This will dictate the size of the weight matrix
        """
        self.context = device_context
        self.queue = device_queue

        self.output_shape = (self.hidden_nodes, )
        self.input_shape = previous_output_shape

        # Initialise the the weight with Glorot-Uniform, a uniform distribution over [-limit, limit],
        # where limit = sqrt(6 / (fan_in + fan_out)) (fan_in is the number of input units in the weight
        # tensor and fan_out is the number of output units).
        limit = np.sqrt(6 / (np.prod(self.input_shape) + np.prod(self.output_shape)))
        W = np.random.uniform(low=-limit, high=limit, size=(*self.output_shape, *previous_output_shape))

        self.W = cl_array.to_device(self.queue, W)
        self.W_F = cl_array.to_device(self.queue, np.asfortranarray(W)) 
        self.b = cl_array.zeros(self.queue, self.output_shape, dtype=np.float32)
        
        if self.trainable_mask is not None:
            assert self.trainable_mask.shape == self.W.shape, f"Trainable mask {self.trainable_mask.shape} must have the " \
                                                              f"same shape as the weight {self.W.shape}"                                                  

        self.built = True

        self.gpu_layer = utils.SingleLayer(self.context, self.queue)

    def predict(self, z, output_only=True, **kwargs):
        """ Returns the output of this layer

            Parameters
            ----------
            z : (d, m) np.array
                z is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of z is the index that inputs is accessed by.
            output_only : bool, optional
                If set to true, then this function will return only the prediction of the neural
                network. If set to false then this will return the outputs of the individual
                layers. Unless back propagation is being performed, this should be set to true.

            Returns
            -------
            (d, n) np.array
                The final output of the layer, post activation

            OR (if `output_only = False`)

            (d, n) np.array, (d, n) np.array
                The first np.array will store the output before it is passed through the activation
                function.
                The second np.array will store the output after it has passed through the
                activation function.
        """
        check_layer(self)

        # store as F-contiguous for transpose and accuracy
        W_F_T = cl_array.transpose(self.W_F)
        z_gpu = cl_array.to_device(self.queue, z)

        prod = self.gpu_layer.naiveMatmul(z_gpu, W_F_T)
        out_a = self.gpu_layer.addVector(prod, self.b)

        if output_only:
            return self.activation_function_(out_a)
        return out_a, self.activation_function_(out_a)

    def get_delta_backprop_(self, g_prime, new_delta, *args):
        """ Returns the delta for the previous layer, delta^{n-1}_{m,m}.

            Notes
            -----
            We want to return delta^{n-1} because the `sequential` class does not have access to the
            weights, W. But it does know the values of g'_{n-1} and delta^n, due to forward propagation
            and the backwards nature of the back propagation algorithm.

            Parameters
            ----------
            g_prime : (d, m) np.array
                Should be the derivative of the output of the previous layer, g'_{n-1}(a^{n-1}_{m,m})
            new_delta : (d, n) np.array
                The delta for this layer, delta^k_{m, m}

            Returns
            -------
            (d, m) np.array
                Returns delta of the previous layer, delta^{n-1}
        """
        check_layer(self)

        g_prime_gpu = cl_array.to_device(self.queue, g_prime)
        new_delta_gpu = cl_array.to_device(self.queue, new_delta)

        delta_result = g_prime_gpu * self.gpu_layer.naiveMatmul(new_delta_gpu, self.W)
        return delta_result

    def get_weight_grad_(self, delta, prev_z):
        """ Returns the associated partial S/partial W^n, that is
            the gradient with respect to the weight matrix in the kth layer

            Parameters
            ----------
            delta : (d, n) np.array
                In latex, this should be delta_k
            prev_z : (d, m) np.array
                This should be the output, post activation, of the previous layer (z_{n-1})

            Returns
            -------
            (n, ) np.array, (n, m) np.array
                The first array is the gradient for the bias unit
                The second array is the gradient for the weight matrix
        """
        check_layer(self)

        delta_gpu = cl_array.to_device(self.queue, delta)
        prev_z_gpu = cl_array.to_device(self.queue, prev_z)
        # store as F-contiguous for transpose and accuracy
        delta_gpu_F = cl_array.to_device(self.queue, np.asfortranarray(delta))
        delta_gpu_T = cl_array.transpose(delta_gpu_F)

        weight_grad = self.gpu_layer.naiveMatmul(delta_gpu_T, prev_z_gpu)
        bias_grad = self.gpu_layer.columnSumUp(delta_gpu)

        return bias_grad, weight_grad

    def update_parameters_(self, bias_updates, weight_updates):
        """ Perform an update to the weights by descending down the gradient

            Parameters
            ----------
            bias_updates : (n, ) np.array
                The gradients for the bias units
            weight_updates : (n, m) np.array
                The gradients for the weight matrix
        """
        check_layer(self)

        bias_updates_gpu = cl_array.to_device(self.queue, bias_updates)
        weight_updates_gpu = cl_array.to_device(self.queue, weight_updates)
        trainable_mask_gpu = cl_array.to_device(self.queue, self.trainable_mask)
 
        regularization_grad = cl_array.zeros(self.queue, self.W.shape, dtype=np.float32)
        if self.l1 > 0:
            regularization_grad += self.l1 * utils.gpu_layer.sign(self.W)
        if self.l2 > 0:
            regularization_grad += self.l2 * self.W

        if self.trainable_mask is None:
            self.W -= (weight_updates_gpu + regularization_grad)
        else:
            self.W -= (weight_updates_gpu + regularization_grad) * trainable_mask_gpu
            
        self.b -= bias_updates_gpu

    def get_weights(self):
        check_layer(self)
        return self.W, self.b

    def summary_(self):
        check_layer(self)
        return f'Dense {(self.hidden_nodes,)}', f'Output Shape {(None, *self.output_shape)}'

    @property
    def activation_function_(self):
        return get_activation_function(self.activation_function, **self.activation_kwargs)

    def __str__(self):
        return f'Dense: Output Shape {(None, *self.output_shape)}'
