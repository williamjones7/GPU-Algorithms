import numpy as np

from PyNetwork import ActivationFunctions
from PyNetwork.layers import Layer
from PyNetwork.validation import check_layer
from PyNetwork.utils import NaiveMatMul, ArrayFunctions, GPUTranspose

import pyopencl as cl
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
        W_gpu : (n, m) pyopencl.array
            The weight matrix
        b_gpu : (n, ) pyopencl.array
            The bias unit

        Notes
        -----
        It is assumed that the input to this layer is a flattened vector. As such, when passing
        a multidimensional input, use a `flatten` layer first
    """
    def __init__(self, hidden_nodes, activation_function, l1=0.0, l2=0.0, trainable_mask_gpu=None, 
                 activation_kwargs=None, matmul_method=NaiveMatMul, **kwargs):
        """ A fully connected layer

            Parameters
            ----------
            hidden_nodes : int
                The number of neurons in this layer
            activation_function : str
                The name of the activation function of this layer
            activation_kwargs : dict of str - :obj:, optional
                The keyword arguments for the activation function if it has hyper-parameters
            matmul_method : dict of str - :obj:, optional
                The keyword arguments for the type of matrix multiplication (naive or tiled)
        """
        self.hidden_nodes = hidden_nodes

        self.activation_function = activation_function
        self.activation_kwargs = {} if activation_kwargs is None else activation_kwargs

        self.matmul_method = matmul_method

        self.output_shape = None
        self.input_shape = None
        
        self.W_gpu = None
        self.b_gpu = None
        
        if trainable_mask_gpu is not None:
            assert isinstance(trainable_mask_gpu, cl.ndarray)
            self.trainable_mask_gpu = trainable_mask_gpu.astype(np.int8)
        else:
            self.trainable_mask_gpu = None

        self.basis = None
        self.coeffs = None
        
        self.l1 = l1
        self.l2 = l2

        self.built = False

    def build(self, context, queue, previous_output_shape):
        """ Initialises the weight and bias units

            Parameters
            ----------
            previous_output_shape : 1 tuple of int
                The output shape of the previous layer. This will dictate the size of the weight matrix
        """
        self.context = context
        self.queue = queue

        self.output_shape = (self.hidden_nodes, )
        self.input_shape = previous_output_shape

        # Initialise the the weight with Glorot-Uniform, a uniform distribution over [-limit, limit],
        # where limit = sqrt(6 / (fan_in + fan_out)) (fan_in is the number of input units in the weight
        # tensor and fan_out is the number of output units).
        limit = np.sqrt(6 / (np.prod(self.input_shape) + np.prod(self.output_shape)))
        W = np.random.uniform(low=-limit, high=limit, size=(*self.output_shape, *previous_output_shape))

        self.W_gpu = cl_array.to_device(self.queue, W.astype(np.float32))
        self.b_gpu = cl_array.zeros(self.queue, self.output_shape, dtype=np.float32)
        
        if self.trainable_mask_gpu is not None:
            assert self.trainable_mask_gpu.shape == self.W_gpu.shape, f"Trainable mask {self.trainable_mask_gpu.shape} must have the " \
                                                              f"same shape as the weight {self.W_gpu.shape}"                                                  

        self.built = True

        self.gpu_maths = ArrayFunctions(self.context, self.queue)
        self.gpu_matmul = self.matmul_method(self.context, self.queue)
        self.gpu_transpose = GPUTranspose(context, queue)
        self.activation = ActivationFunctions(self.context, self.queue)

    def predict(self, z_gpu, output_only=True, **kwargs):
        """ Returns the output of this layer

            Parameters
            ----------
            z_gpu : (d, m) pyopencl.array
                z is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of z is the index that inputs is accessed by.
            output_only : bool, optional
                If set to true, then this function will return only the prediction of the neural
                network. If set to false then this will return the outputs of the individual
                layers. Unless back propagation is being performed, this should be set to true.

            Returns
            -------
            (d, n) pyopencl.array
                The final output of the layer, post activation

            OR (if `output_only = False`)

            (d, n) pyopencl.array, (d, n) pyopencl.array
                The first pyopencl.array will store the output before it is passed through the activation
                function.
                The second pyopencl.array will store the output after it has passed through the
                activation function.
        """
        check_layer(self)

        W_gpu_T = self.gpu_transpose.transpose(self.W_gpu)

        prod = self.gpu_matmul.Matmul(z_gpu, W_gpu_T)
        out_a = self.gpu_maths.addVector(prod, self.b_gpu)

        if output_only:
            return self.activation_function_(out_a)
        return out_a, self.activation_function_(out_a)

    def get_delta_backprop_(self, g_prime_gpu, new_delta_gpu, *args):
        """ Returns the delta for the previous layer, delta^{n-1}_{m,m}.

            Notes
            -----
            We want to return delta^{n-1} because the `sequential` class does not have access to the
            weights, W_gpu. But it does know the values of g'_{n-1} and delta^n, due to forward propagation
            and the backwards nature of the back propagation algorithm.

            Parameters
            ----------
            g_prime_gpu : (d, m) pyopencl.array
                Should be the derivative of the output of the previous layer, g'_{n-1}(a^{n-1}_{m,m})
            new_delta_gpu : (d, n) pyopencl.array
                The delta for this layer, delta^k_{m, m}

            Returns
            -------
            (d, m) pyopencl.array
                Returns delta of the previous layer, delta^{n-1}
        """
        check_layer(self)

        g_prime_gpu = g_prime_gpu.reshape(len(g_prime_gpu), -1)
        
        delta_result = g_prime_gpu * self.gpu_matmul.Matmul(new_delta_gpu, self.W_gpu)
        return delta_result

    def get_weight_grad_(self, delta_gpu, prev_z_gpu):
        """ Returns the associated partial S/partial W_gpu^n, that is
            the gradient with respect to the weight matrix in the kth layer

            Parameters
            ----------
            delta_gpu : (d, n) pyopencl.array
                In latex, this should be delta_k.
                Please import delta as a F-contiguous array.
            prev_z_gpu : (d, m) pyopencl.array
                This should be the output, post activation, of the previous layer (z_{n-1})

            Returns
            -------
            (n, ) pyopencl.array, (n, m) pyopencl.array
                The first array is the gradient for the bias unit
                The second array is the gradient for the weight matrix
        """
        check_layer(self)
        delta_gpu_T = self.gpu_transpose.transpose(delta_gpu)

        weight_grad = self.gpu_matmul.Matmul(delta_gpu_T, prev_z_gpu)
        bias_grad = self.gpu_maths.rowSumUp(delta_gpu)

        return bias_grad, weight_grad

    def update_parameters_(self, bias_updates_gpu, weight_updates_gpu):
        """ Perform an update to the weights by descending down the gradient

            Parameters
            ----------
            bias_updates_gpu : (n, ) pyopencl.array
                The gradients for the bias units
            weight_updates_gpu : (n, m) pyopencl.array
                The gradients for the weight matrix
        """
        check_layer(self)
 
        regularization_grad = cl_array.zeros(self.queue, self.W_gpu.shape, dtype=np.float32)
        if self.l1 > 0:
            regularization_grad += self.l1 * self.gpu_maths.clarray_sign(self.W_gpu)
        if self.l2 > 0:
            regularization_grad += self.l2 * self.W_gpu

        if self.trainable_mask_gpu is None:
            self.W_gpu -= (weight_updates_gpu + regularization_grad)
        else:
            self.W_gpu -= (weight_updates_gpu + regularization_grad) * self.trainable_mask_gpu
            
        self.b_gpu -= bias_updates_gpu

    def get_weights(self):
        check_layer(self)
        return self.W_gpu, self.b_gpu

    def summary_(self):
        check_layer(self)
        return f'Dense {(self.hidden_nodes,)}', f'Output Shape {(None, *self.output_shape)}'

    @property
    def activation_function_(self):
        return self.activation.get_activation_function(self.activation_function, **self.activation_kwargs)

    def __str__(self):
        return f'Dense: Output Shape {(None, *self.output_shape)}'
