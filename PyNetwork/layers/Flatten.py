import numpy as np

from PyNetwork import ActivationFunctions
from PyNetwork.layers import Layer

from PyNetwork.validation import check_layer


class Flatten(Layer):
    """ Given a n-dimensional input, this layer will return the flatten representation
        of the input

        Attributes
        ----------
        input_shape : tuple
            The input shape
        output_shape : 1 tuple
            The output shape
        built : bool
            Has the layer been initialised
        activation_function : str
            Since this is a pass-through layer, the activation function

        Notes
        -----
        When a n-dimensional input is fed into a `Dense` layer, it needs to be flattened
        into a vector first. This `Flatten` class performs such flattening
    """

    def __init__(self):
        self.built = False
        self.activation_function = 'linear'

        self.input_shape = None
        self.output_shape = None

    def build(self, context, queue, previous_output_shape):
        """ Built/initialised the layer

            Parameters
            ----------
            previous_output_shape : tuple
                The shape of the input into this layer.
        """
        self.context = context
        self.queue = queue

        self.input_shape = previous_output_shape
        self.output_shape = (np.prod(previous_output_shape), )

        self.built = True

    def predict(self, z_gpu, output_only=True, pre_activation_of_input=None):
        """ Returns the prediction of this layer

            Parameters
            ----------
            z_gpu : (N, *input_shape) pyopencl.array
                The input to be flattened
            output_only : bool, optional
                If set to true, then this function will return only the prediction of the neural
                network. If set to false then this will return the outputs of the individual
                layers. Unless back propagation is being performed, this should be set to true.
            pre_activation_of_input : (N, *input_shape) pyopencl.array
                The input, z_gpu, before it passed through the activation function

            Returns
            -------
            (N, *output_shape) pyopencl.array
                The flattened representation of the input

            OR (if `output_only = False`)

            (N, *input_shape) pyopencl.array, (N, *output_shape) pyopencl.array
                The first pyopencl.array will store the output before it has been reshaped
                The second pyopencl.array will store the output after it has been reshaped

            Notes
            -----
            Since this layer has no activation function,
        """

        check_layer(self)

        if output_only:
            return z_gpu.reshape(len(z_gpu), self.output_shape[0])
        return pre_activation_of_input, z_gpu.reshape(len(z_gpu), self.output_shape[0])

    def get_delta_backprop_(self, g_prime_gpu, new_delta_gpu, *args, **kwargs):
        """ Returns the delta for the previous layer, delta^{k-1}_{m,j}.

            Parameters
            ----------
            g_prime_gpu : (N, *input_shape) pyopencl.array
                Should be the derivative of the ouput of the previous layer, g'_{k-1}(a^{k-1}_{m,j})
            new_delta_gpu : (N, *output_shape) pyopencl.array
                The delta for this layer, delta^k_{m, j}

            Returns
            -------
            (N, *input_shape) pyopencl.array

            Notes
            -----
            Since this is a pass through layer (i.e. linear activation), g_prime_gpu = 1, and so can be ignored.
            The key to this layer is that the delta of the k+1 layer needs to be reshaped
            for the k-1 layer
        """
        check_layer(self)
        return new_delta_gpu.reshape(len(new_delta_gpu), *self.input_shape)

    def get_weight_grad_(self, *args, **kwargs):
        """ Returns the associated partial S/partial W^k, that is
            the gradient with respect to the weight matrix in the kth layer

            Returns
            -------
            (None, None)

            Notes
            -----
            Since nothing in this layer is trainiable, the gradients is simply None
        """
        check_layer(self)
        return None, None

    def update_parameters_(self, *args, **kwargs):
        """ Perform an update to the weights by descending down the gradient

            Notes
            -----
            Since nothing in this layer is trainiable, we can simply pass
        """
        check_layer(self)
        pass

    def get_weights(self):
        check_layer(self)
        return None, None

    def summary_(self):
        check_layer(self)
        return f'Flatten', f'Output Shape {(None, *self.output_shape)}'

    @property
    def activation_function_(self):
        return ActivationFunctions.get_activation_function(self.activation_function)

    def __str__(self):
        return f'Flatten'
