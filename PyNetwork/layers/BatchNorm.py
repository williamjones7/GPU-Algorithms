import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from pyopencl import clmath
from PyNetwork.validation import check_layer
from PyNetwork.layers import Layer
from PyNetwork import get_activation_function
from PyNetwork import utils


class BatchNormGrads:
    """ Returns the gradients (partial derivatives) of this layer.
        Note that d is meant to be partial

        1. dgamma = dS/d(gamma)
        2. dbeta = dS/d(beta)
        3. dz = dS/d(z^{k-1})
            This is the gradient with respect to the input of the batch-norm layer
        4. dz_hat = dS/d(z_hat^{k-1})
            The gradient with respect to the normalised input of the batch-norm layer
        5. dsigma2 = dS/d(sigma^2)
        6. dmu = dS/d(mu)
    """
    def __init__(self, context, queue):
        BatchNormGrads.context = context
        BatchNormGrads.queue = queue
        BatchNormGrads.gpu_maths = utils.ArrayMathsFunction(context, queue)

    @staticmethod
    def dgamma(new_delta_gpu, z_hat_gpu):
        """ Returns ds/d(gamma)

            Parameters
            ----------
            new_delta_gpu : (N, ...) pyopencl.array
                Should be delta^{k}
            z_hat_gpu : (N, ...) pyopencl.array
                Should be the normalised input of this layer

            Returns
            -------
            (N, ...) pyopencl.array
        """
        return BatchNormGrads.gpu_maths.rowSumUp(new_delta_gpu * z_hat_gpu)

    @staticmethod
    def dbeta(new_delta_gpu):
        """ Returns ds/d(beta)

            Parameters
            ----------
            new_delta_gpu : (N, ...) pyopencl.array
                Should be delta^{k}

            Returns
            -------
            (N, ...) pyopencl.array
        """
        return BatchNormGrads.gpu_maths.rowSumUp(new_delta_gpu)

    @staticmethod
    def dz_hat(new_delta_gpu, gamma_gpu):
        """ Returns dS/d(z_hat) - The gradient with respect to the normalised input of
            the batch-norm layer

            Parameters
            ----------
            new_delta_gpu : (N, ...) pyopencl.array
                Should be delta^{k}
            gamma_gpu : (...) pyopencl.array

            Return
            ------
            (N, ...) pyopencl.array
        """
        return BatchNormGrads.gpu_maths.mulVector(new_delta_gpu, gamma_gpu)

    @staticmethod
    def dsigma2(z_gpu, dz_hat_gpu_, epsilon, mu_gpu=None, sigma_gpu=None):
        """ Returns dS/d(sigma^2)

            Parameters
            ----------
            z_gpu : (N, ...) pyopencl.array
                The input of this layer: z^{k-1}
            dz_hat_gpu_ : (N, ...) pyopencl.array
                The gradient with respect to the normalised input: dS/d(z_hat^{k-1})
            epsilon : float

            mu_gpu : (...) pyopencl.array, optional
                The mean of the input. If None, then it will be computed
            sigma_gpu : (...) pyopencl.array, optional
                The std of the input. If None, then it will be computed

            Returns
            -------
            (...) pyopencl.array
        """
        if mu_gpu is None:
            mu_gpu = BatchNormGrads.gpu_maths.rowMean(z_gpu)
        if sigma_gpu is None:
            sigma_gpu = BatchNormGrads.gpu_maths.rowStd(z_gpu)

        deviation = BatchNormGrads.gpu_maths.addVector(z_gpu, -mu_gpu)
        c = (-0.5 * (sigma_gpu ** 2 + epsilon) ** (-3 / 2))
        return c * BatchNormGrads.gpu_maths.rowSumUp(dz_hat_gpu_ * deviation)

    @staticmethod
    def dmu(z_gpu, dz_hat_gpu_, epsilon, mu_gpu=None, sigma_gpu=None, dsigma2_gpu_=None):
        """ Returns dS/dmu

            Parameters
            ----------
            z_gpu : (N, ...) pyopencl.array
                The input of this layer: z^{k-1}
            dz_hat_gpu_ : (N, ...) pyopencl.array
                The gradient with respect to the normalised input: dS/d(z_hat^{k-1})
            epsilon : float

            mu_gpu : (...) pyopencl.array, optional
                The mean of the input. If None, then it will be computed
            sigma_gpu : (...) pyopencl.array, optional
                The std of the input. If None, then it will be computed
            dsigma2_gpu_ : (...) pyopencl.array, optional
                This should be the gradient ds/d(sigma^2). If it is set to None then it
                will be computed when this function is called
        """

        if mu_gpu is None:
            mu_gpu = BatchNormGrads.gpu_maths.rowMean(z_gpu)
        if sigma_gpu is None:
            sigma_gpu = BatchNormGrads.gpu_maths.rowStd(z_gpu)
        if dsigma2_gpu_ is None:
            dsigma2_gpu_ = BatchNormGrads.dsigma2(z_gpu, dz_hat_gpu_, epsilon, mu_gpu, sigma_gpu)
        
        deviation = BatchNormGrads.gpu_maths.addVector(z_gpu, -mu_gpu)
        c = (-1 / clmath.sqrt(sigma_gpu ** 2 + epsilon))

        return  c * BatchNormGrads.gpu_maths.rowSumUp(dz_hat_gpu_) + dsigma2_gpu_ * BatchNormGrads.gpu_maths.rowMean(-2 * deviation)
    @staticmethod
    def dz(z_gpu, new_delta_gpu, gamma_gpu, epsilon, mu=None, sigma=None):
        """ Returns the partial derivative with respect to the input: dS/dZ^{n-1}

            Parameters
            ----------
            z_gpu : (N, ...) pyopencl.array
                The input of this layer: z^{n-1}
            new_delta_gpu : (N, ...) pyopencl.array
                The back-prop gradient: delta^{n}
            gamma_gpu : (...) pyopencl.array
            epsilon : float
                Arbitrarily small float to prevent division by 0 error

            mu_gpu : (...) pyopencl.array, optional
                The mean of the input. If None, then it will be computed
            sigma_gpu : (...) pyopencl.array, optional
                The std of the input. If None, then it will be computed

            Returns
            -------
            (N, ...) pyopencl.array
        """

        if mu_gpu is None:
            mu_gpu = BatchNormGrads.gpu_maths.rowMean(z_gpu)
        if sigma_gpu is None:
            sigma_gpu = BatchNormGrads.gpu_maths.rowStd(z_gpu)
        m = len(z_gpu)

        dz_hat_gpu_ = BatchNormGrads.dz_hat(new_delta_gpu, gamma_gpu)
        dsigma2_gpu_ = BatchNormGrads.dsigma2(z_gpu, dz_hat_gpu_, epsilon, mu_gpu, sigma_gpu)
        dmu_gpu_ = BatchNormGrads.dmu(z_gpu, dz_hat_gpu_, epsilon, mu_gpu, sigma_gpu, dsigma2_gpu_)

        return dz_hat_gpu_ / clmath.sqrt(sigma_gpu**2 + epsilon) + dsigma2_gpu_ * 2 * (z_gpu - mu_gpu)/m + dmu_gpu_/m


class BatchNorm(Layer):
    """ A BatchNorm layer where the inputs are normalised and then linearly scaled.
        Concretely, given an input z, this layer will return
            gamma * z_hat + beta
        where z_hat is the normliased version of z, and gamma and beta are matrices

        Attributes
        ----------
        activation_function : str
            The name of the activation function

        input_shape : k tuple
            The shape of the input of this layer
        output_shape : k tuple
            The shape of the output of this layer
        gamma : pyopencl.array (of dimension k)
            The scaling factor of the elements
        beta : pyopencl.array (of dimension k)
            The bias units for the elements

        built : bool
            Has the model been initialised

        Notes
        -----
        This implementation of batch-norm assumes that batch-norm is applied AFTER the activation function. For
        example:
            Correct Use: Fully Connected -> Activation Function -> Batch-Norm (Batch-norm after activation)
            Incorrect Use: Fully Connected -> Batch-Norm -> Activation Function (Batch-norm before activation)
    """

    def __init__(self):
        """ Initise Class """
        self.built = False
        self.epsilon = 1e-10

        self.activation_function = 'linear'

        self.input_shape = None
        self.output_shape = None

        self.gamma = None
        self.beta = None

    def build(self, previous_output_shape, context, queue):
        """ Initialise Attributes `gamma` and `beta`

            Parameters
            ----------
            previous_output_shape : k tuple
                The shape of the input of this layer
        """
        self.input_shape = previous_output_shape
        self.output_shape = previous_output_shape

        self.gamma = np.ones(self.input_shape)
        self.beta = cl_array.zeros(self.queue, self.input_shape)

        self.built = True

        self.context = context
        self.queue = queue

    def predict(self, z, output_only=True, **kwargs):
        """ Returns the output of this layer

            Parameters
            ----------
            z : (N, ...) pyopencl.array
                z is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of z is the index that inputs is accessed by
            output_only : bool, optional
                If set to true, then this function will return only the prediction of the neural
                network. If set to false then this will return the outputs of the individual
                layers. Unless back propagation is being performed, this should be set to true.

            Returns
            -------
            (N, ...) pyopencl.array
                The final output of the layer, post activation

            OR (if `output_only = False`)

            (N, ...) pyopencl.array, (N, ...) pyopencl.array
                The first pyopencl.array will store the output before it is passed through the activation
                function.
                The second pyopencl.array will store the output after it has passed through the
                activation function.

            Notes
            -----
            Since the activation function is linear the 2 arrays, when output_only = True, are the same
            array
        """
        check_layer(self)

        mean = np.mean(z, axis=0)
        std = np.std(z, axis=0)

        a = self.gamma * ((z - mean) / np.sqrt(std ** 2 + self.epsilon)) + self.beta

        if output_only:
            return a
        return a, a

    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        """ Returns the delta for the previous layer, delta^{k-1}_{m,j}.

            Parameters
            ----------
            g_prime : (N, ...) pyopencl.array
                Should be the derivative of the ouput of the previous layer, g'_{k-1}(a^{k-1}_{m,j})
            new_delta : (N, ...) pyopencl.array
                The delta for this layer, delta^k_{m, j}
            prev_z : (N, ...) pyopencl.array
                The input for this layer, z^{k-1}

            Returns
            -------
            (N, ...) pyopencl.array
                Returns delta of the previous layer, delta^{k-1}

            Notes
            -----
            We want to return delta^{k-1} because the `sequential` class does not have access to the
            weights, W. But it does know the values of g'_{k-1} and delta^k, due to forward propagation
            and the backwards nature of the back propagation algorithm.
        """
        check_layer(self)

        dz_ = BatchNormGrads.dz(prev_z, new_delta, self.gamma, self.epsilon)
        return dz_ * prev_z

    def get_weight_grad_(self, delta, prev_z):
        """ Returns the gradients with respect to beta and gamma

            Parameters
            ----------
            delta : (N, ...) pyopencl.array
                Should be delta^k
            prev_z : (N, ...) pyopencl.array
                The input of this layer: z^{k-1}

            Returns
            -------
            (...) pyopencl.array, (...) pyopencl.array
                The first pyopencl.array is dS/d(beta)
                The second pyopencl.array is dS/d(gamma)
        """
        check_layer(self)

        z_hat = (prev_z - np.mean(prev_z, axis=0)) / np.sqrt(np.std(prev_z, axis=0) ** 2 + self.epsilon)
        return np.sum(delta, axis=0), np.sum(delta * z_hat, axis=0)

    def update_parameters_(self, beta_updates, gamma_updates):
        """ Perform an update to the weights by descending down the gradient

            Parameters
            ----------
            beta_updates : pyopencl.array (of dimension k)
                Should be dS/d(beta), as scheduled by the optimizer
            gamma_updates : pyopencl.array (of dimension k)
                Should be dS/d(gamma), as scheduled by the optimizer
        """
        check_layer(self)

        self.beta -= beta_updates
        self.gamma -= gamma_updates

    def get_weights(self):
        check_layer(self)
        return self.beta, self.gamma

    def summary_(self):
        check_layer(self)
        return f'Batch Norm', f"Output Shape {(None, *self.output_shape)}"

    @property
    def activation_function_(self):
        return get_activation_function(self.activation_function)

    def __str__(self):
        return f'Batch Norm; built = {self.built}'
