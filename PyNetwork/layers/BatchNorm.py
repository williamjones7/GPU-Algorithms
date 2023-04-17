import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from pyopencl import clmath
from PyNetwork.validation import check_layer
from PyNetwork.layers import Layer
from PyNetwork import ActivationFunctions
from PyNetwork.utils import utils


class BatchNormGrads:
    """ Returns the gradients (partial derivatives) of this layer.
        Note that d is meant to be partial

        1. dgamma = d(S_gpu)/d(gamma_gpu)
        2. dbeta = d(S_gpu)/d(beta_gpu)
        3. dz = d(S_gpu)/d(z_gpu^{k-1})
            This is the gradient with respect to the input of the batch-norm layer
        4. dz_hat = d(S_gpu)/d(z_hat_gpu^{k-1})
            The gradient with respect to the normalised input of the batch-norm layer
        5. dsigma2 = d(S_gpu)/d(sigma_gpu^2)
        6. dmu = d(S_gpu)/d(mu_gpu)
    """
    def __init__(self, context, queue):
        BatchNormGrads.context = context
        BatchNormGrads.queue = queue
        BatchNormGrads.gpu_maths = utils.ArrayFunctions(context, queue)

    @staticmethod
    def dgamma(new_delta_gpu, z_hat_gpu):
        """ Returns d(S_gpu)/d(gamma_gpu)

            Parameters
            ----------
            new_delta_gpu : (N, ...) pyopencl.array
                Should be delta_gpu^{k}
            z_hat_gpu : (N, ...) pyopencl.array
                Should be the normalised input of this layer

            Returns
            -------
            (N, ...) pyopencl.array
        """
        return BatchNormGrads.gpu_maths.rowSumUp(new_delta_gpu * z_hat_gpu)

    @staticmethod
    def dbeta(new_delta_gpu):
        """ Returns d(S_gpu)/d(beta_gpu)

            Parameters
            ----------
            new_delta_gpu : (N, ...) pyopencl.array
                Should be delta_gpu^{k}

            Returns
            -------
            (N, ...) pyopencl.array
        """
        return BatchNormGrads.gpu_maths.rowSumUp(new_delta_gpu)

    @staticmethod
    def dz_hat(new_delta_gpu, gamma_gpu):
        """ Returns d(S_gpu)/d(z_hat_gpu) - The gradient with respect to the normalised input of
            the batch-norm layer

            Parameters
            ----------
            new_delta_gpu : (N, ...) pyopencl.array
                Should be delta_gpu^{k}
            gamma_gpu : (...) pyopencl.array

            Return
            ------
            (N, ...) pyopencl.array
        """
        return BatchNormGrads.gpu_maths.mulVector(new_delta_gpu, gamma_gpu)

    @staticmethod
    def dsigma2(z_gpu, dz_hat_gpu_, epsilon, mu_gpu=None, sigma_gpu=None):
        """ Returns d(S_gpu)/d(sigma_gpu^2)

            Parameters
            ----------
            z_gpu : (N, ...) pyopencl.array
                The input of this layer: z_gpu^{k-1}
            dz_hat_gpu_ : (N, ...) pyopencl.array
                The gradient with respect to the normalised input: d(S_gpu)/d(z_hat_gpu^{k-1})
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

        # Temporary variable equivalent to z_gpu - mu_gpu
        deviation = BatchNormGrads.gpu_maths.addVector(z_gpu, -mu_gpu)

        # Temporary variable for dsigma2
        c = (-0.5 * (sigma_gpu ** 2 + epsilon) ** (-3 / 2))

        return c * BatchNormGrads.gpu_maths.rowSumUp(dz_hat_gpu_ * deviation)

    @staticmethod
    def dmu(z_gpu, dz_hat_gpu_, epsilon, mu_gpu=None, sigma_gpu=None, dsigma2_gpu_=None):
        """ Returns d(S_gpu)/dmu

            Parameters
            ----------
            z_gpu : (N, ...) pyopencl.array
                The input of this layer: z_gpu^{k-1}
            dz_hat_gpu_ : (N, ...) pyopencl.array
                The gradient with respect to the normalised input: d(S_gpu)/d(z_hat_gpu^{k-1})
            epsilon : float

            mu_gpu : (...) pyopencl.array, optional
                The mean of the input. If None, then it will be computed
            sigma_gpu : (...) pyopencl.array, optional
                The std of the input. If None, then it will be computed
            dsigma2_gpu_ : (...) pyopencl.array, optional
                This should be the gradient d(S_gpu)/d(sigma_gpu^2). If it is set to None then it
                will be computed when this function is called
        """

        if mu_gpu is None:
            mu_gpu = BatchNormGrads.gpu_maths.rowMean(z_gpu)
        if sigma_gpu is None:
            sigma_gpu = BatchNormGrads.gpu_maths.rowStd(z_gpu)
        if dsigma2_gpu_ is None:
            dsigma2_gpu_ = BatchNormGrads.dsigma2(z_gpu, dz_hat_gpu_, epsilon, mu_gpu, sigma_gpu)

        # Temporary variable equivalent to z_gpu - mu_gpu       
        deviation = BatchNormGrads.gpu_maths.addVector(z_gpu, -mu_gpu)

        # Temporary variable for dmu
        c = (-1 / clmath.sqrt(sigma_gpu ** 2 + epsilon))

        return  c * BatchNormGrads.gpu_maths.rowSumUp(dz_hat_gpu_) + dsigma2_gpu_ * BatchNormGrads.gpu_maths.rowMean(-2 * deviation)
    
    @staticmethod
    def dz(z_gpu, new_delta_gpu, gamma_gpu, epsilon, mu_gpu=None, sigma_gpu=None):
        """ Returns the partial derivative with respect to the input: d(S_gpu)/dZ^{n-1}

            Parameters
            ----------
            z_gpu : (N, ...) pyopencl.array
                The input of this layer: z_gpu^{n-1}
            new_delta_gpu : (N, ...) pyopencl.array
                The back-prop gradient: delta_gpu^{n}
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

        # Temporary variable equivalent to z_gpu - mu_gpu
        deviation = BatchNormGrads.gpu_maths.addVector(z_gpu, -mu_gpu)

        # Temporary variable equivalent to dz_hat_ / np.sqrt(sigma ** 2 + epsilon)
        c_1 = BatchNormGrads.gpu_maths.divVector(dz_hat_gpu_, clmath.sqrt(sigma_gpu ** 2 + epsilon))

        # Temporary variable equivalent to dsigma2_ * 2 * deviation / m
        c_2 = BatchNormGrads.gpu_maths.mulVector(2 * deviation / m, dsigma2_gpu_)

        # Equivalent to dz_hat_ / np.sqrt(sigma ** 2 + epsilon) + dsigma2_ * 2 * deviation / m + dmu_ / m
        return  c_1 + BatchNormGrads.gpu_maths.addVector(c_2, dmu_gpu_ / m)


class BatchNorm(Layer):
    """ A BatchNorm layer where the inputs are normalised and then linearly scaled.
        Concretely, given an input z_gpu, this layer will return
            gamma_gpu * z_hat_gpu + beta_gpu
        where z_hat_gpu is the normliased version of z_gpu, and gamma_gpu and beta_gpu are matrices

        Attributes
        ----------
        activation_function : str
            The name of the activation function

        input_shape : (k, ) tuple
            The shape of the input of this layer
        output_shape : (k, ) tuple
            The shape of the output of this layer
        gamma_gpu : (k, ) pyopencl.array
            The scaling factor of the elements
        beta_gpu : (k, ) pyopencl.array
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

        self.gamma_gpu = None
        self.beta_gpu = None

    def build(self, context, queue, previous_output_shape):
        """ Initialise Attributes `gamma_gpu` and `beta_gpu`

            Parameters
            ----------
            previous_output_shape : (k, ) tuple
                The shape of the input of this layer
        """
        self.context = context
        self.queue = queue

        self.input_shape = previous_output_shape
        self.output_shape = previous_output_shape

        self.gamma_gpu = 1 + cl_array.zeros(self.queue, self.input_shape, dtype=np.float32)
        self.beta_gpu = cl_array.zeros(self.queue, self.input_shape, dtype=np.float32)

        self.built = True

        self.gpu_maths = utils.ArrayFunctions(self.context, self.queue)
        self.batch_norm_grads = BatchNormGrads(self.context, self.queue)
        self.activation = ActivationFunctions(self.context, self.queue)

    def predict(self, z_gpu, output_only=True, **kwargs):
        """ Returns the output of this layer

            Parameters
            ----------
            z_gpu : (N, ...) pyopencl.array
                z_gpu is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of z_gpu is the index that inputs is accessed by
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

        mu_gpu = self.gpu_maths.rowMean(z_gpu)
        std = self.gpu_maths.rowStd(z_gpu)

        # Temporary variable equivalent to z_gpu - mu_gpu
        deviation = self.gpu_maths.addVector(z_gpu, -mu_gpu)

        # Temporary variable equivalent to (z_gpu - mu_gpu) / np.sqrt(std ** 2 + self.epsilon)
        c_1 = self.gpu_maths.divVector(deviation, clmath.sqrt(std ** 2 + self.epsilon))

        # Temporary variable equivalent to self.gamma * ((z_gpu - mu_gpu) / np.sqrt(std ** 2 + self.epsilon))
        c_2 = self.gpu_maths.mulVector(c_1, self.gamma_gpu)

        # Equivalent to self.gamma * ((z_gpu - mu_gpu) / np.sqrt(std ** 2 + self.epsilon)) + self.beta
        a = self.gpu_maths.addVector(c_2, self.beta_gpu)

        if output_only:
            return a
        return a, a

    def get_delta_backprop_(self, g_prime_gpu, new_delta_gpu, prev_z_gpu):
        """ Returns the delta_gpu for the previous layer, delta_gpu^{k-1}_{m,j}.

            Parameters
            ----------
            g_prime_gpu : (N, ...) pyopencl.array
                Should be the derivative of the ouput of the previous layer, g'_{k-1}(a^{k-1}_{m,j})
            new_delta_gpu : (N, ...) pyopencl.array
                The delta_gpu for this layer, delta_gpu^k_{m, j}
            prev_z_gpu : (N, ...) pyopencl.array
                The input for this layer, z_gpu^{k-1}

            Returns
            -------
            (N, ...) pyopencl.array
                Returns delta_gpu of the previous layer, delta_gpu^{k-1}

            Notes
            -----
            We want to return delta_gpu^{k-1} because the `sequential` class does not have access to the
            weights, W. But it does know the values of g'_{k-1} and delta_gpu^k, due to forward propagation
            and the backwards nature of the back propagation algorithm.
        """
        check_layer(self)

        dz_ = BatchNormGrads.dz(prev_z_gpu, new_delta_gpu, self.gamma_gpu, self.epsilon)
        return dz_ * prev_z_gpu

    def get_weight_grad_(self, delta_gpu, prev_z_gpu):
        """ Returns the gradients with respect to beta_gpu and gamma_gpu

            Parameters
            ----------
            delta_gpu : (N, ...) pyopencl.array
                Should be delta_gpu^k
            prev_z_gpu : (N, ...) pyopencl.array
                The input of this layer: z_gpu^{k-1}

            Returns
            -------
            (...) pyopencl.array, (...) pyopencl.array
                The first pyopencl.array is d(S_gpu)/d(beta_gpu)
                The second pyopencl.array is d(S_gpu)/d(gamma_gpu)
        """
        check_layer(self)

        # Temporary variable equivalent to (prev_z - np.mean(prev_z, axis=0))
        c_1 = self.gpu_maths.addVector(prev_z_gpu, -self.gpu_maths.rowMean(prev_z_gpu))

        # Temporary variable equivalent to np.sqrt(np.std(prev_z, axis=0) ** 2 + self.epsilon)
        c_2 = clmath.sqrt(self.gpu_maths.rowStd(prev_z_gpu) ** 2 + self.epsilon)

        # Equivalent to (prev_z - np.mean(prev_z, axis=0)) / np.sqrt(np.std(prev_z, axis=0) ** 2 + self.epsilon)
        z_hat_gpu = self.gpu_maths.divVector(c_1, c_2)

        beta_grad = self.gpu_maths.rowSumUp(delta_gpu)
        gamma_grad = self.gpu_maths.rowSumUp(delta_gpu * z_hat_gpu)
        return beta_grad, gamma_grad


    def update_parameters_(self, beta_updates_gpu, gamma_updates_gpu):
        """ Perform an update to the weights by descending down the gradient

            Parameters
            ----------
            beta_updates_gpu : (k, ) pyopencl.array
                Should be d(S_gpu)/d(beta_gpu), as scheduled by the optimizer
            gamma_updates_gpu : (k, ) pyopencl.array
                Should be d(S_gpu)/d(gamma_gpu), as scheduled by the optimizer
        """
        check_layer(self)

        self.beta_gpu -= beta_updates_gpu
        self.gamma_gpu -= gamma_updates_gpu

    def get_weights(self):
        check_layer(self)
        return self.beta_gpu, self.gamma_gpu

    def summary_(self):
        check_layer(self)
        return f'Batch Norm', f"Output Shape {(None, *self.output_shape)}"

    @property
    def activation_function_(self):
        return self.activation.get_activation_function(self.activation_function)

    def __str__(self):
        return f'Batch Norm; built = {self.built}'
