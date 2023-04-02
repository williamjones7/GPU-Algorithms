import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel
from pyopencl import clmath
from PyNetwork import utils
from PyNetwork.utils import ArrayFunctions

# platform = cl.get_platforms()
# devices = platform[0].get_devices()
# context = cl.Context(devices)
# queue = cl.CommandQueue(context)

def GPU_sqrt(queue, x):
    x_gpu = cl_array.to_device(queue, x)
    return clmath.sqrt(x_gpu)

def GPU_max(queue, x):
    x_gpu = cl_array.to_device(queue,x)
    return cl_array.max(x_gpu)

def GPU_sum(queue, x):
    x_gpu = cl_array.to_device(queue,x)
    return cl_array.sum(x_gpu,queue)

class ActivationFunctions:
    def __init__(self, context, queue):
        
        ActivationFunctions.context = context
        ActivationFunctions.queue = queue
        ActivationFunctions.relu_program = ElementwiseKernel(ActivationFunctions.context,
                                                                  "double *x, double *out",
                                                                  "out[i] = x[i] > 0 ? x[i] : 0.0",
                                                                  "relu")
        ActivationFunctions.relu_grad_program = ElementwiseKernel(ActivationFunctions.context,
                            "double *x, double *out",
                            "out[i] = x[i] > 0 ? 1.0 : 0.0",
                            "relu_grad")
        

    @staticmethod
    def relu(x_gpu, grad=False):
        print(ActivationFunctions.context)
        x_gpu_precise = x_gpu.astype(np.float64)
        out_gpu = cl_array.zeros_like(x_gpu_precise)

        if grad:
            ActivationFunctions.relu_grad_program(x_gpu_precise, out_gpu).wait()

        ActivationFunctions.relu_program(x_gpu_precise, out_gpu).wait()
        return out_gpu.astype(np.float32)

    @staticmethod
    def linear(x_gpu, grad=False):
        if grad:
            return 1 + cl_array.zeros_like(x_gpu)
        return x_gpu

    @staticmethod
    def get_activation_function(name, **kwargs):
        """ Returns the function of the given name
            Parameters
            ----------
            name : str
                The name of the desired function
            Raises
            ------
            Exception
                If `name` has not been implemented
        """

        if name == 'relu':
            return ActivationFunctions.relu

        elif name == 'linear':
            return ActivationFunctions.linear
            
    #    elif name == 'softmax':
    #        def softmax(x, grad=False):
    #            if grad:
    #                softmax_val = softmax(x, grad=False)
    #                return softmax_val*(1 - softmax_val)
    #
    #            z = x - np.max(x, axis=-1, keepdims=True)
    #            numerator = np.exp(z)
    #            denominator = np.sum(numerator, axis=-1, keepdims=True)
    #            return numerator / denominator
    #
    #        return softmax

        else:
            raise Exception(f'{name} is not a defined function.')


class ErrorFunctions:

    @staticmethod
    def mse(predictions, targets, grad = False):
        if grad:
            return (predictions - targets) * 2
        N, _ = predictions.shape
        RS = ((predictions - targets) * (predictions - targets)) / 2
        return cl_array.sum(RS) / N
    
    @staticmethod
    def cross_entropy(predictions, targets, epsilon=1e-12, grad=False):
        """ Computes cross entropy between targets (encoded as one-hot vectors) and predictions.
            Parameters
            ----------
                predictions : (N, k) pyopencl.array
                targets     : (N, k) pyopencl.array
            Returns
            -------
                float
                    If grad = False then the cross_entropy score is retuned
                OR
                (N, k) pyopencl.array
                    If grad = True then the gradient of the output is returned
        """
        ArrayFunctions.clarray_clip(predictions, epsilon, 1.0 - epsilon)

        if grad:
            return (-targets / predictions + (1.0 - targets) / (1.0 - predictions))

        N, _ = predictions.shape
        MP = targets * clmath.log(predictions + 1e-9)
        ce = -cl_array.sum(MP) / N
        return ce
    

    def get_error_function(name):
        """ Returns the function of the given name
            Parameters
            ----------
            name : str
                The name of the desired function
            Raises
            ------
            Exception
                If `name` has not been implemented
        """
        if name == 'mse':
            return ErrorFunctions.mse
        elif name == 'cross_entropy':
            return ErrorFunctions.cross_entropy
        else:
            raise Exception(f'{name} is not a defined function.')


def get_metric_function(name):
    """ Returns the metric fucntion of a given name

        Parameters
        ----------
        name : str
            The name of the desired function

        Raises
        ------
        Exception
            If `name` has not been implemented
    """
    if name == 'accuracy':
        def accuracy(predictions, target):
            return np.mean(np.argmax(predictions, axis=-1) == np.argmax(target, axis=-1))
        return accuracy
    else:
        raise Exception(f'{name} is not a defined metric.')

    
