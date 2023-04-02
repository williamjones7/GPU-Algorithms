import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel
from pyopencl import clmath

import time

x = np.array([1.0, 4.0, 9.0, 16.0])
# x = np.ones(100)

platform = cl.get_platforms()
devices = platform[0].get_devices()
context = cl.Context(devices)
queue = cl.CommandQueue(context)

def GPU_sqrt(queue, x):
    x_gpu = cl_array.to_device(queue, x)
    return clmath.sqrt(x_gpu)

def GPU_max(queue, x):
    x_gpu = cl_array.to_device(queue,x)
    return cl_array.max(x_gpu)

def GPU_sum(queue, x):
    x_gpu = cl_array.to_device(queue,x)
    return cl_array.sum(x_gpu,queue)


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
        def relu(x_gpu, grad=False):
            x_gpu_precise = x_gpu.astype(np.float64)
            relu_program = ElementwiseKernel(context,
                                 "double *x, double *out",
                                 "out[i] = x[i] > 0 ? x[i] : 0.0",
                                 "relu")

            relu_grad = ElementwiseKernel(context,
                                 "double *x, double *out",
                                 "out[i] = x[i] > 0 ? 1.0 : 0.0",
                                 "relu")
            out_gpu = cl_array.empty_like(x_gpu_precise)

            if grad:
                relu_grad(x_gpu_precise, out_gpu).wait()
                return out_gpu.astype(np.float32)
            relu_program(x_gpu_precise, out_gpu).wait()
            return out_gpu.astype(np.float32)
        # END def relu

        return relu

    elif name == 'linear':
        def linear(x_gpu, grad=False):
            if grad:
                return 1 + cl_array.zeros_like(x_gpu)
            return x_gpu
        return linear
        
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
#        # END def softmax
#
#        return softmax

    else:
        raise Exception(f'{name} is not a defined function.')


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
        def mse(predictions, targets, grad = False):
            if grad:
                return (predictions - targets) * 2
            N = predictions.shape[0]
            RS = ((predictions - targets) * (predictions - targets))/2
            return cl_array.sum(RS)/N
        return mse
    elif name == 'cross_entropy':
        def cross_entropy(predictions, targets, epsilon=1e-12, grad=False):
            """ Computes cross entropy between targets (encoded as one-hot vectors) and predictions.
                Parameters
                ----------
                    predictions : (N, k) np.array
                    targets     : (N, k) np.array
                Returns
                -------
                    float
                        If grad = False then the cross_entropy score is retuned
                    OR
                    (N, k) np.array
                        If grad = True then the gradient of the output is returned
            """
            predictions_precise = predictions.astype(np.float64)
            targets_precise = targets.astype(np.float64)
            clip_clarray_min = ElementwiseKernel(context,
                                                 "double *x, double threshold",
                                                 "x[i] = x[i] > threshold ? "
                                                 "x[i] : threshold",
                                                 "clip_in_place_elementwise")
            clip_clarray_max = ElementwiseKernel(context,
                                                 "double *x, double threshold",
                                                 "x[i] = x[i] < threshold ? "
                                                 "x[i] : threshold",
                                                 "clip_in_place_elementwise")
            def clip_clarray(array, min, max):

                if min is not None:
                    clip_clarray_min(array, min)
                if max is not None:
                    clip_clarray_max(array, max)

            clip_clarray(predictions_precise, epsilon, 1.0 - epsilon)

            if grad:
                return (-targets_precise / predictions_precise + (1.0 - targets_precise) / (1.0 - predictions_precise)).astype(np.float32)

            N = predictions_precise.shape[0]
            MP = targets_precise * clmath.log(predictions_precise + 1e-9)
            ce = -cl_array.sum(MP) / N
            return ce.astype(np.float32)
        return cross_entropy
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

    
