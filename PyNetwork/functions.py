import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel
from pyopencl import clmath
from PyNetwork import utils
from PyNetwork.utils import ArrayFunctions

def GPU_sqrt(queue, x):
    x_gpu = cl_array.to_device(queue, x)
    x_gpu_precise = x_gpu.astype(np.float64)
    return clmath.sqrt(x_gpu_precise,queue=queue)

def GPU_max(queue, x):
    x_gpu = cl_array.to_device(queue,x)
    x_gpu_precise = x_gpu.astype(np.float64)
    return cl_array.max(x_gpu_precise,queue=queue)

def GPU_sum(queue, x):
    x_gpu = cl_array.to_device(queue,x)
    x_gpu_precise = x_gpu.astype(np.float64)
    return cl_array.sum(x_gpu_precise,queue=queue)



class ActivationFunctions:
    def __init__(self, context, queue):
        ActivationFunctions.context = context
        ActivationFunctions.queue = queue

        ActivationFunctions.relu_program = ElementwiseKernel(ActivationFunctions.context,
                                                             "double *x, double *out",
                                                             "out[i] = x[i] > 0.0 ? x[i] : 0.0",
                                                             "relu")
        
        ActivationFunctions.relu_grad_program = ElementwiseKernel(ActivationFunctions.context,
                                                                  "double *x, double *out",
                                                                  "out[i] = x[i] > 0.0 ? 1.0 : 0.0",
                                                                  "relu_grad")
        

    @staticmethod
    def relu(x_gpu, grad=False):
        x_gpu_precise = x_gpu.astype(np.float64)
        out_gpu = cl_array.zeros_like(x_gpu_precise)

        if grad:
            ActivationFunctions.relu_grad_program(x_gpu_precise, out_gpu).wait()
            return out_gpu.astype(np.float32)

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
    def mse(predictions_gpu, targets_gpu, grad = False):
        if grad:
            return (predictions_gpu - targets_gpu) * 2
        N, _ = predictions_gpu.shape
        RS = ((predictions_gpu - targets_gpu) * (predictions_gpu - targets_gpu)) / 2
        return cl_array.sum(RS) / N
    
    @staticmethod
    def cross_entropy(predictions_gpu, targets_gpu, epsilon=1e-12, grad=False):
        """ Computes cross entropy between targets (encoded as one-hot vectors) and predictions.
            Parameters
            ----------
                predictions_gpu : (N, k) pyopencl.array
                targets_gpu     : (N, k) pyopencl.array
            Returns
            -------
                float
                    If grad = False then the cross_entropy score is retuned
                OR
                (N, k) pyopencl.array
                    If grad = True then the gradient of the output is returned
        """
        ArrayFunctions.clarray_clip(predictions_gpu, epsilon, 1.0 - epsilon)

        if grad:
            return (-targets_gpu / predictions_gpu + (1.0 - targets_gpu) / (1.0 - predictions_gpu))

        N, _ = predictions_gpu.shape
        MP = targets_gpu * clmath.log(predictions_gpu + 1e-9)
        ce = -cl_array.sum(MP) / N
        return ce
    
    @staticmethod
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

class MetricFunctions:
    def __init__(self, context, queue):
        MetricFunctions.gpu_maths = utils.ArrayFunctions(context, queue)
    
    @staticmethod
    def accuracy(predictions_gpu, target_gpu):
        N = predictions_gpu.shape[0]
        return cl_array.sum(MetricFunctions.gpu_maths.rowArgmax(predictions_gpu)
                            == MetricFunctions.gpu_maths.rowArgmax(target_gpu)) / N
    
    @staticmethod
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
            return MetricFunctions.accuracy
        else:
            raise Exception(f'{name} is not a defined metric.')

    
