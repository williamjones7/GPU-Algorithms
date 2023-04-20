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

        ActivationFunctions.softmax_program = cl.Program(context, utils.softmax_program).build()
        

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
    def softmax(x_gpu, grad=False):
        x_gpu_precise = x_gpu.astype(np.float64)
        input_size = np.int32(x_gpu_precise.shape[-1])
        max_gpu = ArrayFunctions.rowMax(x_gpu_precise)
        
        soft_value = cl_array.empty_like(x_gpu_precise)

        global_col = x_gpu_precise.shape[-1]
        global_row = x_gpu_precise.shape[:-1]
        global_shape = (np.prod(global_row).astype(np.int32), global_col)

        ActivationFunctions.softmax_program.softmax(ActivationFunctions.queue, global_shape, None,
                                                    x_gpu_precise.data, max_gpu.data, input_size,
                                                    soft_value.data).wait()
        if grad:
            grad_gpu = (1 - soft_value) * soft_value
            return grad_gpu.astype(np.float32)
        return soft_value.astype(np.float32)

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

        elif name == 'softmax':
            return ActivationFunctions.softmax

        elif name == 'linear':
            return ActivationFunctions.linear

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
        
        predicted_labels = MetricFunctions.gpu_maths.rowArgmax(predictions_gpu)
        target_labels = MetricFunctions.gpu_maths.rowArgmax(target_gpu)

        correct_labels = predicted_labels == target_labels
        return cl_array.sum(correct_labels.astype(np.int32)) / N
    
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

    
