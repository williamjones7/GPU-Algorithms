import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from pyopencl.elementwise import ElementwiseKernel
from pyopencl import clmath
from PyNetwork import utils


def buffer_str(input_str, max_buffer=30):
    if len(input_str) < max_buffer:
        return input_str + ' ' * (max_buffer - len(input_str))
    return input_str

class NaiveMatMul:
    def __init__(self, context, queue):
        NaiveMatMul.context = context
        NaiveMatMul.queue = queue
        NaiveMatMul.program = cl.Program(context, utils.naive_matmul_program).build()     

    @staticmethod
    def Matmul(x_gpu, y_gpu):
        '''
        "Naive" Matrix Multiplication
        Parameters
        ----------
        x_gpu: (N, d) pyopencl.array
            can be C-contiguous or F-contiguous
        y_gpu: (d, k) pyopencl.array
            can be C-contiguous or F-contiguous
        Returns
        ----------
        (N, k) C-contiguous pyopencl.array
        '''
        global_size = (x_gpu.shape[0], y_gpu.shape[1])
        local_size = None

        _, input_width = x_gpu.shape
        matrix_out = cl_array.zeros(NaiveMatMul.queue, global_size, dtype=np.float32)
        inputs = (NaiveMatMul.queue, global_size, local_size, x_gpu.data, y_gpu.data, np.int32(input_width), matrix_out.data)

        if x_gpu.flags.c_contiguous and y_gpu.flags.c_contiguous:
            NaiveMatMul.program.naive_matmul(*inputs).wait()
            
        elif x_gpu.flags.c_contiguous and y_gpu.flags.f_contiguous:
            NaiveMatMul.program.naive_matmul_Fortran_Y(*inputs).wait()
            
        elif x_gpu.flags.f_contiguous and y_gpu.flags.c_contiguous:
            NaiveMatMul.program.naive_matmul_Fortran_X(*inputs).wait()
            
        elif x_gpu.flags.f_contiguous and y_gpu.flags.f_contiguous:
            NaiveMatMul.program.naive_matmul_Fortran(*inputs).wait()

        return matrix_out 
    
# Adapted from GPUTranspose in
# https://colab.research.google.com/drive/15yk8JbY-GadZhyUDyb1MLAokatYhJ0PQ?usp=sharing
class GPUTranspose:
    def __init__(self, context, queue):
        GPUTranspose.context = context
        GPUTranspose.queue = queue
        GPUTranspose.program = cl.Program(context, utils.transpose_program).build()

    @staticmethod    
    def transpose(x_gpu):
        '''
        Returns transpose of a matrix

        Parameters
        ----------
        x_gpu: (N, k) pyopencl.array

        Returns
        ----------
        (N, k) pyopencl.array
        '''
        width, height = x_gpu.shape

        # Make the new shape divisible by 16
        x_new_shape = int(16 * np.ceil(max(height, width) / 16))

        global_size = (x_new_shape, x_new_shape)
        local_size = (16, 16)
        
        # Fit input values into the new array divisible by 16
        x_new = np.zeros((x_new_shape, x_new_shape)).astype(np.float32)
        x_new[:width, :height] = x_gpu.get()
        x_new_gpu = cl_array.to_device(GPUTranspose.queue, x_new)

        x_transpose_filled = cl_array.zeros_like(x_new_gpu)
        a_local = cl.LocalMemory(4 * 16 * (16 + 1))
        
        GPUTranspose.program.transpose(GPUTranspose.queue, global_size, local_size, x_transpose_filled.data,
                               x_new_gpu.data, np.int32(x_new_shape), np.int32(x_new_shape), a_local).wait()
        
        # Romove redundant entries in the transposed matrix
        x_transpose = cl_array.zeros(GPUTranspose.queue, (height, width), dtype=np.float32)
        indices = cl_array.arange(GPUTranspose.queue, width, dtype=np.int32)
        
        for row in range(height):
            cl_array.take(x_transpose_filled[row], indices, x_transpose[row])

        return x_transpose

class ArrayFunctions:
    def __init__(self, context, queue):
        ArrayFunctions.context = context
        ArrayFunctions.queue = queue

        ArrayFunctions.program = cl.Program(context, utils.array_functions_program).build()

        ArrayFunctions.sign_program = ElementwiseKernel(ArrayFunctions.context,
                            "double *x, double *out",
                            "out[i] = sign(x[i])",
                            preamble='#define sign(x) (x > 0.0) ? 1.0 : -1.0'
                            )

        ArrayFunctions.clip_min_program = ElementwiseKernel(ArrayFunctions.context,
                                            "double *x, double min",
                                            "x[i] = x[i] > min ? x[i] : min",
                                            "clip_min"
                                            )
        ArrayFunctions.clip_max_program = ElementwiseKernel(ArrayFunctions.context,
                                            "double *x, double max",
                                            "x[i] = x[i] < max ? x[i] : max",
                                            "clip_max"
                                            )

    @staticmethod
    def addVector(x_gpu, b_gpu):
        '''
        Add a vector b to a matrix X
        '''
        global_size = x_gpu.shape
        local_size = None
        matrix_out = cl_array.zeros(ArrayFunctions.queue, global_size, dtype=np.float32)
        
        ArrayFunctions.program.add_vector(ArrayFunctions.queue, global_size, local_size, 
                               x_gpu.data, b_gpu.data, matrix_out.data).wait()
        return matrix_out 

    @staticmethod
    def mulVector(x_gpu, b_gpu):
        '''
        Multiply a matrix X by a vector b
        '''
        global_size = x_gpu.shape
        local_size = None
        matrix_out = cl_array.zeros(ArrayFunctions.queue, global_size, dtype=np.float32)
        
        ArrayFunctions.program.mul_vector(ArrayFunctions.queue, global_size, local_size, 
                               x_gpu.data, b_gpu.data, matrix_out.data).wait()
        return matrix_out 
    
    @staticmethod
    def divVector(x_gpu, b_gpu):
        '''
        Divide a matrix X by a vector b
        '''
        global_size = x_gpu.shape
        local_size = None
        matrix_out = cl_array.zeros(ArrayFunctions.queue, global_size, dtype=np.float32)
        
        ArrayFunctions.program.div_vector(ArrayFunctions.queue, global_size, local_size, 
                               x_gpu.data, b_gpu.data, matrix_out.data).wait()
        return matrix_out 
    
    @staticmethod
    def rowSumUp(x_gpu):
        '''
        Returns the sum of each row.
        Implementation of np.sum(x, axis=0) in OpenCL.
        '''
        global_size = (x_gpu.shape[1], )
        output_width, _ = x_gpu.shape
        local_size = None
        matrix_out = cl_array.zeros(ArrayFunctions.queue, global_size, dtype=np.float32)
        
        ArrayFunctions.program.row_sums(ArrayFunctions.queue, global_size, local_size, 
                               x_gpu.data, np.int32(output_width), matrix_out.data).wait()
        return matrix_out 
    
    @staticmethod
    def rowMean(x_gpu):
        '''
        Returns the mean of each row.
        Implementation of np.mean(x, axis=0) in OpenCL.
        '''
        global_size = (x_gpu.shape[1], )
        output_width, _ = x_gpu.shape
        local_size = None
        matrix_out = cl_array.zeros(ArrayFunctions.queue, global_size, dtype=np.float32)
        
        ArrayFunctions.program.row_means(ArrayFunctions.queue, global_size, local_size, 
                               x_gpu.data, np.int32(output_width), matrix_out.data).wait()
        return matrix_out 
    
    @staticmethod
    def rowStd(x_gpu):
        '''
        Returns the standard deviations of each row.
        Implementation of np.std(x, axis=0) in OpenCL.
        '''
        global_size = (x_gpu.shape[1], )
        output_width, _ = x_gpu.shape
        local_size = None
        matrix_out = cl_array.zeros(ArrayFunctions.queue, global_size, dtype=np.float32)
        
        mean = ArrayFunctions.rowMean(x_gpu)

        ArrayFunctions.program.row_vars(ArrayFunctions.queue, global_size, local_size, x_gpu.data,
                                        mean.data, np.int32(output_width), matrix_out.data).wait()
        return clmath.sqrt(matrix_out) 
    
    @staticmethod
    def rowArgmax(x_gpu):
        '''
        Returns the index of the maximum of each row.
        Implementation of np.argmax(x, axis=-1) in OpenCL.
        '''
        output_height = np.prod(x_gpu.shape[:-1])
        output_width = x_gpu.shape[-1]
        
        global_size = (output_height,)
        local_size = None
        
        matrix_out = cl_array.zeros(ArrayFunctions.queue, x_gpu.shape[:-1], dtype=np.int32)

        ArrayFunctions.program.row_argmaxs(ArrayFunctions.queue, global_size, local_size, 
                               x_gpu.data, np.int32(output_width), matrix_out.data).wait()
        return matrix_out
                
    @staticmethod
    def clarray_sign(x_gpu):
        '''
        Implementation of np.sign(x) in OpenCL.
        '''
        # Change the datatype to return a precise result 
        x_gpu_precise = x_gpu.astype(np.float64)
        out_precise = cl_array.zeros_like(x_gpu_precise)
        ArrayFunctions.sign_program(x_gpu_precise, out_precise).wait()
        out = out_precise.astype(np.float32)
        return out
    
    @staticmethod
    def clarray_clip(x_gpu, min, max):
        '''
        Implementation of np.clip(x, min, max) in OpenCL.
        '''
        x_gpu_precise = x_gpu.astype(np.float64)

        if min is not None:
            ArrayFunctions.clip_min_program(x_gpu_precise, min).wait()
            return x_gpu_precise.astype(np.float32)
        if max is not None:
            ArrayFunctions.clip_max_program(x_gpu_precise, max).wait()
            return x_gpu_precise.astype(np.float32)
            