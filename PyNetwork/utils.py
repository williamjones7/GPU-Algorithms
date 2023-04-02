import pyopencl as cl
import pyopencl.array as cl_array

import numpy as np

from pyopencl.elementwise import ElementwiseKernel
from pyopencl import clmath

def buffer_str(input_str, max_buffer=30):
    if len(input_str) < max_buffer:
        return input_str + ' ' * (max_buffer - len(input_str))
    return input_str

# C kernel for naive calculations
array_functions_program = """
//naive matrix multiplication
__kernel void naive_matmul(__global float *X, __global float *Y, int input_width, __global float *out){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int output_width = get_global_size(1);
    double product = 0.0;
    
    for (int k = 0; k < input_width; k++){
        product += X[i * input_width + k] * Y[k * output_width + j];
    }
    out[i * output_width + j] = product;
}

//add a vector b to a matrix X
__kernel void add_vector(__global float *X, __global float *b, __global float *out){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int output_width = get_global_size(1);
    int index = i * output_width + j;
    double sum = 0.0;
    sum = X[index] + b[j];
    out[index] = sum;
}

//multiply a matrix X by a vector b
__kernel void mul_vector(__global float *X, __global float *b, __global float *out){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int output_width = get_global_size(1);
    int index = i * output_width + j;
    double prod = 0.0;
    prod = X[index] * b[j];
    out[index] = prod;
}

//divide a matrix X by a vector b
__kernel void div_vector(__global float *X, __global float *b, __global float *out){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int output_width = get_global_size(1);
    int index = i * output_width + j;
    double div = 0.0;
    div = X[index] / (double) b[j];
    out[index] = div;
}

//returns the sum of each row
__kernel void row_sums(__global float *delta, int num_rows, __global float *sums_out){
    int i = get_global_id(0);
    int num_cols = get_global_size(0);
    double sum = 0.0;
    for (int k = 0; k < num_rows; k++){
        sum += delta[k * num_cols + i];
    }
    sums_out[i] = sum;
}

//returns the mean of each row
__kernel void row_means(__global float *delta, int num_rows, __global float *means_out){
    int i = get_global_id(0);
    int num_cols = get_global_size(0);
    double row_mean = 0.0;
    for (int k = 0; k < num_rows; k++){
        row_mean += delta[k * num_cols + i] / (double) num_rows;
    }
    means_out[i] = row_mean;
}

//returns the variance of each row
__kernel void row_vars(__global float *delta, __global float *mean, int num_rows, __global float *vars_out){
    int i = get_global_id(0);
    int num_cols = get_global_size(0);

    double row_var = 0.0;
    double row_derivation = 0.0;
    double row_mean = mean[i];

    for (int k = 0; k < num_rows; k++){
        row_derivation = delta[k * num_cols + i] - row_mean;
        row_var += row_derivation * row_derivation / (double) num_rows;
    }
    vars_out[i] = row_var;
}
"""

class ArrayMathsFunction:
    def __init__(self, context, queue):
        ArrayMathsFunction.context = context
        ArrayMathsFunction.queue = queue
        ArrayMathsFunction.program = cl.Program(context, array_functions_program).build()
    
    @staticmethod
    def naiveMatmul(x_gpu, y_gpu):
        '''
        "Naive" Matrix Multiplication
        '''
        global_size = (x_gpu.shape[0], y_gpu.shape[1])
        local_size = None

        _, input_width = x_gpu.shape
        matrix_out = cl_array.zeros(ArrayMathsFunction.queue, global_size, dtype=np.float32)
        
        ArrayMathsFunction.program.naive_matmul(ArrayMathsFunction.queue, global_size, local_size, 
                               x_gpu.data, y_gpu.data, np.int32(input_width), matrix_out.data).wait()
        return matrix_out

    @staticmethod
    def addVector(x_gpu, b_gpu):
        '''
        Add a vector b to a matrix X
        '''
        global_size = x_gpu.shape
        local_size = None
        matrix_out = cl_array.zeros(ArrayMathsFunction.queue, global_size, dtype=np.float32)
        
        ArrayMathsFunction.program.add_vector(ArrayMathsFunction.queue, global_size, local_size, 
                               x_gpu.data, b_gpu.data, matrix_out.data).wait()
        return matrix_out 

    @staticmethod
    def mulVector(x_gpu, b_gpu):
        '''
        Add a vector b to a matrix X
        '''
        global_size = x_gpu.shape
        local_size = None
        matrix_out = cl_array.zeros(ArrayMathsFunction.queue, global_size, dtype=np.float32)
        
        ArrayMathsFunction.program.mul_vector(ArrayMathsFunction.queue, global_size, local_size, 
                               x_gpu.data, b_gpu.data, matrix_out.data).wait()
        return matrix_out 
    
    @staticmethod
    def divVector(x_gpu, b_gpu):
        '''
        Add a vector b to a matrix X
        '''
        global_size = x_gpu.shape
        local_size = None
        matrix_out = cl_array.zeros(ArrayMathsFunction.queue, global_size, dtype=np.float32)
        
        ArrayMathsFunction.program.div_vector(ArrayMathsFunction.queue, global_size, local_size, 
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
        matrix_out = cl_array.zeros(ArrayMathsFunction.queue, global_size, dtype=np.float32)
        
        ArrayMathsFunction.program.row_sums(ArrayMathsFunction.queue, global_size, local_size, 
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
        matrix_out = cl_array.zeros(ArrayMathsFunction.queue, global_size, dtype=np.float32)
        
        ArrayMathsFunction.program.row_means(ArrayMathsFunction.queue, global_size, local_size, 
                               x_gpu.data, np.int32(output_width), matrix_out.data).wait()
        return matrix_out 
    
    @staticmethod
    def rowStd(x_gpu):
        '''
        Returns the mean of each row.
        Implementation of np.std(x, axis=0) in OpenCL.
        '''
        global_size = (x_gpu.shape[1], )
        output_width, _ = x_gpu.shape
        local_size = None
        matrix_out = cl_array.zeros(ArrayMathsFunction.queue, global_size, dtype=np.float32)
        
        mean = ArrayMathsFunction.rowMean(x_gpu)

        ArrayMathsFunction.program.row_vars(ArrayMathsFunction.queue, global_size, local_size, 
                               x_gpu.data, mean.data, np.int32(output_width), matrix_out.data).wait()
        return clmath.sqrt(matrix_out) 
                
    @staticmethod
    def sign(x_gpu):
        '''
        Implementation of np.sign in OpenCL.
        '''
        sign_program = ElementwiseKernel(ArrayMathsFunction.context,
                                    "double *x, double *out",
                                    "out[i] = sign(x[i])",
                                    preamble='#define sign(x) (x > 0.0) ? 1.0 : -1.0'
                                    )
        # Change the datatype to return a precise result 
        x_gpu_precise = x_gpu.astype(np.float64)
        out_precise = cl_array.zeros_like(x_gpu_precise)
        sign_program(x_gpu_precise, out_precise).wait()
        out = out_precise.astype(np.float32)
        return out