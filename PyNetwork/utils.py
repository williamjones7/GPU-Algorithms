import pyopencl as cl
import pyopencl.array as cl_array

import numpy as np

from pyopencl.elementwise import ElementwiseKernel

def buffer_str(input_str, max_buffer=30):
    if len(input_str) < max_buffer:
        return input_str + ' ' * (max_buffer - len(input_str))
    return input_str

# C kernel for naive calculations
naive_program = """
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

//returns the sum of each columns

__kernel void column_sums(__global float *delta, int output_width, __global float *bias_out){
    int i = get_global_id(0);
    int input_width = get_global_size(0);
    bias_out[i] = 0.0;
    for (int k = 0; k < output_width; k++){
        bias_out[i] += delta[k * input_width + i];
    }
}
"""

class SingleLayer:
    def __init__(self, context, queue):
        self.context = context
        self.queue = queue
        self.program = cl.Program(context, naive_program).build()

    def naiveMatmul(self, X, Y):
        '''
        "Naive" Matrix Multiplication
        '''
        global_size = (X.shape[0], Y.shape[1])
        local_size = None

        _, input_width = X.shape
        matrix_out = cl_array.zeros(self.queue, global_size, dtype=np.float32)
        
        self.program.naive_matmul(self.queue, global_size, local_size, 
                               X.data, Y.data, np.int32(input_width), matrix_out.data).wait()
        return matrix_out

    def addVector(self, X, b):
        '''
        Add a vector b to a matrix X
        '''
        global_size = X.shape
        local_size = None
        matrix_out = cl_array.zeros(self.queue, global_size, dtype=np.float32)
        
        self.program.add_vector(self.queue, global_size, local_size, 
                               X.data, b.data, matrix_out.data).wait()
        return matrix_out 

    def columnSumUp(self, X):
        '''
        Returns the sum of each column
        '''
        global_size = (X.shape[1], )
        output_width, _ = X.shape
        local_size = None
        matrix_out = cl_array.zeros(self.queue, global_size, dtype=np.float32)
        
        self.program.column_sums(self.queue, global_size, local_size, 
                               X.data, np.int32(output_width), matrix_out.data).wait()
        return matrix_out 

    # Implementation of np.sign in OpenCL
    def sign(self, x):
        '''
        Implementation of np.sign in OpenCL.
        Only works for 64-bit float number or number with higher precision.
        '''
        sign_program = ElementwiseKernel(self.context,
                                    "double *x, double *out",
                                    "out[i] = sign(x[i])",
                                    preamble='#define sign(x) (x > 0) ? 1 : -1'
                                    )
        x_gpu = cl_array.to_device(self.queue, x)

        out = cl_array.zeros_like(x_gpu)
        sign_program(x_gpu, out)
        return out