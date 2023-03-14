import pyopencl as cl
import pyopencl.array as cl_array

import numpy as np

def buffer_str(input_str, max_buffer=30):
    if len(input_str) < max_buffer:
        return input_str + ' ' * (max_buffer - len(input_str))
    return input_str


single_layer_c_code = """
__kernel void vecmul(__global float *W, __global float *z, __global float *b, int input_index, __global float *out){
    int j = get_global_id(0);
    out[j] = 0.0;
    
    for (int k = 0; k < input_index; k++){
        out[j] += z[k] * W[j * input_index + k];
    }
    out[j] += b[j];
}

__kernel void matmul(__global float *W, __global float *Z, __global float *b, int input_index, __global float *out){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int output_index = get_global_size(1);
    out[i * output_index + j] = 0.0;
    
    for (int k = 0; k < input_index; k++){
        out[i * output_index + j] += Z[i * input_index + k] * W[j * input_index + k];
    }
    out[i * output_index + j] += b[j];
}
"""

# Clean up
class SingleLayer:
    """ Returns the output of this layer

    Parameters
    ----------
    z : (d, m) np.array
        z is assumed to be a list of all the inputs to be forward propagated. In particular
        it is assumed that the first index of z is the index that inputs is accessed by.
        z is assumed to be sent to the device.
    W : (n, m) np.array
        W is assumed to be a list of all the weights.
        W is assumed to be sent to the device.
    b : (n, ) np.array
        b is assumed to be a vector of biases.
        b is assumed to be sent to the device.

    Returns
    -------
    (d, n) np.array
        The final output of the layer, post activation
        """
    def __init__(self, context, queue):
        self.context = context
        self.queue = queue
        self.program = cl.Program(context, single_layer_c_code).build()

    # Computation
    def matmul(self, W, Z, b):
        global_size = (Z.shape[0], b.shape[0])
        local_size = None

        _, input_index = W.shape
        matrix_out = cl_array.zeros(self.queue, global_size, dtype=np.float32)
        
        self.program.matmul(self.queue, global_size, local_size, 
                            W.data, Z.data, b.data, np.int32(input_index), matrix_out.data).wait()
        return matrix_out