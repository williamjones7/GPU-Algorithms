# C kernel for naive calculations
naive_matmul_program = """
//naive matrix multiplication X @ Y for both C-contiguous matrices
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

//naive matrix multiplication X @ Y for F-contiguous X and C-contiguous Y
__kernel void naive_matmul_Fortran_X(__global float *X, __global float *Y, int input_width, __global float *out){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int output_height = get_global_size(0);
    int output_width = get_global_size(1);
    double product = 0.0;
    
    for (int k = 0; k < input_width; k++){
        product += X[k * output_height + i] * Y[k * output_width + j];
    }
    out[i * output_width + j] = product;
}

//naive matrix multiplication X @ Y for C-contiguous X and F-contiguous Y
__kernel void naive_matmul_Fortran_Y(__global float *X, __global float *Y, int input_width, __global float *out){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int output_width = get_global_size(1);
    double product = 0.0;
    
    for (int k = 0; k < input_width; k++){
        product += X[i * input_width + k] * Y[j * input_width + k];
    }
    out[i * output_width + j] = product;
}

//naive matrix multiplication X @ Y for both F-contiguous matrices
__kernel void naive_matmul_Fortran(__global float *X, __global float *Y, int input_width, __global float *out){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int output_height = get_global_size(0);
    int output_width = get_global_size(1);
    double product = 0.0;
    
    for (int k = 0; k < input_width; k++){
        product += X[k * output_height + i] * Y[j * input_width + k];
    }
    out[i * output_width + j] = product;
}
"""

# C kernel for transpose
transpose_program = """
// Return the transpose of a square matrix with dimension divisible by 16
// Taken from 
// https://colab.research.google.com/drive/15yk8JbY-GadZhyUDyb1MLAokatYhJ0PQ?usp=sharing

__kernel void transpose(__global float *a_t, __global float *a, int width, int height){
    int global_col = get_global_id(1);
    int global_row = get_global_id(0);


    a_t[global_row + global_col * width] = a[global_row * height + global_col];
}
"""

# softmax program
softmax_program = '''
__kernel void softmax(__global const double *x, __global const double *max, const int num_cols, __global double *out){
    int i = get_global_id(0);
    int j = get_global_id(1);

    double idx_max = max[i];
    double total = 0.0;

    for (int k = 0; k < num_cols; k++){
        total += exp((double) x[i*num_cols + k] - idx_max);
    }

    out[i*num_cols + j] = exp(x[i*num_cols + j] - idx_max) / total;
}
'''

# C kernel for calculations between a matrix and a vector
array_functions_program = """
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
__kernel void row_sums(__global float *x, int num_rows, __global float *sums_out){
    int i = get_global_id(0);
    int num_cols = get_global_size(0);
    double sum = 0.0;
    for (int k = 0; k < num_rows; k++){
        sum += x[k * num_cols + i];
    }
    sums_out[i] = sum;
}

//returns the mean of each row
__kernel void row_means(__global float *x, int num_rows, __global float *means_out){
    int i = get_global_id(0);
    int num_cols = get_global_size(0);
    double row_mean = 0.0;
    for (int k = 0; k < num_rows; k++){
        row_mean += x[k * num_cols + i] / (double) num_rows;
    }
    means_out[i] = row_mean;
}

//returns the variance of each row
__kernel void row_vars(__global float *x, __global float *mean, int num_rows, __global float *vars_out){
    int i = get_global_id(0);
    int num_cols = get_global_size(0);

    double row_var = 0.0;
    double row_derivation = 0.0;
    double row_mean = mean[i];

    for (int k = 0; k < num_rows; k++){
        row_derivation = x[k * num_cols + i] - row_mean;
        row_var += row_derivation * row_derivation / (double) num_rows;
    }
    vars_out[i] = row_var;
}

//returns the max of each row
__kernel void row_maxs(__global float *x, int num_cols, __global int *max_out){
    int i = get_global_id(0);
    double max = x[i * num_cols];
    double current = 0.0;

    for (int k = 1; k < num_cols; k++){
        current = x[i * num_cols + k];
        if (current > max){
            max=current;
        }
        max_out[i] = max;
    }
}

//returns the argmax of each row
__kernel void row_argmaxs(__global float *x, int num_cols, __global int *argmax_out){
    int i = get_global_id(0);
    int idx_max = 0;
    double current = 0.0;
    double max = x[i * num_cols];

    for (int k = 1; k < num_cols; k++){
        current = x[i * num_cols + k];

        if (current > max) {
            max = current;
            idx_max = k;
        }
    }
    argmax_out[i] = idx_max;
}
"""


fast_matmul_program = """
// Thread block size
#define BLOCK_SIZE %(block_size)d
// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA %(w_a)d // Matrix A width
#define HA %(h_a)d // Matrix A height
#define WB %(w_b)d // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width
#define HC HA  // Matrix C height

__kernel __attribute__((reqd_work_group_size(16,16,1))) 
void
tiled_matmul( __global float* C, __global float* A, __global float* B)
{
    __local float As[BLOCK_SIZE*BLOCK_SIZE];
    __local float Bs[BLOCK_SIZE*BLOCK_SIZE];
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);
    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    // Index of the first sub-matrix of A processed by the block
    int aBegin = WA * BLOCK_SIZE * by;
    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + WA - 1;
    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;
    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;
    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * WB;
    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0.0f;
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[tx + ty * (BLOCK_SIZE)] = A[a + WA * ty + tx];
        Bs[tx + ty * (BLOCK_SIZE)] = B[b + WB * ty + tx];
        // Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[k + ty * (BLOCK_SIZE)] * Bs[tx + k * (BLOCK_SIZE)];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = Csub;
}
"""


fast_matmul_program_original = """
// Thread block size
#define BLOCK_SIZE %(block_size)d
// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA %(w_a)d // Matrix A width
#define HA %(h_a)d // Matrix A height
#define WB %(w_b)d // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width
#define HC HA  // Matrix C height

#define AS(j, i) As[i + j * BLOCK_SIZE]
#define BS(j, i) Bs[i + j * BLOCK_SIZE]

__kernel __attribute__((reqd_work_group_size(16,16,1))) 
void
tiled_matmul( __global float* C, __global float* A, __global float* B)
{
    __local float As[BLOCK_SIZE*BLOCK_SIZE];
    __local float Bs[BLOCK_SIZE*BLOCK_SIZE];
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);
    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    // Index of the first sub-matrix of A processed by the block
    int aBegin = WA * BLOCK_SIZE * by;
    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + WA - 1;
    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;
    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;
    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * WB;
    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0.0f;
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + WA * ty + tx];
        BS(ty, tx) = B[b + WB * ty + tx];
        // Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = Csub;
}
"""

