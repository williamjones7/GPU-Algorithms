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