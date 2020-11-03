#include "cuda.h"
#include "cuda_runtime.h"
#include <math.h>
#include <iostream>

__global__ void dot_product_kernel(float *x, float *y, float *dot, unsigned int n)
{
	unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int stride = blockDim.x*gridDim.x;

    // Calculating avg of vectors
    __shared__ float avg_x[256]; 
    __shared__ float avg_y[256]; 

    double temp_x = 0.0;
    double temp_y = 0.0;

    while(index < n)
    {
        temp_x += x[index];
        temp_y += y[index];

		index += stride;
    }
    avg_x[threadIdx.x] = temp_x;
    avg_y[threadIdx.x] = temp_y;

    __syncthreads();

    unsigned int i = blockDim.x/2;
    while(i != 0)
    {
        if(threadIdx.x < i)
        {
            avg_x[threadIdx.x] += avg_x[threadIdx.x + i];
            avg_y[threadIdx.x] += avg_y[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    // // calculation denominator x error sum
    // index = threadIdx.x + blockDim.x*blockIdx.x;
    // stride = blockDim.x*gridDim.x;
    
    // __shared__ float sum_error_x[256]; 
    // __shared__ float sum_error_y[256]; 

    // temp_x = 0.0;
    // temp_y = 0.0;

    // while(index < n)
    // {
    //     temp_x += (x[index] - avg_x[0]) * (x[index] - avg_x[0]);
    //     temp_y += (y[index] - avg_y[0]) * (y[index] - avg_y[0]);

	// 	index += stride;
    // }
    // sum_error_x[threadIdx.x] = temp_x;
    // sum_error_y[threadIdx.x] = temp_y;

    // __syncthreads();

    // i = blockDim.x/2;
    // while(i != 0)
    // {
    //     if(threadIdx.x < i)
    //     {
    //         sum_error_x[threadIdx.x] += sum_error_x[threadIdx.x + i];
    //         sum_error_y[threadIdx.x] += sum_error_y[threadIdx.x + i];
    //     }
    //     __syncthreads();
    //     i /= 2;
    // }


    // // final reduction
    // index = threadIdx.x + blockDim.x*blockIdx.x;
	// stride = blockDim.x*gridDim.x;
    
	// __shared__ float cache[256];

	// double temp = 0.0;
    // while(index < n)
    // {
	// 	temp += (x[index] - avg_x[0]) * (y[index] - avg_y[0]);
	// 	index += stride;
	// }

	// cache[threadIdx.x] = temp;

	// __syncthreads();

	// // reduction
	// i = blockDim.x/2;
    // while(i != 0)
    // {
    //     if(threadIdx.x < i)
    //     {
	// 		cache[threadIdx.x] += cache[threadIdx.x + i];
	// 	}
	// 	__syncthreads();
	// 	i /= 2;
    // }
    
    if(threadIdx.x == 0)
    {
        //cache[0] /= sqrtf(sum_error_x[0] * sum_error_y[0]);


		atomicAdd(dot, avg_x[0]);
	}
}

int main()
{

    unsigned int n = 10;
    int signal_count = 2;
        

    float** h_x = (float**)malloc(signal_count*sizeof(float*));
    
    for(int i = 0; i < signal_count; i++)
    {
        h_x[i] = (float*)malloc(n * sizeof(float));
    }


    
    // fill host array with data
    // for(int i = 0; i < signal_count; i++)
    // {
    //     for(int j = 0; j < n; j++)
    //     {
    //         h_x[i][j] = (float)(rand()%n) / n;
    //     }
    // }

    h_x[0][0] = 56;  h_x[1][0] = 66;
    h_x[0][1] = 75;  h_x[1][1] = 70; 
    h_x[0][2] = 45;  h_x[1][2] = 40; 
    h_x[0][3] = 71;  h_x[1][3] = 60; 
    h_x[0][4] = 61;  h_x[1][4] = 65; 
    h_x[0][5] = 64;  h_x[1][5] = 56; 
    h_x[0][6] = 58;  h_x[1][6] = 59; 
    h_x[0][7] = 80;  h_x[1][7] = 77; 
    h_x[0][8] = 76;  h_x[1][8] = 67;  
    h_x[0][9] = 61;  h_x[1][9] = 63; 	

    dim3 gridSize = 256;
    dim3 blockSize = 256;

    float *h_prod = (float*)malloc(sizeof(float));   

    float *d_prod;
    float *d_x, *d_y;


    cudaMalloc((void**)&d_prod, sizeof(float));
    cudaMalloc((void**)&d_x, n*sizeof(float));
    cudaMalloc((void**)&d_y, n*sizeof(float));
        
    
    cudaMemset(d_prod, 0.0, sizeof(float));
    cudaMemcpy(d_x, h_x[0], n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_x[1], n*sizeof(float), cudaMemcpyHostToDevice);
    
    dot_product_kernel<<<gridSize, blockSize>>>(d_x, d_y, d_prod, n);
    
    cudaMemcpy(h_prod, d_prod, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << *h_prod << std::endl;
    system("pause");
}