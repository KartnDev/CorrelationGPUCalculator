#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>
#include <time.h>

__global__ void dot_product_kernel(float *x, float *y, float *num, float *denom, unsigned int n)
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

    if(threadIdx.x == 0)
    {
        avg_x[0] = avg_x[0] / n;
        avg_y[0] = avg_y[0] / n;
	}

    // calculation denominator x error sum
    index = threadIdx.x + blockDim.x*blockIdx.x;
    stride = blockDim.x*gridDim.x;
    
    __shared__ float sum_error_x[256]; 
    __shared__ float sum_error_y[256]; 

    temp_x = 0.0;
    temp_y = 0.0;

    while(index < n)
    {
        temp_x += (x[index] - avg_x[0]) * (x[index] - avg_x[0]);
        temp_y += (y[index] - avg_y[0]) * (y[index] - avg_y[0]);

	    index += stride;
    }
    sum_error_x[threadIdx.x] = temp_x;
    sum_error_y[threadIdx.x] = temp_y;

    __syncthreads();

    i = blockDim.x/2;
    while(i != 0)
    {
        if(threadIdx.x < i)
        {
            sum_error_x[threadIdx.x] += sum_error_x[threadIdx.x + i];
            sum_error_y[threadIdx.x] += sum_error_y[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }


    // final reduction
    index = threadIdx.x + blockDim.x*blockIdx.x;
	stride = blockDim.x*gridDim.x;
    
	__shared__ float cache[256];

	double temp = 0.0;
    while(index < n)
    {
	    temp += (x[index] - avg_x[0]) * (y[index] - avg_y[0]);
	    index += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

	// reduction
	i = blockDim.x/2;
    while(i != 0)
    {
        if(threadIdx.x < i)
        {
	        cache[threadIdx.x] += cache[threadIdx.x + i];
	    }
	    __syncthreads();
	    i /= 2;
    }
    

    if(threadIdx.x == 0)
    {
        atomicAdd(denom, sqrtf(sum_error_x[0] * sum_error_y[0]));
        atomicAdd(num, cache[0]);
	}
}

float* gpgpu_mat_correlation(float** host_vectors, int count_of_vector, int n)
{
    dim3 gridSize = 256;
    dim3 blockSize = 256;

    float *result = (float*)malloc(count_of_vector * count_of_vector * sizeof(float));   
    float *temp = (float*)malloc(sizeof(float));

    float *d_prod_num, *d_prod_denom;
    float *d_x, *d_y;


    cudaMalloc((void**)&d_prod_num, sizeof(float));
    cudaMalloc((void**)&d_prod_denom, sizeof(float));
    cudaMalloc((void**)&d_x, n*sizeof(float));
    cudaMalloc((void**)&d_y, n*sizeof(float));
      
    
    for(int i =0; i < count_of_vector; i++)
    {
        for(int j =0; j < count_of_vector; j++)
        {
            cudaMemset(d_prod_num, 0.0, sizeof(float));
            cudaMemset(d_prod_denom, 0.0, sizeof(float));
            cudaMemcpy(d_x, (void*)host_vectors[i], n*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, (void*)host_vectors[j], n*sizeof(float), cudaMemcpyHostToDevice);
        
        
            dot_product_kernel<<<gridSize, blockSize>>>(d_x, d_y, d_prod_num, d_prod_denom, n);
            cudaMemcpy(&result[i * count_of_vector + j], d_prod_num, sizeof(float), cudaMemcpyDeviceToHost);
            
            cudaMemcpy(temp, d_prod_denom, sizeof(float), cudaMemcpyDeviceToHost);
            result[i * count_of_vector + j] *= powf(*temp, -1);
        }
    }
    
    
    return result;
}



int main()
{
    srand( (unsigned)time(NULL) );
    unsigned int n = 1 * 256 * 256;
    int signal_count = 2;
        

    float** h_x = (float**)malloc(signal_count*sizeof(float*));
    
    for(int i = 0; i < signal_count; i++)
    {
        h_x[i] = (float*)malloc(n * sizeof(float));
    }

    for(int i = 0; i < signal_count; i++)
    {
        for(int j = 0; j < n; j++)
        {
            h_x[i][j] = float(rand()%n) / n;
        }
    }

    
    float* res = gpgpu_mat_correlation(h_x, signal_count, n);
    
    for(int i =0; i < signal_count; i++)
    {
        for(int j =0; j < signal_count; j++)
        {
            std::cout << res[i * signal_count + j] << "\t\t";
        }
        std::cout << "\n";
    }
    
    system("pause");
}