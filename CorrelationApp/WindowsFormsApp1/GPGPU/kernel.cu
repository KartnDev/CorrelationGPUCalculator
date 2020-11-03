#include "cuda_runtime.h"
#include "cuda.h"

#include <iostream>

__global__ void dot_product_kernel(float *rank_x, float *y_rank, float *dot, unsigned int n)
{
	unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int stride = blockDim.x*gridDim.x;

	__shared__ float cache[256];

	double temp = 0.0;
    while(index < n)
    {
		temp += (rank_x[index] - y_rank[index]) * (rank_x[index] - y_rank[index]);

		index += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

	// reduction
	unsigned int i = blockDim.x/2;
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
		atomicAdd(dot, cache[0]);
	}
}

float gpgpu_spearman(float* rank_x, float* rank_y, int n)
{
    float *h_prod;
    float *d_prod;
    float *d_x, *d_y;

    h_prod = (float*)malloc(sizeof(float)); 

    cudaMalloc((void**)&d_prod, sizeof(float));
	cudaMalloc((void**)&d_x, n*sizeof(float));
	cudaMalloc((void**)&d_y, n*sizeof(float));
    cudaMemset(d_prod, 0.0, sizeof(float));
    
    dim3 gridSize = 256;
    dim3 blockSize = 256;
    
    dot_product_kernel<<<gridSize, blockSize>>>(d_x, d_y, d_prod, n);
    
    cudaMemcpy(h_prod, d_prod, sizeof(float), cudaMemcpyDeviceToHost);

    float res = *h_prod;

    free(h_prod);

	cudaFree(d_prod);
	cudaFree(d_x);
    cudaFree(d_y);
    
    return 1.0f - (6.0f * res) / (n*n*n - n);
}


int main()
{
	unsigned int n = 1000*256*256;
	
	float *h_x, *h_y;
	

	// allocate memory
	h_x = (float*)malloc(n*sizeof(float));
	h_y = (float*)malloc(n*sizeof(float));
	

	// fill host array with data
    for(unsigned int i=0;i<n;i++)
    {
		h_x[i] = float(rand()%n) / n;
		h_y[i] = float(rand()%n) / n;
	}

    float val = gpgpu_spearman(h_x, h_y, n);

    std::cout << val << std::endl;

	
	free(h_x);
    free(h_y);
    system("pause");

}