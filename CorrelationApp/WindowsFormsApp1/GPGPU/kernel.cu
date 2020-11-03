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




float* gpgpu_spearman(float** rank_x, float** rank_y, int n, int signal_count)
{
    dim3 gridSize = 256;
    dim3 blockSize = 256;

    float *h_prod;
    

    h_prod = (float*)malloc(signal_count * signal_count * sizeof(float));   

    float *d_prod;
    float *d_x, *d_y;
    cudaMalloc((void**)&d_prod, sizeof(float));
    cudaMalloc((void**)&d_x, n*sizeof(float));
    cudaMalloc((void**)&d_y, n*sizeof(float));
        
        
    for(int i = 0; i < signal_count; i++)
    {
        for (int j = 0; j < signal_count; j++)
        {
            cudaMemset(d_prod, 0.0, sizeof(float));
            cudaMemcpy(d_x, rank_x[i], n*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, rank_x[j], n*sizeof(float), cudaMemcpyHostToDevice);
            
            dot_product_kernel<<<gridSize, blockSize>>>(d_x, d_y, d_prod, n);
            
            cudaMemcpy((void*)&(h_prod[signal_count * i + j]), d_prod, sizeof(float), cudaMemcpyDeviceToHost);
            h_prod[signal_count * i + j] = 1.0f - (6.0f * h_prod[signal_count * i + j]) / (n*n*n - n);
        }
    }


    free(h_prod);

	cudaFree(d_prod);
	cudaFree(d_x);
    cudaFree(d_y);
    
    return h_prod;
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


    float* val = gpgpu_spearman(h_x, h_x, n, signal_count);

    for(int i = 0; i < signal_count; i++)
    {
        for (int j = 0; j < signal_count; j++)
        {
            std::cout << val[i * signal_count + j] << "\t\t";
        }
        std::cout << std::endl;
    }
	
	free(h_x);
    system("pause");

}