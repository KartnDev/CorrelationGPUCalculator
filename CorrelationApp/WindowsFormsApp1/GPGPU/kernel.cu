#include "cuda_runtime.h"
#include "cuda.h"

#include <iostream>


 __device__ float correlationCoefficient(float* X, int n, float* res)
{
    for (int i = 0; i < n; i++)
	{
		int r = 1, s = 1;

		// Count no of smaller elements 
		// in 0 to i-1 
		for (int j = 0; j < i; j++) 
		{
			if (X[j] < X[i]) r++;
			if (X[j] == X[i]) s++;
		}

		// Count no of smaller elements 
		// in i+1 to N-1 
		for (int j = i + 1; j < n; j++) 
		{
			if (X[j] < X[i]) r++;
			if (X[j] == X[i]) s++;
		}

		// Use Fractional Rank formula 
		// fractional_rank = r + (n-1)/2 
		res[i] = r + (s - 1) * 0.5;
	}
}


__global__ void window_slide_correlations(float*** signals, int n, int sig_count, int window_size, int window_step, float** result,
                                            int* active_signals, int main_signal)
{
    int block_grid_id = blockIdx.x;
    int steps = (int)n / window_step;

    // invoke all window count 
    if(block_grid_id < steps)
    {
        float** current_window_sig = signals[block_grid_id];

        __shared__ float* main_rank;

        // computing main signal rank
        if(threadIdx.x == 0)
        {
            cudaMalloc((void*)main_rank, n * sizeof(float));
            correlationCoefficient(current_window_sig[main_signal], n, main_rank);
        }
        __syncthreads();

        int threadId = threadIdx.x;

        if(threadId < sig_count)
        {
            float* rank_at_thread;
            cudaMalloc((void*)rank_at_thread, n * sizeof(float));
            correlationCoefficient(current_window_sig[threadId], n, rank_at_thread);
            
            float sum = 0.0f;
            for (int i = 0; i < n; i++)
            {
                sum += (rank_X[i] - rank_Y[i]) * (rank_X[i] - rank_Y[i]);
            }
            result[block_grid_id][threadId] =  1.0 - 6.0 * sum / (n * n * n - n);

        }
        

        
    }
    
}



int main()
{
    dim3 threadsPerBlock(2);
    dim3 numBlocks(20);
    window_slide_correlations<<<numBlocks, threadsPerBlock>>>(0, 10000, 3, 0, 1000, 1002);
    


    system("pause");

}