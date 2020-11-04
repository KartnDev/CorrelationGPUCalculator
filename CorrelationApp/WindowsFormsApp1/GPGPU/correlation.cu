#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <time.h>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <vector>
extern "C" {
float* gpgpu_correlation_mat(float** signals, int n, int signal_count);
void SplitByBatches(float** currentShiftSignals, int n, int signalCount, int shiftWidth, int batchSize);
void shift_compute(float** fullSignals, int n, int signalCount, int shiftWidth, int batchSize);



    __global__ void correlation(float *x, float *y, float *num, float *denom, unsigned int n, float avg_x, float avg_y)
    {
        unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;
        unsigned int stride = blockDim.x*gridDim.x;

        __shared__ float sum_x[256];
        __shared__ float sum_y[256];
        __shared__ float sum_pairwise[256];


        double temp_x = 0.0;
        double temp_y = 0.0;
        double temp_pairwise = 0.0;

        while(index < n)
        {
            temp_x += (x[index] - avg_x) * (x[index] - avg_x);
            temp_y += (y[index] - avg_y) * (y[index] - avg_y);
            temp_pairwise += (x[index] - avg_x) * (y[index] - avg_y);

            index += stride;
        }

        sum_x[threadIdx.x] = temp_x;
        sum_y[threadIdx.x] = temp_y;
        sum_pairwise[threadIdx.x] = temp_pairwise;

        __syncthreads();

        // reduction
        unsigned int i = blockDim.x/2;
        while(i != 0)
        {
            if(threadIdx.x < i)
            {
                sum_x[threadIdx.x] += sum_x[threadIdx.x + i];
                sum_y[threadIdx.x] += sum_y[threadIdx.x + i];
                sum_pairwise[threadIdx.x] += sum_pairwise[threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }


        if(threadIdx.x == 0)
        {
            atomicAdd(denom, sqrtf(sum_x[0] * sum_y[0]));
            atomicAdd(num, sum_pairwise[0]);
        }
    }

    double mean(float* data, int n)
    {
        double sum = 0;
        #pragma omp parallel for schedule(static) reduction (+:sum)
        for(int i = 0; i < n; i++)
        {
            sum += data[i];
        }
        return sum / n;
    }



    void SplitByBatches(float** currentShiftSignals, int n, int signalCount, int shiftWidth, int batchSize)
    {
        float** batch = (float**)malloc(signalCount*sizeof(float*));
        for(int k = 0; k < signalCount; k++)
        {
            batch[k] = (float*)malloc(batchSize * sizeof(float));
        }
        

        for (int batchIndex = 0; batchIndex < n - batchSize; batchIndex += batchSize)
        {
            printf("Batch %i\n", batchIndex);
            for(int k = 0; k < signalCount; k++)
            {
                for(int j = 0; j < batchSize; j++)
                {
                    batch[k][j] = currentShiftSignals[k][j + batchIndex];
                }
            }

            float * result = gpgpu_correlation_mat(batch, batchSize, signalCount);

            for(int i = 0; i < signalCount; i++)
            {
                for(int j = 0; j < signalCount; j++)
                {   
                    printf("%.2f\t|\t", result[i * signalCount + j]);
                }
                std::cout << "\n";
            }
           
        }
        //freeing
        for(int k = 0; k < signalCount; k++)
        {
            free(batch[k]);
        }
        free(batch);
    }


    float* gpgpu_correlation_mat(float** signals, int n, int signal_count)
    {
        dim3 gridSize = 256;
        dim3 blockSize = 256;

        float* result = (float*)malloc(signal_count * signal_count * sizeof(float));

        float *denom_res = (float*)malloc(sizeof(float));   
        float *num_res = (float*)malloc(sizeof(float));

        float *d_prod_num, *d_prod_denom;
        float *d_x, *d_y;


        cudaMalloc((void**)&d_prod_num, sizeof(float));
        cudaMalloc((void**)&d_prod_denom, sizeof(float));
        cudaMalloc((void**)&d_x, n*sizeof(float));
        cudaMalloc((void**)&d_y, n*sizeof(float));

        float x_mean, y_mean;

        for(int i = 0; i < signal_count; i++)
        {
            for(int j = 0; j < signal_count; j++)
            {   
                cudaMemset(d_prod_num, 0.0f, sizeof(float));
                cudaMemset(d_prod_denom, 0.0f, sizeof(float));
                cudaMemcpyAsync(d_x, (void*)signals[i], n*sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpyAsync(d_y, (void*)signals[j], n*sizeof(float), cudaMemcpyHostToDevice);

                x_mean = mean(signals[i], n);
                y_mean = mean(signals[j], n);

                correlation<<<gridSize, blockSize>>>(d_x, d_y, d_prod_num, d_prod_denom, n, x_mean, y_mean);
                cudaMemcpyAsync(num_res, d_prod_num, sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpyAsync(denom_res, d_prod_denom, sizeof(float), cudaMemcpyDeviceToHost);

                result[i * signal_count + j] = (*num_res) / (*denom_res);
            }
        }

        cudaFree(d_prod_num);
        cudaFree(d_prod_denom);
        cudaFree(d_x);
        cudaFree(d_y);

        free(denom_res);
        free(num_res);

        return result;
    }
}




int main(int argc, char** argv)
{
    std::ifstream f;
       
    f.open ("ClosedEyes.asc");

    std::string line, val;                  
    std::vector<std::vector<float>> array;    

    while (std::getline (f, line)) {      
    std::vector<float> v;                 
    std::stringstream s (line);         
    while (getline (s, val, ' '))       
        v.push_back (std::stof (val));  
    array.push_back (v);                
    }

    unsigned int n = array.size();
    int signal_count = array[0].size();
    

    float** h_x = (float**)malloc(signal_count*sizeof(float*));

    for(int i = 0; i < signal_count; i++)
    {
        h_x[i] = (float*)malloc(n * sizeof(float));
    }

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < signal_count; j++)
        {
            h_x[j][i] = array[i][j];
        }
    }


    SplitByBatches(h_x, n, signal_count, 100, 100);


    system("pause");

}