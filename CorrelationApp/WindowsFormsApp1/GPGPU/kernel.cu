#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <time.h>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <chrono>
#include <ctime> 
#include <windows.h>
#include <algorithm>

void pause_say(const char* msg)
{
	std::cout << msg << std::endl;
	system("pause");
}


 __device__ void correlationCoefficient(float* X, int n, float* res)
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


__global__ void window_slide_correlations(float*** signals, unsigned int n, int sig_count, int window_size, int window_step, float** result,
                                            int* active_signals, unsigned int active_count, int main_signal)
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
            main_rank = (float*)malloc(n * sizeof(float));
            correlationCoefficient(current_window_sig[main_signal], n, main_rank);
        }
        __syncthreads();

        int threadId = threadIdx.x;

        if(threadId < sig_count)
        {
            float* rank_at_thread;
            rank_at_thread = (float*)malloc(n * sizeof(float));
            correlationCoefficient(current_window_sig[threadId], n, rank_at_thread);
            
            float sum = 0.0f;
            for (int i = 0; i < n; i++)
            {
                sum += (main_rank[i] - rank_at_thread[i]) * (main_rank[i] - rank_at_thread[i]);
            }
            result[block_grid_id][threadId] =  1.0 - 6.0 * sum / (n * n * n - n);
        }
        
    }
    
}

void circularShift(float* a, int size, int shift)
{
	assert(size >= shift);
	std::rotate(a, a + size - shift, a + size);
}


float** split_by_windows(float* signal, int window_size, int window_step, int n)
{
    int window_count = (int)(n - window_size) / window_step;

    float** allocated_res = (float**)malloc(window_count * sizeof(float*));
    for(int i = 0; i < window_count; i++)
    {
        allocated_res[i] = (float*)malloc(window_size * sizeof(float));
    }

    for(int curr_win = 0; curr_win < window_count; curr_win++)
    {
        for(int j = 0; j < window_size; j++)
        {
            allocated_res[curr_win][j] = signal[j + curr_win * window_step];
        }
    }
    return allocated_res;
}


float** get_correlations_shift(float** signals, int signals_count, int n, int window_size, int window_step, int* active_signals, unsigned int active_count, int main_signal)
{
    float*** splitted_by_windows = (float***)malloc(signals_count * sizeof(float**));

    for(int curr_sig = 0; curr_sig < signals_count; curr_sig++)
    {
        splitted_by_windows[curr_sig] = split_by_windows(signals[curr_sig], window_size, window_step, n);
    }
	
    float** device_result;
    int window_count = (int)(n - window_size) / window_step;
	pause_say("before on middleware allocation");

    cudaMalloc((void**)device_result, window_count * sizeof(float*));
    for(int i = 0; i < window_count; i++)
    {
        cudaMalloc((void**)device_result[i], (active_count + 1) * sizeof(float));
    }
    int* gpu_actives;
    cudaMalloc((void**)gpu_actives, active_count * sizeof(float));
    cudaMemcpy(gpu_actives, active_signals, active_count*sizeof(float), cudaMemcpyHostToDevice);
	pause_say("allocated, start backend");
    window_slide_correlations<<<window_count, active_count>>>(splitted_by_windows, n, signals_count, window_size, window_step, device_result, gpu_actives, active_count, main_signal);

    float** result = (float**)malloc(window_count * sizeof(float*));
    for(int i = 0; i < window_count; i++)
    {
        result[i] = (float*)malloc(window_size * sizeof(float));
    }


    cudaMemcpy(result, device_result, active_count*sizeof(float), cudaMemcpyDeviceToHost);

    return result;
}


void ShiftCompute(float** currentShiftSignals, int n, int signalCount, int shiftWidth, int batchSize, int batchStep, std::string prev_filename, std::string outputPath,
	int mainSignal, std::vector<int>& actives)
{
	int* activesArr = &actives[0];
	

	float** result = get_correlations_shift(currentShiftSignals, signalCount, n, batchSize, batchStep, activesArr, actives.size(), mainSignal);

	// Writing to ss
	int window_count = (int)(n - batchSize) / batchStep;
	for(int i = 0; i < window_count; i++)
    {
		for(int j = 0; j < actives.size(); j++)
		{
			std::cout << result[i][j] << " ";
		}
		std::cout << std::endl;
    }

}



int main(int argc, char** argv)
{
	if (argc < 8)
	{
		std::cerr << "Bad parameters... Argc: " << argc << std::endl;
		for (int i = 0; i < argc; i++)
		{
			std::cout << argv[i] << std::endl;
		}
		system("pause");
		return -1;
	}

	int mainSignal = std::stoi(argv[7]);
	std::vector<int> actives;

	for (int i = 8; i < argc; i++)
	{
		actives.push_back(std::stoi(argv[i]));
	}

	std::ifstream f;
	f.open(argv[1]);

	if (!f.good())
	{
		std::cerr << "Bad filepath input (bad file)..." << std::endl;
		system("pause");
		return -2;
	}


	std::string line, val;
	std::vector<std::vector<float>> array;

	while (std::getline(f, line))
	{
		std::vector<float> v;
		std::stringstream s(line);
		while (getline(s, val, ' '))
		{
			v.push_back(std::stof(val));
		}
		array.push_back(v);
	}

	unsigned int n = array.size();
	int signal_count = array[0].size();


	float** h_x = (float**)malloc(signal_count * sizeof(float*));

	for (int i = 0; i < signal_count; i++)
	{
		h_x[i] = (float*)malloc(n * sizeof(float));
	}

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < signal_count; j++)
		{
			h_x[j][i] = array[i][j];
		}
	}

	std::cout << "Start Computing..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	
	ShiftCompute(h_x, n, signal_count, std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), argv[5],  argv[6], mainSignal, actives);
	
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	std::cout << "End Computing" << std::endl;
	std::cout << "Duration: " << (double)duration.count() / 1000 << std::endl;
	for (int i = 0; i < actives.size(); i++)
	{
		free(h_x[i]);
	}
	free(h_x);
	system("pause");
}