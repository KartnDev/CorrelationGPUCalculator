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


__device__ void rankify(float* X, int n, float* res)
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

__device__ float coorelate(float* x_rank, float* y_rank, int n)
{
	float sum = 0.0f;
	for (int i = 0; i < n; i++)
	{
		sum += (x_rank[i] - y_rank[i]) * (x_rank[i] - y_rank[i]);
	}
	return 1.0 - 6.0 * sum / (n * n * n - n);
}

__device__ float full_correlate(float* x, float* y, int n)
{
	float* rank_x = (float*)malloc(n * sizeof(float));
	float* rank_y = (float*)malloc(n * sizeof(float));

	rankify(x, n, rank_x);
	rankify(y, n, rank_y);

	float res = coorelate(rank_x, rank_y, n);

	free(rank_x);
	free(rank_y);

	return res;
}

__device__ float* current_signal_wnd(float* signals, int signal_count, int n, int window_size, int window_step, int blockId, int threadId)
{
	float* result = (float*)malloc(window_size * sizeof(float));
	int window_count = (int)(n - window_size) / window_step;

	for(int i = 0; i < window_size; i++)
	{
		result[i] = signals[threadId * (window_count * window_size) + blockId * window_size + i];
	}

	return result;
}



__global__ void window_slide_correlations(float* signals, unsigned int n, int sig_count, int window_size, int window_step, float* result,
                                            int* active_signals, unsigned int active_count, int main_signal)
{
	int block_grid_id = blockIdx.x;
	int threadId = threadIdx.x;

	for(int block_p = 0; block_p < (int)(n - window_size) / window_step; block_p+=((int)(n - window_size) / window_step))
	{
		
		if(block_grid_id < (int)(n - window_size) / window_step)
		{
			
		
			float* curr_x = current_signal_wnd(signals, sig_count, n, window_size, window_step, block_grid_id, main_signal);
			float* curr_y = current_signal_wnd(signals, sig_count, n, window_size, window_step, block_grid_id, threadId);
		
			
			result[(block_grid_id + block_p) * active_count + threadId] = full_correlate(curr_x, curr_y, window_size);
		
			free(curr_x);
			free(curr_y);
		}
		
	}
}

void circularShift(float* a, int size, int shift)
{
	assert(size >= shift);
	std::rotate(a, a + size - shift, a + size);
}


float* wrap_singals_to_array_wnd(float** signals, int n, int signal_count, int window_size, int window_step)
{
    int window_count = (int)(n - window_size) / window_step;

    float* allocated_res = (float*)malloc(signal_count * window_size * window_count * sizeof(float));
	

	for(int sig_ind = 0; sig_ind < signal_count; sig_ind++)
	{
		for(int curr_wnd = 0; curr_wnd < window_count; curr_wnd++)
		{
			for(int i = 0; i < window_size; i++)
			{
				allocated_res[sig_ind * window_size * window_count + curr_wnd * window_size + i] = signals[sig_ind][curr_wnd * window_step + i];
			}
		}
	}


    return allocated_res;
}


float* get_correlations_shift(float** signals, int signals_count, int n, int window_size, int window_step, int* active_signals, unsigned int active_count, int main_signal)
{
	int window_count = (int)(n - window_size) / window_step;


	float* splitted_by_windows = wrap_singals_to_array_wnd(signals, n, signals_count, window_size, window_step);


	float* device_data;
	float* device_result;
	int* gpu_actives;
    
	cudaMalloc((void**)&device_result, active_count * window_count * sizeof(float));
	cudaMalloc((void**)&device_data, signals_count * window_size * window_count * sizeof(float));
	cudaMalloc((void**)&gpu_actives, active_count * sizeof(float));
	
	cudaMemcpy(gpu_actives, (void*)active_signals, active_count*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data, (void*)splitted_by_windows, signals_count * window_size * window_count * sizeof(float), cudaMemcpyHostToDevice);

    window_slide_correlations<<<window_count, active_count>>>(device_data, n, signals_count, window_size, window_step, device_result, gpu_actives, active_count, main_signal);

	float* result = (float*)malloc(active_count * window_count * sizeof(float));

    cudaMemcpy(result, device_result, active_count * window_count*sizeof(float), cudaMemcpyDeviceToHost);

    return result;
}


void ShiftCompute(float** currentShiftSignals, int n, int signalCount, int shiftWidth, int batchSize, int batchStep, std::string prev_filename, std::string outputPath,
	int mainSignal, std::vector<int>& actives)
{
	int* activesArr = &actives[0];
	

	float* result = get_correlations_shift(currentShiftSignals, signalCount, n, batchSize, batchStep, activesArr, actives.size(), mainSignal);

	// Writing to ss
	int window_count = (int)(n - batchSize) / batchStep ;
	for(int i = 0; i < window_count; i++)
    {
		for(int j = 0; j < actives.size(); j++)
		{
			std::cout << result[i * actives.size() + j] << "\t";
		}
		std::cout << std::endl;
    }

}



int main(int argc, char** argv)
{
	// float** a = (float**)malloc(3 * sizeof(float*));
	// pause_say("wait");

	// int n_count = 20;
	// int _sigs = 3;
	// int win_size = 10;
	// int win_step = 5;
	// int window_count = (int)(n_count - win_size) / win_step;

	// for(int i = 0; i < n_count; i++)
	// {
	// 	a[i] = (float*)malloc(n_count * sizeof(float));
	// }
	// for(int i = 0; i < n_count; i++)
	// {
	// 	a[0][i] = i;
	// 	a[1][i] = -i;
	// 	a[2][i] = (i+1) * (i+1);
	// }
	// pause_say("wait");
	// float* wraped = wrap_singals_to_array_wnd(a, n_count, _sigs, win_size, win_step);
	// pause_say("wait");
	// for(int i = 0; i < _sigs; i++)
	// {
	// 	for(int j = 0; j < window_count; j++)
	// 	{
	// 		for(int k = 0; k < win_size; k++)
	// 		{
	// 			std::cout << wraped[i * (window_count * win_size) + j * win_size + k] << " ";
	// 		}
	// 		std::cout << " | ";
	// 	}
	// 	std::cout << "\n";
	// }
	// pause_say("wait");

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