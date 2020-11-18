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

#define float double

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
	int window_count = (int)(n - window_size) / window_step + 1;

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

	for(int block_p = 0; block_p < (int)(n - window_size) / window_step + 1; block_p+=50)
	{
		
		if(block_grid_id < (int)(n - window_size) / window_step + 1)
		{
			
		
			float* curr_x = current_signal_wnd(signals, sig_count, n, window_size, window_step, block_grid_id, main_signal);
			float* curr_y = current_signal_wnd(signals, sig_count, n, window_size, window_step, block_grid_id, active_signals[threadId]);
		
			
			result[(block_grid_id + block_p) * active_count + threadId] = full_correlate(curr_x, curr_y, window_size);
		
			free(curr_x);
			free(curr_y);
		}
		
	}
}


void circularShift(float* a, int size, int shift)
{
	if (shift > 0)
	{
		assert(size >= shift);
		std::rotate(a, a + size - shift, a + size);
	}
	else
	{
		assert(size >= -shift);
		std::rotate(a, a - shift, a + size);
	}
}


float* wrap_singals_to_array_wnd(float** signals, int n, int signal_count, int window_size, int window_step)
{
    int window_count = (int)(n - window_size) / window_step + 1;

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
	int window_count = (int)(n - window_size) / window_step + 1;


	float* splitted_by_windows = wrap_singals_to_array_wnd(signals, n, signals_count, window_size, window_step);


	float* device_data;
	float* device_result;
	int* gpu_actives;
    
	cudaMalloc((void**)&device_result, active_count * window_count * sizeof(float));
	cudaMalloc((void**)&device_data, signals_count * window_size * window_count * sizeof(float));
	cudaMalloc((void**)&gpu_actives, active_count * sizeof(float));
	
	cudaMemcpy(gpu_actives, (void*)active_signals, active_count*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data, (void*)splitted_by_windows, signals_count * window_size * window_count * sizeof(float), cudaMemcpyHostToDevice);

    window_slide_correlations<<<50, active_count>>>(device_data, n, signals_count, window_size, window_step, device_result, gpu_actives, active_count, main_signal);

	float* result = (float*)malloc(active_count * window_count * sizeof(float));

    cudaMemcpy(result, device_result, active_count * window_count*sizeof(float), cudaMemcpyDeviceToHost);

	free(splitted_by_windows);
	cudaFree((void**)&device_result);
	cudaFree((void**)&device_data);
	cudaFree((void**)&gpu_actives);

    return result;
}
void write_file(int curr_shift, int batchSize, int batchStep, std::string prev_filename, std::stringstream& ss, std::string outputPath)
{

	std::string filename = outputPath + "\\" + std::to_string(curr_shift) + " " + std::to_string(batchSize) + " " + std::to_string(batchStep) + " " + prev_filename;

	std::ofstream outFile(filename);
	std::cout << filename << std::endl;
	outFile << ss.rdbuf();

	outFile.close();
}

inline float round6p(float x)
{
	return roundf(x * 1000000) / 1000000;
} 

void ShiftComputeAtCurr(float** currentShiftSignals, int n, int signalCount, int shiftWidth, int batchSize, int batchStep, std::string prev_filename, std::string outputPath,
	int mainSignal, std::vector<int>& actives, int curr)
{
	int* activesArr = &actives[0];

	
	std::stringstream ss;
	for (int i = 0; i < actives.size(); i++)
	{
		ss << "Active  " << actives[i] << "\t\t";
	}
	ss << std::endl;
	if(curr != 0)
	{
		circularShift(currentShiftSignals[mainSignal], n, shiftWidth);
	}
	
	float * res = get_correlations_shift(currentShiftSignals, signalCount, n, batchSize, batchStep, activesArr, actives.size(), mainSignal);
	
	

	int window_count = (int)(n - batchSize) / batchStep + 1;
	std::string temp = "";
	for(int i = 0; i < window_count; i++)
	{
		
		for(int j = 0; j < actives.size(); j++)
		{
			temp += std::to_string( round6p(res[signalCount * i + j]));

			if(temp.size() > 9)
			{
				for(int i =0; i < 9 - temp.size(); i++)
				{
					temp += " ";
				}
			}

			
			ss << temp << "\t\t";
			temp = "";
		}


		
		ss << std::endl;
	}
	
	write_file(curr, batchSize, batchStep, prev_filename, ss, outputPath);
    ss.str("");

	
}
void ShiftCompute(float** currentShiftSignals, int n, int signalCount, int shiftWidth, int leftShift, int rightShift, int batchSize, int batchStep, std::string prev_filename, std::string outputPath,
	int mainSignal, std::vector<int>& active)
{
	

	//left Shift
	for(int i = 1; i < leftShift + 1; i++)
	{
		ShiftComputeAtCurr(currentShiftSignals, n, signalCount, -shiftWidth, batchSize, batchStep, prev_filename, outputPath, mainSignal, active, -i);
	}
	// Zeroing the shift
	for(int i = 1; i < leftShift + 1; i++)
	{
		circularShift(currentShiftSignals[mainSignal], n, shiftWidth);
	}
	//Zero Shift
	ShiftComputeAtCurr(currentShiftSignals, n, signalCount, shiftWidth, batchSize, batchStep, prev_filename, outputPath, mainSignal, active, 0);
	//Right Shift
	for(int i = 1; i < rightShift + 1; i++)
	{
		ShiftComputeAtCurr(currentShiftSignals, n, signalCount, shiftWidth, batchSize, batchStep, prev_filename, outputPath, mainSignal, active, i);
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

	int mainSignal = std::stoi(argv[9]);
	std::vector<int> actives;

	for (int i = 10; i < argc; i++)
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
	
	ShiftCompute(h_x, n, signal_count, std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[6]), argv[7],  argv[8], mainSignal, actives);
	
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