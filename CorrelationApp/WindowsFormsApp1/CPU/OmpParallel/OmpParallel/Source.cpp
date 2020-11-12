
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
#include <cmath>

float* cpgpu_correlation_mat(float** signals, int n, int signal_count, int mainSignal, std::vector<int>& actives);
void SplitByBatches(float** currentShiftSignals, int n, int signalCount, int shiftWidth, int batchSize, int batchStep, std::stringstream& ss, int mainSignal, std::vector<int>& actives);
void ShiftCompute(float** currentShiftSignals, int n, int signalCount, int shiftWidth, int batchSize, int batchStep, std::string prev_filename, std::string outputPath,
	int mainSignal, std::vector<int>& actives);
void write_file(int shiftWidth, int batchSize, std::string prev_filename, std::stringstream& ss, std::string outputPath);
float* cpgpu_correlation_spearmanr(float** signals, int n, int signal_count, int mainSignal, std::vector<int>& actives);

typedef std::vector<float> Vector;


void rankify_parallel(float* X, int n, float* res) {

#pragma omp parallel for
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



// function that returns 
// Pearson correlation coefficient. 
float correlationCoefficient(float* rank_X, float* rank_Y, int n)
{
	float sum = 0.0f;

#pragma omp parallel for schedule(static) reduction (+:sum)
	for (int i = 0; i < n; i++)
	{
		sum += (rank_X[i] - rank_Y[i]) * (rank_X[i] - rank_Y[i]);
	}

	return  1 - 6 * sum / (n * n * n - n);
}



float correlation(float* x, float* y, int n)
{
	float avg_x = 0, avg_y = 0;

	for (int i = 0; i < n; i++)
	{
		avg_x += x[i];
		avg_y += y[i];
	}
	avg_x /= n;
	avg_y /= n;

	
	float x_sum_error = 0, y_sum_error = 0, pairwise_sum = 0;
#pragma omp parallel for schedule(static) reduction (+:x_sum_error, y_sum_error, pairwise_sum) shared(avg_x, avg_y)
	for (int i = 0; i < n; i++)
	{
		x_sum_error += (x[i] - avg_x) * (x[i] - avg_x);
		y_sum_error += (y[i] - avg_y) * (y[i] - avg_y);
		pairwise_sum += (x[i] - avg_x) * (y[i] - avg_y);
	}


	return pairwise_sum / sqrtf(x_sum_error * y_sum_error);
}


inline float round3p(float x)
{
	return roundf(x * 10000) / 10000;
}

void SplitByBatches(float** currentShiftSignals, int n, int signalCount, int shiftWidth, int batchSize, int batchStep, std::stringstream& ss, int mainSignal, std::vector<int>& actives, std::string filename, int currentShift)
{


	float** batch = (float**)malloc(signalCount * sizeof(float*));
	for (int k = 0; k < signalCount; k++)
	{
		batch[k] = (float*)malloc(batchSize * sizeof(float));
	}


	for (int batchIndex = 0; batchIndex < (n - batchSize); batchIndex += batchStep)
	{
		for (int k = 0; k < signalCount; k++)
		{
			for (int j = 0; j < batchSize; j++)
			{
				batch[k][j] = currentShiftSignals[k][j + batchIndex];
			}
		}

		float* result = cpgpu_correlation_spearmanr(batch, batchSize, signalCount, mainSignal, actives);
		for (int j = 0; j < actives.size(); j++)
		{
			ss << round3p(result[j]) << "\t\t";
		}
		ss << std::endl;
		free(result);
	}
	//freeing
	for (int k = 0; k < signalCount; k++)
	{
		free(batch[k]);
	}
	free(batch);
}

void circularShift(float* a, int size, int shift)
{
	assert(size >= shift);
	std::rotate(a, a + size - shift, a + size);
}

void ShiftCompute(float** currentShiftSignals, int n, int signalCount, int shiftWidth, int batchSize, int batchStep, std::string prev_filename, std::string outputPath,
	int mainSignal, std::vector<int>& actives)
{
	std::stringstream ss;
	for (int i = 0; i < actives.size(); i++)
	{
		ss << "Active" << actives[i] << "\t\t";
	}
	ss << std::endl;

	for (int i = 0; i < n; i += shiftWidth)
	{	
		circularShift(currentShiftSignals[mainSignal], n, shiftWidth);	

		SplitByBatches(currentShiftSignals, n, signalCount, shiftWidth, batchSize, batchStep, ss, mainSignal, actives, prev_filename, i);
	}

	write_file(shiftWidth, batchSize, prev_filename, ss, outputPath);

}

void write_file(int shiftWidth, int batchSize, std::string prev_filename, std::stringstream& ss, std::string outputPath)
{

	std::string filename = outputPath + "\\" + std::to_string(shiftWidth) + " " + std::to_string(batchSize) + " " + prev_filename;

	std::ofstream outFile(filename);
	std::cout << filename << std::endl;
	outFile << ss.rdbuf();

	outFile.close();
}



float* cpgpu_correlation_mat(float** signals, int n, int signal_count, int mainSignal, std::vector<int>& actives)
{

	float* result = (float*)malloc(actives.size() * sizeof(float));
	
	for (int i = 0; i < actives.size(); i++)
	{
		result[i] = correlation(signals[mainSignal], signals[actives[i]], n);
	}
	
	return result;
}

float* cpgpu_correlation_spearmanr(float** signals, int n, int signal_count, int mainSignal, std::vector<int>& actives)
{

	float* result = (float*)malloc(actives.size() * sizeof(float));

	float* rank_y = (float*)malloc(n * sizeof(float));
	float* rank_x = (float*)malloc(n * sizeof(float));


	rankify_parallel(signals[mainSignal], n, rank_x);
	for (int i = 0; i < actives.size(); i++)
	{

		rankify_parallel(signals[actives[i]], n, rank_y);

		result[i] = correlationCoefficient(rank_x, rank_y, n);
	}

	return result;
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