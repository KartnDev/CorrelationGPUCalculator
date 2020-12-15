
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
#include <map>

#define float double

float* cpu_correlation_mat(float** signals, int n, int signal_count, int mainSignal, std::vector<int>& actives);
void SplitByBatches(float** currentShiftSignals, int n, int signalCount, int shiftWidth, int batchSize, int batchStep,
	int mainSignal, std::vector<int>& actives, std::string filename, int currentShift, std::string outputPath);

float* cpgpu_correlation_spearmanr(float** signals, int n, int signal_count, int mainSignal, std::vector<int>& actives);


size_t index_of(float* arr, float val, int len, int start = 0)
{
	return  std::distance(arr, std::find(arr + start, arr + len, val));
}

void rankify_parallel(float* X, int n, float* res)
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

void rank(double* f,int n, double* fr)
{
	double* a = new double[n];
	for (int i = 0; i < n; i++)
	{
		a[i] = f[i];
	}

	std::sort(a, a + n);
	
	for (int i = 0; i < n; i++)
	{
	
		int index;
		if (i < n - 1 && a[i] == a[i + 1])
		{
			index = 0;
			int counter = 2;
			double mean;
			double mean_i;
			mean_i = i + 1.0;
			mean = i + 1.0;
			int j = i;
			do
			{
				j++;
				if (j < n - 1 && a[j] == a[j + 1])
				{
					counter++;
				}
			} while (j < n - 1 && a[j] == a[j + 1]);
			for (int n = 1; n < counter; n++)
			{
				mean_i++;
				mean += mean_i;
			}
			mean = mean / counter;
			for (int n = 0; n < counter; n++)
			{
				index = index_of(f, a[i], n, index);
				fr[index] = mean;
				index++;
			}
			i += counter - 1;
		}
		else
		{
			int index_s = index_of(f, a[i], n);
			fr[index_s] = i + 1;
		}
	}
}



float correlationCoefficient(float* rank_X, float* rank_Y, int n)
{
	float sum = 0.0f;

#pragma omp parallel for schedule(static) reduction (+:sum)
	for (int i = 0; i < n; i++)
	{
		sum += pow((rank_X[i] - rank_Y[i]) ,2);
	}

	float additional_res = 0;
	
	return  1.0 - 6.0 * ((sum) / (float)(n * n * n - n));
}




inline float round6p(float x)
{
	return roundf(x * 1000000) / 1000000;
}

void write_file(int curr_shift, int batchSize, int batchStep, std::string prev_filename, std::stringstream& ss, std::string outputPath)
{

	std::string filename = outputPath + "\\" + std::to_string(curr_shift) + " " + std::to_string(batchSize) + " " + std::to_string(batchStep) + " " + prev_filename;

	std::ofstream outFile(filename);
	std::cout << filename << std::endl;
	outFile << ss.rdbuf();

	outFile.close();
}


void SplitByBatches(float** currentShiftSignals, int n, int signalCount, int shiftWidth, int batchSize, int batchStep, 
	int mainSignal, std::vector<int>& actives, std::string filename, int currentShift, std::string outputPath)
{
	std::stringstream ss;
	for (int i = 0; i < actives.size(); i++)
	{
		ss << "Active  " << actives[i] << "\t\t";
	}
	ss << std::endl;

	
	float** batch = (float**)malloc(signalCount * sizeof(float*));
	for (int k = 0; k < signalCount; k++)
	{
		batch[k] = (float*)malloc(batchSize * sizeof(float));
	}


	for (int batchIndex = 0; batchIndex < (n - batchSize) + 1; batchIndex += batchStep)
	{
		for (int k = 0; k < signalCount; k++)
		{
			for (int j = 0; j < batchSize; j++)
			{
				batch[k][j] = currentShiftSignals[k][j + batchIndex];
			}
		}

		float* result = cpgpu_correlation_spearmanr(batch, batchSize, signalCount, mainSignal, actives);

		std::string temp = "";
		
		for (int j = 0; j < actives.size(); j++)
		{
			temp = "";
			temp += std::to_string( round6p(result[j]));

			if(temp.size() > 9)
			{
				for(int i =0; i < 9 - temp.size(); i++)
				{
					temp += " ";
				}
			}

			
			ss << temp << "\t\t";
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

	write_file(currentShift, batchSize, batchStep, filename, ss, outputPath);
}

void circularShift(float* a, int size, int shift)
{
	if (shift > 0)
	{
		std::rotate(a, a + size - shift, a + size);
	}
	else
	{
		std::rotate(a, a - shift, a + size);
	}
}

void ShiftCompute(float** currentShiftSignals, int n, int signalCount, int shiftWidth, int leftShift, int rightShift, int batchSize, int batchStep, std::string prev_filename, std::string outputPath,
	int mainSignal, std::vector<int>& actives)
{
	//Zero_Shift
	SplitByBatches(currentShiftSignals, n, signalCount, shiftWidth, batchSize, batchStep, mainSignal, actives, prev_filename, 0, outputPath);
	
	
	for (int i = 1; i < leftShift + 1; i ++)
	{
		circularShift(currentShiftSignals[mainSignal], n, -shiftWidth);
		SplitByBatches(currentShiftSignals, n, signalCount, shiftWidth, batchSize, batchStep, mainSignal, actives, prev_filename, -i, outputPath);
	}
	// Normalize 
	for (int i = 1; i < leftShift + 1; i++)
	{
		circularShift(currentShiftSignals[mainSignal], n, shiftWidth);
	}
	// Normalize 
	for (int i = 1; i < rightShift + 1; i ++)
	{
		circularShift(currentShiftSignals[mainSignal], n, shiftWidth);
		SplitByBatches(currentShiftSignals, n, signalCount, shiftWidth, batchSize, batchStep, mainSignal, actives, prev_filename, i, outputPath);
	}
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




char GetCurrentSeparator(std::string filepath)
{
	SetConsoleOutputCP(1251);
	SetConsoleCP(1251);

	
	std::ifstream ifs(filepath, std::ios::in);
	if (!ifs.is_open()) { // couldn't read file.. probably want to handle it.
		throw new std::exception("Bad file format!");;
	}
	std::string firstLine((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	ifs.close();
	
	for(auto sep : " \t,") // TODO all separators here
	{
		if (firstLine.find(sep) != std::string::npos)
		{
			return sep;
		}
	}

	throw new std::exception("Bad file format!");
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
	std::string filepath = argv[1];
	std::string outputFilepath = argv[8];

	int mainSignal = std::stoi(argv[9]);
	std::vector<int> actives;

	for (int i = 10; i < argc; i++)
	{
		actives.push_back(std::stoi(argv[i]));
	}

	std::cout << filepath << "\n";
	
	std::ifstream f;
	std::replace(filepath.begin(), filepath.end(), '+', ' ');
	std::replace(outputFilepath.begin(), outputFilepath.end(), '+', ' ');

	std::cout << filepath << "\n";
	
	f.open(filepath);

	if (!f.good())
	{
		std::cerr << "Bad filepath input (bad file)..." << std::endl;
		system("pause");
		return -2;
	}


	std::string line, val;
	std::vector<std::vector<float>> array;

	char separator = GetCurrentSeparator(filepath);


	while (std::getline(f, line))
	{
		std::vector<float> v;
		std::stringstream s(line);
		while (getline(s, val, separator))
		{
			std::replace(val.begin(), val.end(), ',', '.');
			v.push_back(std::stod(val));
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

	ShiftCompute(h_x, n, signal_count, std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[6]), argv[7], outputFilepath, mainSignal, actives);

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