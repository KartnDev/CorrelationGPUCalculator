using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using WindowsFormsApp1.Math.Statistics;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using GASS.CUDA;

namespace WindowsFormsApp1.DeviceComputes
{
    public class CUDACompute: ResultWrapper, IComputeDevice
    {
        private readonly CUDA _cuda = new CUDA(true);
        private readonly GPGPU _gpu;

        private unsafe float** ConvertFrom2D(double[][] signals)
        {
            float*[] pointer = new float*[signals.Length];
            for (int i = 0; i < signals.GetLength(0); i++)
            {
                fixed (float* singleArray = new float[signals.Length])
                {
                    for (int j = 0; j < signals[0].Length; j++)
                    {
                        singleArray[j] = (float)signals[i][j];
                    }
                    
                    pointer[i] = singleArray;
                }
            }

            fixed (float** res = pointer)
            {
                return res;
            }
        }
        
        
        
        [DllImport("kernel.dll")]
        [SuppressMessage("ReSharper", "InconsistentNaming")]
        public unsafe static extern float* gpgpu_correlation_mat(float** signals, int n, int signal_count);

        public double[,] MatrixCorrelation(double[][] signals)
        {
            double[,] result = new double[signals.GetLength(0), signals.GetLength(0)];
            unsafe
            {
                var res = gpgpu_correlation_mat(ConvertFrom2D(signals), 
                    signals.GetLength(1), 
                    signals.GetLength(0));

                for (int i = 0; i < signals.GetLength(0); i++)
                {
                    for (int j = 0; j < signals.GetLength(0); j++)
                    {
                        result[i, j] = res[i * signals.GetLength(0) + j];
                    }
                }
            }

            return result;
        }
        
        private readonly List<double[,]> _resultShiftsList = new List<double[,]>();
        public void ShiftCompute(List<double[]> fullSignals, int shiftWidth, int batchSize)
        {
            for(int i = 0; i < fullSignals.First().Length; i+= shiftWidth)
            {
                    var currentShift = new List<double[]>();

                    foreach (var signalFull in fullSignals)
                    {
                        currentShift.Add(signalFull.Skip(i).ToArray());                    
                    }

                    SplitByBatches(currentShift, 1000);
            }
            
            WriteMatrixesToFile(_resultShiftsList, batchSize, shiftWidth);
        }
        
        private void SplitByBatches(IEnumerable<double[]> currentShiftSignals, int batchSize)
        {
            int batchEpochs = currentShiftSignals.First().Count() % batchSize;
            
            for (int batchIndex = 0; batchIndex < currentShiftSignals.First().Count(); batchIndex += batchSize)
            {
                var batchList = new List<double[]>();
                foreach (var signal in currentShiftSignals)
                {
                    batchList.Add(signal.Skip(batchIndex).Take(batchIndex + batchSize).ToArray());
                }

                CalculateBatchCorrelationMatrix(batchList);
            }
        }

        private void CalculateBatchCorrelationMatrix(List<double[]> batch)
        {
            _resultShiftsList.Add(MatrixCorrelation( batch.ToArray()));
        }
        


        public int MaxParallelDegree { get; set; } = 24;
        public int RoundValue { get; set; } = 2;

        public CUDACompute(string outputFolder, string prevName) : base(outputFolder, prevName)
        {
        }
    }
}