using System;
using System.Collections.Generic;
using Cudafy;
using GASS.CUDA;

namespace WindowsFormsApp1.DeviceComputes
{
    public class CUDACompute: ResultWrapper, IComputeDevice
    {
        CUDA _cuda = new CUDA(true);
        
        
        public CUDACompute(string outputFolder, string prevName) : base(outputFolder, prevName)
        {
        }

        public void ShiftCompute(List<double[]> fullSignals, int shiftWidth, int batchSize)
        {
              
        }
        [Cudafy]
        private void ComputeSpearmanrCorrelationGpu(GThread thread, double[] a, double[] b, ref double c, int n)
        {
            int tid = thread.blockIdx.x;

            if (tid < a.Length)
                c += (a[tid] - b[tid]) * (a[tid] - b[tid]);

            c = 1 - 1.0 - (6.0 * c) / (n * n * n - n);
        }
        
        
        public int MaxParallelDegree { get; set; }
        public int RoundValue { get; set; }
    }
}