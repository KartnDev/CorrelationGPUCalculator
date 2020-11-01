using System;
using System.Collections.Generic;
using System.Linq;
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
        
        public CUDACompute(string outputFolder, string prevName) : base(outputFolder, prevName)
        {
            CudafyModule km = CudafyModule.TryDeserialize(typeof(CUDACompute).Name);
            if (km == null || !km.TryVerifyChecksums())
            {
                km = CudafyTranslator.Cudafy(typeof(CUDACompute));
                km.Serialize();
            }
            
            _gpu = CudafyHost.GetDevice(eGPUType.Cuda);
            _gpu.LoadModule(km);
            
        }
        private readonly List<double[,]> _resultShiftsList = new List<double[,]>();
        public void ShiftCompute(List<double[]> fullSignals, int shiftWidth, int batchSize)
        {
            Parallel.For(0, fullSignals.First().Length, 
                new ParallelOptions(){MaxDegreeOfParallelism = MaxParallelDegree},
                index =>
                {
                    var currentShift = new List<double[]>();

                    foreach (var signalFull in fullSignals)
                    {
                        currentShift.Add(signalFull.Skip(index).ToArray());                    
                    }

                    SplitByBatches(currentShift, 1000);
                });
            
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
            double[,] correlationMatrix = new double[batch.Count, batch.Count];

            for (int i = 0; i < batch.Count; i++)
            {
                for (int j = 0; j < batch.Count; j++)
                {
                    var rankX = RankFinder.Rank(batch[i]);
                    var rankY = RankFinder.Rank(batch[j]);
                    
                    double[] deviceRankX = _gpu.CopyToDevice(rankX);
                    double[] deviceRankY = _gpu.CopyToDevice(rankY);

                    double result = 0.0;

                    _gpu.Launch(rankX.Length, 1)
                        .ComputeSpearmanrCorrelationGpu(deviceRankX, deviceRankY, ref result, rankX.Length);
                    correlationMatrix[i, j] = System.Math.Round(result , RoundValue);
                }
            }
            _resultShiftsList.Add(correlationMatrix);
        }
        
        [Cudafy]
        private static void ComputeSpearmanrCorrelationGpu(GThread thread, double[] a, double[] b, ref double c, int n)
        {
            int tid = thread.blockIdx.x;

            if (tid < a.Length)
                c += (a[tid] - b[tid]) * (a[tid] - b[tid]);

            c = 1 - 1.0 - (6.0 * c) / (n * n * n - n);
        }


        public int MaxParallelDegree { get; set; } = 24;
        public int RoundValue { get; set; }
    }
}