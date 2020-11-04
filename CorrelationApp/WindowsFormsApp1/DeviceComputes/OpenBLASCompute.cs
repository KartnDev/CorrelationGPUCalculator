using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MathNet;

namespace WindowsFormsApp1.DeviceComputes
{
    public class OpenBLASCompute: ResultWrapper, IComputeDevice
    {
        public int MaxParallelDegree { get; set; } = 24;
        public int RoundValue { get; set; } = 2;
        
        private readonly ConcurrentBag<double[,]> _resultShiftsList = new ConcurrentBag<double[,]>(); // TODO Is dense need ?
        public OpenBLASCompute(string outputFolder, string filePath) : base(outputFolder, filePath)
        {
        }
        
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
            
            WriteMatrixesToFile(_resultShiftsList.ToList(), batchSize, shiftWidth);
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

        private void CalculateBatchCorrelationMatrix(List<double[]> batchList)
        {
            _resultShiftsList.Add(MathNet.Numerics.Statistics.Correlation.SpearmanMatrix(batchList).ToArray());
        }
    }
}