using System.Collections.Generic;

namespace WindowsFormsApp1.DeviceComputes
{
    public class CUDACompute: ResultWrapper, IComputeDevice
    {
        public CUDACompute(string outputFolder, string prevName) : base(outputFolder, prevName)
        {
        }

        public void ShiftCompute(List<double[]> fullSignals, int shiftWidth, int batchSize)
        {
            
        }

        public int MaxParallelDegree { get; set; }
        public int RoundValue { get; set; }
    }
}