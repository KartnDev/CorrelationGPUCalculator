using System.Collections.Generic;

namespace WindowsFormsApp1.DeviceComputes
{
    public interface IComputeDevice
    {
        void ShiftCompute(List<double[]> fullSignals, int shiftWidth, int batchSize);
        
        int MaxParallelDegree { get; set; }
        int RoundValue { get; set; }
    }
}