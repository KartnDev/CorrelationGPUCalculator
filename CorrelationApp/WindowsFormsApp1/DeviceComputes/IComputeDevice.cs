using System.Collections.Generic;

namespace WindowsFormsApp1.DeviceComputes
{
    public interface IComputeDevice
    {
        void ShiftCompute(List<double[]> fullSignals, 
            int shiftWidth, 
            int shiftLeft,
            int shiftRight,
            int batchSize, 
            int batchStep,
            int mainSignal,
            List<int> actives);

        int MaxParallelDegree { get; set; }
        int RoundValue { get; set; }
    }
}