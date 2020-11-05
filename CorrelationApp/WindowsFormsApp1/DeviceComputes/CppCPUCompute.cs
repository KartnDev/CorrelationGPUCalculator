using System.Collections.Generic;

namespace WindowsFormsApp1.DeviceComputes
{
    public class CppCPUCompute: ResultWrapper, IComputeDevice
    {
        public void ShiftCompute(List<double[]> fullSignals, int shiftWidth, int batchSize, int mainSignal, List<int> actives)
        {
            string activesStr = "";
            foreach (var item in actives)
            {
                activesStr += $" {item}";
            }

            string filename = @"C:\Users\Dmitry\Documents\GitHub\CorrelationApp\CorrelationGPUCalculator\CorrelationApp\WindowsFormsApp1\CPU\OmpParallel.exe";
            string @params = $"{pathNamePath} {shiftWidth} {batchSize} {prevName} {outputFolder} {mainSignal}{activesStr}";

            var proc = System.Diagnostics.Process.Start(filename,@params);
        }

        public int MaxParallelDegree { get; set; }
        public int RoundValue { get; set; }


        public CppCPUCompute(string outputFolder, string filePath) : base(outputFolder, filePath)
        {

        }
    }
}