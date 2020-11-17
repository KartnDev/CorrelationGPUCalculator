using System.Collections.Generic;

namespace WindowsFormsApp1.DeviceComputes
{
    public class CppCPUCompute: ResultWrapper, IComputeDevice
    {
        public void ShiftCompute(List<double[]> fullSignals, int shiftWidth, int shiftLeft, int shiftRight,   int batchSize, int batchStep, int mainSignal, List<int> actives)
        {
            string activesStr = "";
            foreach (var item in actives)
            {
                activesStr += $" {item}";
            }

            string filename = @"C:\Users\Dmitry\Documents\GitHub\CorrelationApp\CorrelationGPUCalculator\CorrelationApp\WindowsFormsApp1\CPU\OmpParallel\Release\OmpParallel.exe";
            string @params = $"{pathNamePath} {shiftWidth} {shiftLeft} {shiftRight} {batchSize} {batchStep} {prevName} {outputFolder} {mainSignal}{activesStr}";

            var proc = System.Diagnostics.Process.Start(filename,@params);
        }

        public int MaxParallelDegree { get; set; }
        public int RoundValue { get; set; }


        public CppCPUCompute(string outputFolder, string filePath) : base(outputFolder, filePath)
        {

        }
    }
}