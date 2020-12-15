using System.Collections.Generic;
using System.IO;

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
            string absPath = Path.GetDirectoryName(Path.GetDirectoryName(System.IO.Directory.GetCurrentDirectory()));
            string filename = $@"{absPath}\CPU\OmpParallel\Release\OmpParallel.exe";
            string @params = $"{pathNamePath.Replace(" ", "+")} {shiftWidth} {shiftLeft} " +
                             $"{shiftRight} {batchSize} {batchStep} {prevName} " +
                             $"{outputFolder.Replace(" ", "+")} {mainSignal}{activesStr}";

            var proc = System.Diagnostics.Process.Start(filename,@params);
        }

        public int MaxParallelDegree { get; set; }
        public int RoundValue { get; set; }


        public CppCPUCompute(string outputFolder, string filePath) : base(outputFolder, filePath)
        {

        }
    }
}