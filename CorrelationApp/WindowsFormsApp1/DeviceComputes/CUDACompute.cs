using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
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

        public void ShiftCompute(List<double[]> fullSignals, int shiftWidth, int shiftLeft, int shiftRight, int batchSize, int batchStep, int mainSignal, List<int> actives)
        {
            string activesStr = "";
            foreach (var item in actives)
            {
                activesStr += $" {item}";
            }

            string filename = @"C:\Users\Dmitry\Documents\GitHub\CorrelationApp\CorrelationGPUCalculator\CorrelationApp\WindowsFormsApp1\GPGPU\a.exe";
            string @params = $"{pathNamePath} {shiftWidth} {shiftLeft} {shiftRight} {batchSize} {batchStep} {prevName} {outputFolder} {mainSignal}{activesStr}";

            var proc = System.Diagnostics.Process.Start(filename,@params);
        }

        public int MaxParallelDegree { get; set; }
        public int RoundValue { get; set; }


        public CUDACompute(string outputFolder, string filePath) : base(outputFolder, filePath)
        {

        }
    }
}