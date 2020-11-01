using System.Collections.Generic;
using System.IO;

namespace WindowsFormsApp1.DeviceComputes
{
    public abstract class ResultWrapper
    {
        private readonly string _outputFolder;
        private readonly string _prevName;

        public ResultWrapper(string outputFolder, string prevName)
        {
            this._outputFolder = outputFolder;
            _prevName = prevName;

            if (!Directory.Exists(outputFolder))
            {
                Directory.CreateDirectory(outputFolder);
            }
        }
        
        public async void WriteMatrixesToFile(List<double[,]> matrixes, int batchSize, int shiftStep)
        {
            string filename = $"{_prevName}_{shiftStep}_{batchSize}";
            
            using (TextWriter tw = new StreamWriter(filename))
            {
                foreach (var matrix in matrixes)
                {
                    for (int j = 0; j < matrix.GetLength(0); j++)
                    {
                        for (int i = 0; i < matrix.GetLength(1); i++)
                        {
                            if (i != 0)
                            {
                                await tw.WriteAsync(" ");
                            }
                            tw.Write(matrix[i, j]);
                        }
                        await tw.WriteLineAsync();
                    }
                }
                await tw.WriteLineAsync();
            }
        }
    }
}