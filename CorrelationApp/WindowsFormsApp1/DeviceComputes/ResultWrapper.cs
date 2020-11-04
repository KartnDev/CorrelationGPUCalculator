using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WindowsFormsApp1.DeviceComputes
{
    public abstract class ResultWrapper
    {
        private readonly string _outputFolder;
        protected readonly string prevName;
        protected readonly string pathNamePath;
        
        public ResultWrapper(string outputFolder, string filePath)
        {
            _outputFolder = outputFolder;
            
            this.pathNamePath = filePath;
            prevName = filePath.Split("\\".ToCharArray()).Last();

            if (!Directory.Exists(outputFolder))
            {
                Directory.CreateDirectory(outputFolder);
            }
        }

        public async void WriteMatrixesToFile(List<double[,]> matrixes, int batchSize, int shiftStep)
        {
            string filename = $"{_outputFolder}//{shiftStep}_{batchSize}_{prevName}";

            if (File.Exists(filename))
            {
                File.Delete(filename);
            }

            StringBuilder stringBuilder = new StringBuilder();

            foreach (var matrix in matrixes)
            {
                for (int j = 0; j < matrix.GetLength(0); j++)
                {
                    for (int i = 0; i < matrix.GetLength(1); i++)
                    {
                        if (i != 0)
                        {
                            stringBuilder.Append(" ");
                        }

                        stringBuilder.Append(matrix[i, j].ToString());
                    }

                    stringBuilder.Append("\n");
                }

                stringBuilder.Append("\n\r\n\r");
            }

            Task.Run(() => { File.WriteAllText(filename, stringBuilder.ToString()); });
        }
    }
}