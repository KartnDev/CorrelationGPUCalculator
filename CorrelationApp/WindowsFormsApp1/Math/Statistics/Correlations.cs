using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace WindowsFormsApp1.Math.Statistics
{
    public static class Correlations
    {
        public static double Spearmanr(double[] rankX, double[] rankY)
        {
            double n = rankX.Length;
            
            
            double sum = 0.0;
            Parallel.For(0, (int)n, i =>
            {
                double temp = rankX[i] - rankY[i];
                sum += System.Math.Pow(temp, 2);
            });
            
            return 1.0 - (6.0 * sum) / (n*n*n -  n);
        }
    }
}