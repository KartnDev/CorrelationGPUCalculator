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
            int n = rankX.Length;

            double sum = 0;

            for (int i = 0; i < n; i++)
            {
                sum += (rankX[i] - rankY[i]) * (rankX[i] - rankY[i]);
            }
            
            return 1 - (6 * sum) / (n*n*n - n);
        }
    }
}