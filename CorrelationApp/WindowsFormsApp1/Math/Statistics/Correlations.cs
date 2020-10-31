using System;
using System.Collections.Concurrent;
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
            double sum = 0;


            for (int i = 0; i < (int) n; i++)
            {
                double temp = rankX[i] - rankY[i];
                sum += temp * temp;
            }

            return 1.0 - (6.0 * sum) / (n * n * n - n);
        }
    }
}