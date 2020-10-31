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
            var bag = new ConcurrentBag<double>();
            ParallelOptions po = new ParallelOptions();
            Console.WriteLine(po.MaxDegreeOfParallelism);
            
            Parallel.For(0, (int)n,new ParallelOptions { MaxDegreeOfParallelism = 24 } ,i =>
            {
                double temp = rankX[i] - rankY[i];
                bag.Add(System.Math.Pow(temp, 2));
                
            });

            return 1.0 - (6.0 * bag.Sum()) / (n*n*n -  n);
        }
    }
}