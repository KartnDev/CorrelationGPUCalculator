using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.Statistics;

namespace WindowsFormsApp1.Math.Statistics
{
    public class RankFinder
    {
        public static double[] Rank(IEnumerable<double> series) => series == null ? new double[0] : ArrayStatistics.RanksInplace(series.ToArray<double>());
    }
}