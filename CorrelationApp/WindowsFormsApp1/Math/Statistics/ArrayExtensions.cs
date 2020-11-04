using System.Linq;

namespace WindowsFormsApp1.Math.Statistics
{
    public static class ArrayExtensions
    {
        public static double[] RightShift(double[] array, int shift)
        {
            return array.AsParallel().Skip(array.Length - shift).Concat(array.Take(array.Length - shift)).ToArray();
        }
    }
}