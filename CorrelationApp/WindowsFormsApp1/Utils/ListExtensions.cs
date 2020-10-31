using System;
using System.Collections.Generic;
using System.Linq;

namespace WindowsFormsApp1.Utils
{
    public static class ListExtensions
    {
        public static int[] StandardRank(IEnumerable<double> data)
        {
            return data.AsParallel()
                .Select((x, i) => new {OldIndex = i, Value = x, NewIndex = -1})
                .OrderByDescending(x => x.Value)
                .Select((x, i) => new {OldIndex = x.OldIndex, Value = x.Value, NewIndex = i + 1})
                .OrderBy(x => x.OldIndex).Select((arg => arg.NewIndex))
                .ToArray();
        }

        
    }
}