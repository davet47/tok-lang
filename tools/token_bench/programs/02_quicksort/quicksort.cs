using System;
using System.Collections.Generic;
using System.Linq;

class Program {
    static List<int> QSort(List<int> a) {
        if (a.Count <= 1) return a;
        int p = a[a.Count / 2];
        var lo = a.Where(x => x < p).ToList();
        var eq = a.Where(x => x == p).ToList();
        var hi = a.Where(x => x > p).ToList();
        var result = QSort(lo);
        result.AddRange(eq);
        result.AddRange(QSort(hi));
        return result;
    }

    static void Main() {
        var sorted = QSort(new List<int> {3, 6, 8, 10, 1, 2, 1});
        Console.WriteLine(string.Join(", ", sorted));
    }
}
