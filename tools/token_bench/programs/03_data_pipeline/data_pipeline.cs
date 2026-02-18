using System;
using System.Collections.Generic;
using System.Linq;

class Record {
    public string Name;
    public int Score;
    public string Category;
}

class Program {
    static int Lcg(int seed) {
        return (int)(((long)seed * 1103515245 + 12345) % 2147483648);
    }

    static void Main() {
        var records = new List<Record>();
        int seed = 42;
        for (int i = 0; i < 5000; i++) {
            seed = Lcg(seed);
            int score = seed % 100;
            int cat = score % 4;
            string category = cat switch {
                0 => "alpha",
                1 => "beta",
                2 => "gamma",
                _ => "delta"
            };
            records.Add(new Record {Name = $"item{i}", Score = score, Category = category});
        }

        var highScores = records.Where(r => r.Score >= 50).ToList();

        var transformed = highScores.Select(r => new Record {
            Name = r.Name, Score = r.Score * 2, Category = r.Category
        }).ToList();

        var groups = new Dictionary<string, List<int>>();
        foreach (var r in transformed) {
            if (!groups.ContainsKey(r.Category))
                groups[r.Category] = new List<int>();
            groups[r.Category].Add(r.Score);
        }

        foreach (var (k, scores) in groups) {
            scores.Sort();
            int avg = scores.Sum() / scores.Count;
            Console.WriteLine($"{k}: count={scores.Count} avg={avg} min={scores[0]} max={scores[^1]}");
        }
    }
}
