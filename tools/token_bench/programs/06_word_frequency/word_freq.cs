using System;
using System.Collections.Generic;
using System.Linq;

class Program {
    static void Main() {
        string text = "the quick brown fox jumps over the lazy dog the fox the dog and the cat sat on the mat the quick brown fox";

        var words = text.Split(' ');
        Console.WriteLine($"Total: {words.Length}");
        Console.WriteLine($"Unique: {words.Distinct().Count()}");

        var counts = new Dictionary<string, int>();
        foreach (var w in words)
            counts[w] = counts.GetValueOrDefault(w) + 1;
        foreach (var (word, count) in counts)
            Console.WriteLine($"{word}: {count}");

        var longest = words.OrderByDescending(w => w.Length).First();
        Console.WriteLine($"Longest: {longest} ({longest.Length} chars)");

        double avg = words.Average(w => w.Length);
        Console.WriteLine($"Avg length: {avg}");
    }
}
