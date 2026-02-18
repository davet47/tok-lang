using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;

class Program {
    static int Fib(int n) {
        if (n < 2) return n;
        return Fib(n - 1) + Fib(n - 2);
    }

    static void Main() {
        var tasks = Enumerable.Range(0, 4).Select(_ => Task.Run(() => Fib(27))).ToArray();
        Task.WaitAll(tasks);
        int r = tasks.Sum(t => t.Result);
        Console.WriteLine($"parallel: {r}");

        var inputs = new[] {27, 27, 27, 27};
        var results = inputs.AsParallel().Select(Fib).ToArray();
        int total = results.Sum();
        Console.WriteLine($"pmap: {total}");

        var queue = new BlockingCollection<int>(10);
        Task.Run(() => {
            for (int i = 0; i < 10; i++) queue.Add(i * i);
            queue.CompleteAdding();
        });
        foreach (var v in queue.GetConsumingEnumerable()) {
            Console.WriteLine(v);
        }
    }
}
