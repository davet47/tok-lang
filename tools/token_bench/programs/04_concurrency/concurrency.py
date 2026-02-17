from concurrent.futures import ThreadPoolExecutor
import queue
import threading

def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

with ThreadPoolExecutor() as pool:
    futures = [pool.submit(fib, 27) for _ in range(4)]
    r = sum(f.result() for f in futures)
    print(f"parallel: {r}")

with ThreadPoolExecutor() as pool:
    results = list(pool.map(fib, [27, 27, 27, 27]))
    total = sum(results)
    print(f"pmap: {total}")

q = queue.Queue(10)

def producer():
    for i in range(10):
        q.put(i * i)

t = threading.Thread(target=producer)
t.start()
for _ in range(10):
    print(q.get())
t.join()
