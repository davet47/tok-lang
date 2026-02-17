def qsort(a):
    if len(a) <= 1:
        return a
    p = a[len(a) // 2]
    lo = [x for x in a if x < p]
    eq = [x for x in a if x == p]
    hi = [x for x in a if x > p]
    return qsort(lo) + eq + qsort(hi)

print(qsort([3, 6, 8, 10, 1, 2, 1]))
