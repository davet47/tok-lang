function qsort(a) {
    if (a.length <= 1) return a;
    const p = a[Math.floor(a.length / 2)];
    const lo = a.filter(x => x < p);
    const eq = a.filter(x => x === p);
    const hi = a.filter(x => x > p);
    return [...qsort(lo), ...eq, ...qsort(hi)];
}

console.log(qsort([3, 6, 8, 10, 1, 2, 1]));
