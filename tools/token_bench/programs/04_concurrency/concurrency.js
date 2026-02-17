const { Worker, isMainThread, parentPort, workerData } = require("worker_threads");

function fib(n) {
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);
}

if (!isMainThread) {
    parentPort.postMessage(fib(workerData));
} else {
    async function runParallel() {
        const promises = Array.from({length: 4}, () =>
            new Promise((resolve, reject) => {
                const w = new Worker(__filename, {workerData: 27});
                w.on("message", resolve);
                w.on("error", reject);
            })
        );
        const results = await Promise.all(promises);
        const r = results.reduce((a, b) => a + b, 0);
        console.log(`parallel: ${r}`);

        const promises2 = [27, 27, 27, 27].map(n =>
            new Promise((resolve, reject) => {
                const w = new Worker(__filename, {workerData: n});
                w.on("message", resolve);
                w.on("error", reject);
            })
        );
        const results2 = await Promise.all(promises2);
        const total = results2.reduce((a, b) => a + b, 0);
        console.log(`pmap: ${total}`);

        const {Channel} = require("worker_threads");
        const values = [];
        for (let i = 0; i < 10; i++) values.push(i * i);
        for (const v of values) console.log(v);
    }
    runParallel();
}
