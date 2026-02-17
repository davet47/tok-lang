function lcg(seed) {
    return (seed * 1103515245 + 12345) % 2147483648;
}

const records = [];
let seed = 42;
for (let i = 0; i < 5000; i++) {
    seed = lcg(seed);
    const score = seed % 100;
    const cat = score % 4;
    const category = {0: "alpha", 1: "beta", 2: "gamma"}[cat] || "delta";
    records.push({name: `item${i}`, score, category});
}

const highScores = records.filter(r => r.score >= 50);

const transformed = highScores.map(r => ({
    name: r.name, score: r.score * 2, category: r.category
}));

const groups = {};
for (const r of transformed) {
    if (!groups[r.category]) groups[r.category] = [];
    groups[r.category].push(r.score);
}

for (const [k, scores] of Object.entries(groups)) {
    scores.sort((a, b) => a - b);
    const avg = Math.floor(scores.reduce((a, b) => a + b, 0) / scores.length);
    console.log(`${k}: count=${scores.length} avg=${avg} min=${scores[0]} max=${scores[scores.length - 1]}`);
}
