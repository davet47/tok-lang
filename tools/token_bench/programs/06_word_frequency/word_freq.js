const text = "the quick brown fox jumps over the lazy dog the fox the dog and the cat sat on the mat the quick brown fox";

const words = text.split(" ");
console.log(`Total: ${words.length}`);
console.log(`Unique: ${new Set(words).size}`);

const counts = {};
for (const w of words) counts[w] = (counts[w] || 0) + 1;
for (const [word, count] of Object.entries(counts)) {
    console.log(`${word}: ${count}`);
}

const longest = words.reduce((a, w) => w.length > a.length ? w : a);
console.log(`Longest: ${longest} (${longest.length} chars)`);

const avg = words.reduce((a, w) => a + w.length, 0) / words.length;
console.log(`Avg length: ${avg}`);
