from collections import Counter

text = "the quick brown fox jumps over the lazy dog the fox the dog and the cat sat on the mat the quick brown fox"

words = text.split()
print(f"Total: {len(words)}")
print(f"Unique: {len(set(words))}")

counts = Counter(words)
for word, count in counts.items():
    print(f"{word}: {count}")

longest = max(words, key=len)
print(f"Longest: {longest} ({len(longest)} chars)")

avg = sum(len(w) for w in words) / len(words)
print(f"Avg length: {avg}")
