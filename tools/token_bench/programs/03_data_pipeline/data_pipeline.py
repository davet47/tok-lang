def lcg(seed):
    return (seed * 1103515245 + 12345) % 2147483648

records = []
seed = 42
for i in range(5000):
    seed = lcg(seed)
    score = seed % 100
    cat = score % 4
    category = {0: "alpha", 1: "beta", 2: "gamma"}.get(cat, "delta")
    records.append({"name": f"item{i}", "score": score, "category": category})

high_scores = [r for r in records if r["score"] >= 50]

transformed = [
    {"name": r["name"], "score": r["score"] * 2, "category": r["category"]}
    for r in high_scores
]

groups = {}
for r in transformed:
    key = r["category"]
    if key not in groups:
        groups[key] = []
    groups[key].append(r["score"])

for k, scores in groups.items():
    scores.sort()
    avg = sum(scores) // len(scores)
    print(f"{k}: count={len(scores)} avg={avg} min={scores[0]} max={scores[-1]}")
