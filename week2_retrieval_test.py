from src.retrieval import RetrievalEngine

engine = RetrievalEngine()

query = "three women sitting in a village"

backends = [
    "bruteforce",
    "flat",
    "hnsw",
]

print("\n========================================")
print("WEEK 2 - BACKEND COMPARISON")
print("========================================")
print("Query:", query)

for backend in backends:
    print("\n----------------------------------------")
    print("Backend:", backend)
    print("----------------------------------------")

    result = engine.search(
        query,
        k=5,
        backend=backend
    )

    print("Latency:", result["latency"])

    for item in result["results"]:
        caption = (
            item["captions"][0]
            if item["captions"]
            else "No caption"
        )

        print(
            f'{item["rank"]}. '
            f'{item["image_id"]} | '
            f'score={item["score"]:.4f} | '
            f'{caption}'
        )