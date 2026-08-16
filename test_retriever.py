from src.rag import FBRRetriever

retriever = FBRRetriever(top_k=5)

results = retriever.retrieve(
    "standard sales tax rate under Sales Tax Act 1990"
)

for r in results:
    print(r["metadata"]["source"], r["score"])