Project LLM Evaluation backend
A project to evaluate llm responses against ideal set of answers for specific question and answers dataset.
Evaluation done based on different metrics
1. Cosine Similartiy (embeddings)
2. BLEU
3. METEOR
4. ROUGE

Composite Scoring done based on :
1. Semantic Similarity  (Cosine similarity)- done  (Measuring the cosine similarity between the llm response embeddings and the ideal answer embeddings)
2. Factual verification - pending
3. Contextual relevance - done