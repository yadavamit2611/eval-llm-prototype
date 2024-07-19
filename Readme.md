Project LLM Evaluation backend
A project to evaluate llm responses against ideal set of answers for specific question and answers dataset.
Evaluation process includes a way to get a composite scoring based on :
1. Semantic Similarity - done  (Measuring the cosine similarity between the llm response embeddings and the ideal answer embeddings)
2. Factual verification - pending
3. Contextual relevance - done