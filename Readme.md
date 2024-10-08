Project LLM Evaluation backend
A project to evaluate llm questin and answer systems.
Evaluation done based on different metrics
1. Semantic Similarity
2. BLEU
3. METEOR
4. ROUGE
5. BERTScore
6. Perplexity

Composite Scoring done based on :
1. Semantic Similarity - calculate cosine similarity using bert embeddings
2. Factual verification - done using llms
3. Contextual relevance - BERTscore