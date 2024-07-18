import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv('responses.csv')

# Ensure necessary columns are present
required_columns = {'ideal_answer', 'gpt3.5-turbo_response', 'gpt-4-turbo_response'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"The CSV file must contain the columns: {required_columns}")

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to compute BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Compute BERT embeddings
df['Ideal Answer Embedding'] = df['ideal_answer'].apply(get_bert_embedding)
df['LLM GPT 3.5 Response Embedding'] = df['gpt3.5-turbo_response'].apply(get_bert_embedding)
df['LLM GPT 4 Response Embedding'] = df['gpt-4-turbo_response'].apply(get_bert_embedding)

# Function to compute cosine similarity
def compute_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Apply cosine similarity computation
df['SemanticSimilarity GPT3.5 Response'] = df.apply(
    lambda row: compute_cosine_similarity(row['Ideal Answer Embedding'], row['LLM GPT 3.5 Response Embedding']), axis=1)
df['SemanticSimilarity GPT4 Response'] = df.apply(
    lambda row: compute_cosine_similarity(row['Ideal Answer Embedding'], row['LLM GPT 4 Response Embedding']), axis=1)

# Placeholder functions for factual verification and contextual relevance
# Replace these with actual implementations if needed
def factual_verification(llm_response):
    return 1  # Example: always factually correct

def contextual_relevance(question, llm_response):
    return 1  # Example: always contextually relevant

# Apply factual verification and contextual relevance
df['Factual Verification GPT3.5 Response'] = df['gpt3.5-turbo_response'].apply(factual_verification)
df['Factual Verification GPT4 Response'] = df['gpt-4-turbo_response'].apply(factual_verification)
df['Contextual Relevance GPT3.5 Response'] = df.apply(
    lambda row: contextual_relevance(row['ideal_answer'], row['gpt3.5-turbo_response']), axis=1)
df['Contextual Relevance GPT4 Response'] = df.apply(
    lambda row: contextual_relevance(row['ideal_answer'], row['gpt-4-turbo_response']), axis=1)

# Function to compute composite score
def compute_composite_score(row, weights):
    semantic_similarity = row['SemanticSimilarity']
    factual_verification = row['Factual Verification']
    contextual_relevance = row['Contextual Relevance']
    return (weights['semantic'] * semantic_similarity +
            weights['factual'] * factual_verification +
            weights['contextual'] * contextual_relevance)

# Weights for the composite score
weights = {
    'semantic': 0.4,
    'factual': 0.4,
    'contextual': 0.2
}

# Apply composite score computation
df['Composite Score GPT3.5'] = df.apply(
    lambda row: compute_composite_score({
        'SemanticSimilarity': row['SemanticSimilarity GPT3.5 Response'],
        'Factual Verification': row['Factual Verification GPT3.5 Response'],
        'Contextual Relevance': row['Contextual Relevance GPT3.5 Response']
    }, weights), axis=1)

df['Composite Score GPT4'] = df.apply(
    lambda row: compute_composite_score({
        'SemanticSimilarity': row['SemanticSimilarity GPT4 Response'],
        'Factual Verification': row['Factual Verification GPT4 Response'],
        'Contextual Relevance': row['Contextual Relevance GPT4 Response']
    }, weights), axis=1)

# Drop embedding columns as they are no longer needed
df = df.drop(columns=['Ideal Answer Embedding', 'LLM GPT 3.5 Response Embedding', 'LLM GPT 4 Response Embedding'])

# Save the DataFrame with composite scores to a new CSV file
output_file = 'responses_with_composite_score.csv'
df.to_csv(output_file, index=False)

print(f"Composite scoring completed and saved to '{output_file}'")
