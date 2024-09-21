import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import wikipediaapi
import spacy

# Load the pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('punkt')

# Set dummy to True for testing
dummy = False
mongo_uri = "mongodb://localhost:27017/"
client = MongoClient(mongo_uri)
db = client["eval_db"]
response_collection = db["dummy_responses_token"] if dummy else db["responses_token"]
score_collection = db["dummy_composite_scores"] if dummy else db["composite_scores"]

# Extract data from MongoDB
data = list(response_collection.find({}))
df = pd.DataFrame(data)

# Data Cleaning and Standardization
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Ensure necessary columns are present
required_columns = {'Question_tokens', 'Answer_tokens', 'gpt3.5-turbo_response_tokens', 'gpt-4-turbo_response_tokens'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Database must contain the columns: {required_columns}")

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
df['Question Embedding'] = df['Question_tokens'].apply(get_bert_embedding)
df['Ideal Answer Embedding'] = df['Answer_tokens'].apply(get_bert_embedding)
df['LLM GPT 3.5 Response Embedding'] = df['gpt3.5-turbo_response_tokens'].apply(get_bert_embedding)
df['LLM GPT 4 Response Embedding'] = df['gpt-4-turbo_response_tokens'].apply(get_bert_embedding)

# Function to compute cosine similarity
def compute_cosine_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return 0.0
    embedding1 = embedding1.reshape(1, -1) if embedding1.ndim == 1 else embedding1
    embedding2 = embedding2.reshape(1, -1) if embedding2.ndim == 1 else embedding2
    return cosine_similarity(embedding1, embedding2)[0][0]

# Apply cosine similarity computation
df['SemanticSimilarity GPT3.5 Response'] = df.apply(
    lambda row: compute_cosine_similarity(row['Ideal Answer Embedding'], row['LLM GPT 3.5 Response Embedding']), axis=1)
df['SemanticSimilarity GPT4 Response'] = df.apply(
    lambda row: compute_cosine_similarity(row['Ideal Answer Embedding'], row['LLM GPT 4 Response Embedding']), axis=1)

# Initialize Wikipedia API with a proper User-Agent
def get_wikipedia_summary(query):
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent="FactChecker/1.0 (factchecker@example.com)"
    )
    page = wiki_wiki.page(query)
    if page.exists():
        return page.summary
    return None

# Function to get embeddings
def get_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Function to compare claim with each sentence in evidence
def compare_claim_with_evidence(claim, evidence, tokenizer, model):
    claim_embedding = get_embeddings(claim, tokenizer, model)
    sentences = evidence.split('. ')
    max_similarity = -1
    best_sentence = None

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            sentence_embedding = get_embeddings(sentence, tokenizer, model)
            similarity = 1 - compute_cosine_similarity(claim_embedding, sentence_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                best_sentence = sentence

    return best_sentence, max_similarity

# Main function to run fact-checking pipeline
def fact_check_claim(claim, topic):
    print(f"Claim: {claim}")

    # Retrieve evidence from Wikipedia
    evidence = get_wikipedia_summary(topic)
    if evidence:
        print(f"Evidence from Wikipedia:\n{evidence}...\n")  # Limiting output for readability

        # Compare claim with evidence sentences
        best_sentence, max_similarity = compare_claim_with_evidence(claim, evidence, tokenizer, model)

        if best_sentence:
            print(f"Best Matching Sentence: {best_sentence}")
            print(f"Similarity Score: {max_similarity:.2f}")
        else:
            print("No relevant sentences found.")
    else:
        print("No relevant evidence found on Wikipedia.")

    return max_similarity

def extract_named_entities(sentence):
    # Process the sentence with spaCy
    print(sentence)

    doc = nlp(sentence)
    
    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return entities

# Placeholder functions for factual verification and contextual relevance
def factual_verification(sector, llm_response):
   """  topic = extract_named_entities(sector)
    print(topic, llm_response) """
   result = fact_check_claim(llm_response, sector)
   return result

def contextual_relevance(question_embedding, llm_response_embedding):
    return compute_cosine_similarity(question_embedding, llm_response_embedding)

# Apply factual verification and contextual relevance
df['Factual Verification GPT3.5 Response'] = df.apply(lambda row: factual_verification(row['Sector'],row['gpt3.5-turbo_response_tokens']),axis=1)
df['Factual Verification GPT4 Response'] = df.apply(lambda row: factual_verification(row['Sector'],row['gpt-4-turbo_response_tokens']),axis=1)
df['Contextual Relevance GPT3.5 Response'] = df.apply(
    lambda row: contextual_relevance(row['Question Embedding'], row['LLM GPT 3.5 Response Embedding']), axis=1)
df['Contextual Relevance GPT4 Response'] = df.apply(
    lambda row: contextual_relevance(row['Question Embedding'], row['LLM GPT 4 Response Embedding']), axis=1)

# Function to compute composite score
def compute_composite_score(semantic, factual, contextual, weights):
    return (weights['semantic'] * semantic +
            weights['factual'] * factual +
            weights['contextual'] * contextual)

# Weights for the composite score
weights = {
    'semantic': 0.4,
    'factual': 0.4,
    'contextual': 0.2
}

# Apply composite score computation
df['Composite Score GPT3.5'] = df.apply(
    lambda row: compute_composite_score(
        row['SemanticSimilarity GPT3.5 Response'],
        row['Factual Verification GPT3.5 Response'],
        row['Contextual Relevance GPT3.5 Response'],
        weights
    ), axis=1)

df['Composite Score GPT4'] = df.apply(
    lambda row: compute_composite_score(
        row['SemanticSimilarity GPT4 Response'],
        row['Factual Verification GPT4 Response'],
        row['Contextual Relevance GPT4 Response'],
        weights
    ), axis=1)

# Function to calculate BLEU score
def calculate_bleu(reference, hypothesis):
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=SmoothingFunction().method1)

# Function to calculate ROUGE score
def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

# Function to calculate METEOR score
def calculate_meteor(reference, hypothesis):
    reference = ' '.join(reference) if isinstance(reference, list) else reference
    hypothesis = ' '.join(hypothesis) if isinstance(hypothesis, list) else hypothesis
    return meteor_score([reference], hypothesis)

# Calculate BLEU, ROUGE, and METEOR scores for GPT-3.5 responses
df['BLEU GPT3.5'] = df.apply(lambda row: calculate_bleu(row['Answer_tokens'], row['gpt3.5-turbo_response_tokens']), axis=1)
df['ROUGE GPT3.5'] = df.apply(lambda row: calculate_rouge(row['Answer_tokens'], row['gpt3.5-turbo_response_tokens']), axis=1)
df['METEOR GPT3.5'] = df.apply(lambda row: calculate_meteor(row['Answer_tokens'], row['gpt3.5-turbo_response_tokens']), axis=1)

# Calculate BLEU, ROUGE, and METEOR scores for GPT-4 responses
df['BLEU GPT4'] = df.apply(lambda row: calculate_bleu(row['Answer_tokens'], row['gpt-4-turbo_response_tokens']), axis=1)
df['ROUGE GPT4'] = df.apply(lambda row: calculate_rouge(row['Answer_tokens'], row['gpt-4-turbo_response_tokens']), axis=1)
df['METEOR GPT4'] = df.apply(lambda row: calculate_meteor(row['Answer_tokens'], row['gpt-4-turbo_response_tokens']), axis=1)

# Drop embedding columns as they are no longer needed
df.drop(columns=['Question Embedding', 'Ideal Answer Embedding', 'LLM GPT 3.5 Response Embedding', 'LLM GPT 4 Response Embedding'], inplace=True)

# Save the DataFrame with composite scores and additional metrics to a new CSV file
output_file = 'final_score.csv'
df.to_csv(output_file, index=False)

# Prepare data for insertion into MongoDB
score_docs = df.to_dict(orient='records')

# Insert data into 'composite_scores' collection
score_collection.insert_many(score_docs)

print(f"Composite scoring completed and saved to '{output_file}' and stored in MongoDB collection '{'dummy_composite_scores' if dummy else 'composite_scores'}'.")
