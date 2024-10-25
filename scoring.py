import pandas as pd
import torch
from transformers import BertModel, BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import spacy
import openai
from bert_score import score
import numpy as np

# Load the pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

#for calculating perplexity
# Load pre-trained GPT-2 model and tokenizer
gpt2_model_name = "gpt2"
gpt2tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2model.eval()

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('punkt')

# Set dummy to True for testing
dummy = False
mongo_uri = "mongodb://localhost:27017/"
client = MongoClient(mongo_uri)
db = client["eval_db"]

if dummy==False:
    if "composite_scores" in db.list_collection_names():
        db.drop_collection("composite_scores")
        print("dropped composite_scores")
else:
    if "dummy_composite_scores" in db.list_collection_names():
        db.drop_collection("dummy_composite_scores")
        print("dropped dummy_composite_scores")        

response_collection = db["dummy_responses"] if dummy else db["responses"]
score_collection = db["dummy_composite_scores"] if dummy else db["composite_scores"]

# Extract data from MongoDB
data = list(response_collection.find({}))
df = pd.DataFrame(data)

# Data Cleaning and Standardization
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Ensure necessary columns are present
required_columns = {'Question', 'Answer', 'gpt3.5-turbo_response', 'gpt-4-turbo_response'}
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
df['Question Embedding'] = df['Question'].apply(get_bert_embedding)
df['Ideal Answer Embedding'] = df['Answer'].apply(get_bert_embedding)
df['LLM GPT 3.5 Response Embedding'] = df['gpt3.5-turbo_response'].apply(get_bert_embedding)
df['LLM GPT 4 Response Embedding'] = df['gpt-4-turbo_response'].apply(get_bert_embedding)

# Function to compute cosine similarity
def compute_cosine_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return 0.0
    embedding1 = embedding1.reshape(1, -1) if embedding1.ndim == 1 else embedding1
    embedding2 = embedding2.reshape(1, -1) if embedding2.ndim == 1 else embedding2
    result = cosine_similarity(embedding1, embedding2)[0][0]
    return round(result,2)

# Apply cosine similarity computation
df['SemanticSimilarity GPT3.5 Response'] = df.apply(
    lambda row: round(compute_cosine_similarity(row['Ideal Answer Embedding'], row['LLM GPT 3.5 Response Embedding']),2), axis=1)
df['SemanticSimilarity GPT4 Response'] = df.apply(
    lambda row: round(compute_cosine_similarity(row['Ideal Answer Embedding'], row['LLM GPT 4 Response Embedding']),2), axis=1)

#verify from gpt-4
def llmFactualVerification(question, claim):
    prompt = f"""
    Question: {question}
    Answer: {claim}
    """
    # Make API call to OpenAI's GPT model
    response = openai.ChatCompletion.create(
        model="gpt-4",  # You can switch to "gpt-4" if needed
        messages=[
            {"role": "system", "content": "You are a judge who responds with just one numerical value in between 0.0 to 1.0 based on the prompt provided"},
            {"role": "user", "content": prompt}
        ],
    )

    # Extracting the numerical value from the response
    result = response['choices'][0]['message']['content'].strip()
    return float(result)

def contextual_relevance(question_embedding, llm_response_embedding):
    cont_rel = compute_cosine_similarity(question_embedding, llm_response_embedding)
    return round(cont_rel,2)

# Apply factual verification and contextual relevance
""" df['Factual Verification GPT3.5 Response'] = df.apply(lambda row: factual_verification(row['Question_tokens'],row['gpt3.5-turbo_response']),axis=1)
df['Factual Verification GPT4 Response'] = df.apply(lambda row: factual_verification(row['Question_tokens'],row['gpt-4-turbo_response']),axis=1) """
df['LLM Fact Verification GPT3.5 Response'] = df.apply(lambda row: llmFactualVerification(row['Question'], row['gpt3.5-turbo_response']),axis=1)
df['LLM Fact Verification GPT4 Response'] = df.apply(lambda row: llmFactualVerification(row['Question'], row['gpt-4-turbo_response']),axis=1)
""" df['Contextual Relevance GPT3.5 Response'] = df.apply(
    lambda row: round(contextual_relevance(row['Question Embedding'], row['LLM GPT 3.5 Response Embedding']),2), axis=1)
df['Contextual Relevance GPT4 Response'] = df.apply(
    lambda row: round(contextual_relevance(row['Question Embedding'], row['LLM GPT 4 Response Embedding']),2), axis=1)
 """

# Weights for the composite score
weights = {
    'semantic': 0.4,
    'factual': 0.4,
    'contextual': 0.2
}

# Function to compute composite score
def compute_composite_score(semantic, factual, contextual, weights):
    return round((weights['semantic'] * semantic +
            weights['factual'] * factual +
            weights['contextual'] * contextual),2)


# Function to calculate BLEU score
def calculate_bleu(reference, hypothesis):
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    return round(sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=SmoothingFunction().method1),2)

# Function to calculate ROUGE score
def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

# Function to calculate METEOR score
def calculate_meteor(reference, hypothesis):
    reference = ' '.join(reference) if isinstance(reference, list) else reference
    hypothesis = ' '.join(hypothesis) if isinstance(hypothesis, list) else hypothesis
    return round(meteor_score([reference], hypothesis),2)

# 4. BERTScore
def calculate_BERTScore(references,predictions):
    # BERTScore expects lists of strings as input
    references = [references]  # Wrap the reference in a list
    predictions = [predictions]  # Wrap the prediction in a list
    P, R, F1 = score(predictions, references, lang='en', verbose=True)
    return round(F1.mean().item(),2) 


def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs.input_ids
    # Disable gradient calculation to speed up inference
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    # Get the loss (cross-entropy loss) from the output
    loss = outputs.loss
    perplexity = torch.exp(loss)
    
    return round(perplexity.item(),2)


""" df['Composite Score GPT3.5'] = df.apply(
    lambda row: compute_composite_score(
        row['SemanticSimilarity GPT3.5 Response'],
        row['LLM Fact Verification GPT3.5 Response'],
        row['Contextual Relevance GPT4 Response'],
        weights
    ), axis=1)

df['Composite Score GPT4'] = df.apply(
    lambda row: compute_composite_score(
        row['SemanticSimilarity GPT4 Response'],
        row['LLM Fact Verification GPT4 Response'],
        row['Contextual Relevance GPT4 Response'],
        weights
    ), axis=1) """

# Calculate BLEU, ROUGE, and METEOR scores for GPT-3.5 responses
df['BLEU GPT3.5'] = df.apply(lambda row: calculate_bleu(row['Answer'], row['gpt3.5-turbo_response']), axis=1)
df['ROUGE GPT3.5'] = df.apply(lambda row: calculate_rouge(row['Answer'], row['gpt3.5-turbo_response']), axis=1)
df['METEOR GPT3.5'] = df.apply(lambda row: calculate_meteor(row['Answer'], row['gpt3.5-turbo_response']), axis=1)
df['PERPLEXITY GPT3.5'] = df.apply(lambda row: calculate_perplexity(gpt2model, gpt2tokenizer, row['gpt3.5-turbo_response'] ), axis=1)
df['BERTSCORE GPT3.5'] = df.apply(lambda row: calculate_BERTScore(row['Answer'], row['gpt3.5-turbo_response']), axis=1)

# Calculate BLEU, ROUGE, and METEOR scores for GPT-4 responses
df['BLEU GPT4'] = df.apply(lambda row: calculate_bleu(row['Answer'], row['gpt-4-turbo_response']), axis=1)
df['ROUGE GPT4'] = df.apply(lambda row: calculate_rouge(row['Answer'], row['gpt-4-turbo_response']), axis=1)
df['METEOR GPT4'] = df.apply(lambda row: calculate_meteor(row['Answer'], row['gpt-4-turbo_response']), axis=1)
df['PERPLEXITY GPT4'] = df.apply(lambda row: calculate_perplexity(gpt2model, gpt2tokenizer, row['gpt-4-turbo_response'] ), axis=1)
df['BERTSCORE GPT4'] = df.apply(lambda row: calculate_BERTScore(row['Answer'], row['gpt-4-turbo_response']), axis=1)
# Drop embedding columns as they are no longer needed

# Apply composite score computation
df['Composite Score GPT3.5'] = df.apply(
    lambda row: compute_composite_score(
        row['SemanticSimilarity GPT3.5 Response'],
        row['LLM Fact Verification GPT3.5 Response'],
        row['BERTSCORE GPT3.5'],
        weights
    ), axis=1)

df['Composite Score GPT4'] = df.apply(
    lambda row: compute_composite_score(
        row['SemanticSimilarity GPT4 Response'],
        row['LLM Fact Verification GPT4 Response'],
        row['BERTSCORE GPT4'],
        weights
    ), axis=1)

df.drop(columns=['Question Embedding', 'Ideal Answer Embedding', 'LLM GPT 3.5 Response Embedding', 'LLM GPT 4 Response Embedding'], inplace=True)

# Save the DataFrame with composite scores and additional metrics to a new CSV file
if dummy:
    output_file = 'dummy_final_score.csv'
else:
    output_file = 'final_score.csv'

df.to_csv(output_file, index=False)

# Prepare data for insertion into MongoDB
score_docs = df.to_dict(orient='records')

# Insert data into 'composite_scores' collection
score_collection.insert_many(score_docs)

print(f"Composite scoring completed and saved to '{output_file}' and stored in MongoDB collection '{'dummy_composite_scores' if dummy else 'composite_scores'}'.")
