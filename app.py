# app.py
from flask import Flask, request, jsonify
from pymongo import MongoClient
from flask_cors import CORS
import os
import pandas as pd
import openai
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import torch
from bert_score import score
from transformers import BertModel, BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)

# Configure MongoDB connection
mongo_uri = "mongodb://localhost:27017/"
client = MongoClient(mongo_uri)
db = client["eval_db"]

""" nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) """

#for calculating perplexity
# Load pre-trained GPT-2 model and tokenizer
gpt2_model_name = "gpt2"
gpt2tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2model.eval()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)


# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the file to the specified directory
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.csv')
    file.save(file_path)

    return jsonify({'message': 'File uploaded successfully', 'filename': file.filename}), 200

# Function to get LLM response
def get_llm_response(Question,test_model):
    completion = openai.ChatCompletion.create(
    model=test_model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Provide short and concise answers"},
        {"role": "user", "content": Question}
    ]
    )
    return completion.choices[0].message.content


@app.route('/api/generateLLMResponse', methods=['POST'])
def generate():
    payload = request.get_json()

    # Check if 'frontEndRequest' key is in the payload
    if 'frontEndRequest' not in payload:
        return jsonify({'error': 'Missing type parameter in request payload'}), 400
    
    frontEndRequest = payload['frontEndRequest']

    if frontEndRequest == "True":
        # Check if 'uploads/dataset.csv' file exists
        if not os.path.exists('uploads/dataset.csv'):
            return jsonify({'error': 'Dataset file not found in uploads directory'}), 400

        response_collection = db["web_responses_token"]
        df = pd.read_csv('uploads/dataset.csv')
    else:
        response_collection = db["web_responses_token"]
        # Extract data from MongoDB
        data = list(response_collection.find({}))
        for record in data:
            record['_id'] = str(record['_id'])
        return jsonify(data)

    # Data Cleaning and Standardization
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    df['Question'] = df['Question'].str.lower().str.strip()
    df['Answer'] = df['Answer'].str.lower().str.strip()

    # Add LLM responses
    df['gpt3_5_response'] = df['Question'].apply(lambda q: get_llm_response(q, "gpt-3.5-turbo"))
    df['gpt_4_response'] = df['Question'].apply(lambda q: get_llm_response(q, "gpt-4-turbo"))

    # Prepare data for insertion into MongoDB
    response_docs = df.to_dict(orient='records')

    # Insert data into 'responses_token' collection
    response_collection.insert_many(response_docs)

    # Print confirmation
    print(f"Inserted {len(response_docs)} documents into the ", "web_responses_token" if frontEndRequest else "responses_token", " collection.")

    # Optional: Save the processed dataset to CSV for local verification
    output_file = 'web_responses.csv'
    if os.path.exists(output_file):
        print(f"The file {output_file} already exists.")
    else:
        print(f"The file {output_file} has been created.")
    df.to_csv(output_file, index=False)

    print("Processing complete.")
    data = list(response_collection.find({}))
    for record in data:
        record['_id'] = str(record['_id'])
    return jsonify(data)

@app.route('/api/webResults', methods=['GET'])
def get_Webdata():
    collection = db["web_responses_token"]
    data = list(collection.find({}))
    for record in data:
        record['_id'] = str(record['_id'])  # Convert ObjectId to string
    return jsonify(data)

#evaluation methods

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.astype(float).tolist()

# Function to compute cosine similarity
def compute_cosine_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    # Convert lists back to numpy arrays
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    
    result = cosine_similarity(embedding1, embedding2)[0][0]
    return round(result, 2)

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

# 4. BERTScore
def calculate_BERTScore(references,predictions):
    # BERTScore expects lists of strings as input
    references = [references]  # Wrap the reference in a list
    predictions = [predictions]  # Wrap the prediction in a list
    P, R, F1 = score(predictions, references, lang='en', verbose=True)
    return round(F1.mean().item(),2) 

def contextual_relevance(question_embedding, llm_response_embedding):
    cont_rel = compute_cosine_similarity(question_embedding, llm_response_embedding)
    return round(cont_rel,2)

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


@app.route('/api/evaluate', methods=['POST'])
def evaluateAll():
    # Retrieve all records in the collection
    payload = request.get_json()

    # Check if 'frontEndRequest' key is in the payload
    if 'frontEndRequest' not in payload:
        return jsonify({'error': 'Missing type parameter in request payload'}), 400
    
    frontEndRequest = payload['frontEndRequest']

    collection = db["web_responses_token"]
    records = list(collection.find({}))
    
    if frontEndRequest == "False":
        for record in records:
            record['_id'] = str(record['_id'])
        return jsonify(records),200

    # Iterate through each record and calculate scores
    for record in records:
        question = record['Question']
        ideal_answer = record['Answer']
        gpt3_response = record['gpt3_5_response']
        gpt4_response = record['gpt_4_response']

        # Calculate BERT embeddings for the ideal answer and both model responses
        ideal_answer_embedding = get_bert_embedding(ideal_answer)
        gpt3_embedding = get_bert_embedding(gpt3_response)
        gpt4_embedding = get_bert_embedding(gpt4_response)    

        # Factual verification using GPT-4 for both responses
        gpt3_factual = llmFactualVerification(question, gpt3_response)
        gpt4_factual = llmFactualVerification(question, gpt4_response)

        # Calculate scores for GPT-3.5 response
        gpt3score = {
            "hypothesis": gpt3_response,
            "bleu": calculate_bleu(ideal_answer, gpt3_response),
            "meteor": calculate_meteor(ideal_answer, gpt3_response),
            "rouge": calculate_rouge(ideal_answer, gpt3_response),
            "semantic": compute_cosine_similarity(ideal_answer_embedding, gpt3_embedding),
            "bert_score": calculate_BERTScore(ideal_answer, gpt3_response),  
            "perplexity": calculate_perplexity(gpt2model, gpt2tokenizer, gpt3_response),
            "factual": gpt3_factual,
        }

        # Calculate scores for GPT-4 response
        gpt4score = {
            "hypothesis": gpt4_response,
            "bleu": calculate_bleu(ideal_answer, gpt4_response),
            "meteor": calculate_meteor(ideal_answer, gpt4_response),
            "rouge": calculate_rouge(ideal_answer, gpt4_response),
            "semantic": compute_cosine_similarity(ideal_answer_embedding, gpt4_embedding),
            "bert_score": calculate_BERTScore(ideal_answer, gpt4_response),
            "perplexity": calculate_perplexity(gpt2model, gpt2tokenizer, gpt4_response),
            "factual": gpt4_factual,
        }

        # Compute composite scores
        gpt3_composite_score = compute_composite_score(
            gpt3score["semantic"],
            gpt3score["factual"],
            gpt3score["bert_score"],
            weights
        )

        gpt4_composite_score = compute_composite_score(
            gpt4score["semantic"],
            gpt4score["factual"],
            gpt4score["bert_score"],
            weights
        )        

        # Add calculated scores back into the record document
        evaluated_results = {
            "gpt3_results": gpt3score,
            "gpt3_composite_score": gpt3_composite_score,
            "gpt4_results": gpt4score,
            "gpt4_composite_score": gpt4_composite_score
        }


        # Update the document in MongoDB
        collection.update_one(
            {"_id": record["_id"]},
            {"$set": evaluated_results}
        )

    return jsonify({"status": "Scoring and updates complete for all records"}), 200


@app.route('/api/score', methods=['POST'])
def generateScore():
    payload = request.get_json()
    required = ['question', 'ideal_answer', 'gpt_4_response', 'gpt_3_5_response']
    if required[0] and required[1] and required[2] and required[3] not in payload:
        return jsonify({'error': 'Missing required parameters in request payload'}), 400
    question = payload['question']
    ideal_answer = payload['ideal_answer']
    gpt3_response = payload['gpt-3.5-response']
    gpt4_response= payload['gpt-4-response']

    gpt3score = {
        "hypothesis" : gpt3_response,
        "bleu" : calculate_bleu(ideal_answer, gpt3_response),
        "meteor" : calculate_meteor(ideal_answer, gpt3_response),
        "rouge" : calculate_rouge(ideal_answer, gpt3_response)
    }

    gpt4score = {
        "hypothesis": gpt4_response,
        "bleu" : calculate_bleu(ideal_answer, gpt4_response),
        "meteor" : calculate_meteor(ideal_answer, gpt4_response),
        "rouge" : calculate_rouge(ideal_answer, gpt4_response)
    }

    evaluated_results = {
        "question" : question,
        "ideal_answer" : ideal_answer,
        "gpt3-results" : gpt3score,
        "gpt4-results" : gpt4score
    }
    
    return jsonify(evaluated_results),200

@app.route('/api/evalOne', methods=['POST'])
def evalOne():
    payload = request.get_json()
    required = ['question', 'ideal_answer', 'llm_response']
    if required[0] and required[1] and required[2] not in payload:
        return jsonify({'error': 'Missing required parameters in request payload'}), 400
    question = payload['question']
    ideal_answer = payload['ideal_answer']
    llm_response = payload['llm_response']
    print(question, ideal_answer, llm_response)
    
    llmScore = {
        "hypothesis" : llm_response,
        "bleu" : calculate_bleu(ideal_answer, llm_response),
        "meteor" : calculate_meteor(ideal_answer, llm_response),
        "rouge" : calculate_rouge(ideal_answer, llm_response)
    }

    evaluated_results = {
        "question" : question,
        "ideal_answer" : ideal_answer,
        "llm-results" : llmScore,
    }
    
    return jsonify(evaluated_results),200

#old code

@app.route('/api/data', methods=['GET'])
def get_data():
    collection = db["composite_scores"]
    data = list(collection.find({}))
    for record in data:
        record['_id'] = str(record['_id'])  # Convert ObjectId to string
    return jsonify(data)

@app.route('/api/testdata', methods=['GET'])
def get_testdata():
    collection = db["question_answers"]
    data = list(collection.find({}))
    for record in data:
        record['_id'] = str(record['_id'])  # Convert ObjectId to string
    return jsonify(data)

@app.route('/api/generate-answers', methods=['POST'])
def generateAnswers():
    payload = request.get_json()
    required = ['question', 'ideal_answer']
    if required[0] and required[1] not in payload:
        return jsonify({'error': 'Missing question and ideal answer parameter in request payload'}), 400
    question = payload['question']
    ideal_answer = payload['ideal_answer']

    gpt3pt5_response = get_llm_response(question, 'gpt-3.5-turbo')
    gpt4_response = get_llm_response(question, 'gpt-4-turbo')

    llm_responses = {
        "question" : question,
        "ideal_answer" : ideal_answer,
        "gpt-4-response": gpt4_response,
        "gpt-3.5-response": gpt3pt5_response
    }

    return jsonify(llm_responses),200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
