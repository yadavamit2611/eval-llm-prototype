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

app = Flask(__name__)
CORS(app)

# Configure MongoDB connection
mongo_uri = "mongodb://localhost:27017/"
client = MongoClient(mongo_uri)
db = client["eval_db"]

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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


@app.route('/api/generate',methods=['POST'])
def generate():
    payload = request.get_json()

        # Check if 'type' key is in the payload
    if 'dummy' not in payload:
        return jsonify({'error': 'Missing type parameter in request payload'}), 400
    
    dummy = payload['dummy']

    if(dummy=="True"):
        response_collection = db["dummy_responses_token"]
        df = pd.read_csv('uploads/politics-dataset.csv')
    else:
        collection = db["question_answers"]
        response_collection = db["responses_token"]
            # Extract data from MongoDB
        data = list(collection.find({}))
        df = pd.DataFrame(data)

    # Data Cleaning and Standardization
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    df['Question'] = df['Question'].str.lower().str.strip()
    df['Answer'] = df['Answer'].str.lower().str.strip()

    # Tokenization and Normalization
    df['Question_tokens'] = df['Question'].apply(preprocess_text)
    df['Answer_tokens'] = df['Answer'].apply(preprocess_text)

    # Add LLM responses
    df['gpt3.5-turbo_response'] = df['Question'].apply(lambda q: get_llm_response(q, "gpt-3.5-turbo"))
    df['gpt-4-turbo_response'] = df['Question'].apply(lambda q: get_llm_response(q, "gpt-4-turbo"))

    df['gpt3.5-turbo_response_tokens'] = df['gpt3.5-turbo_response'].apply(preprocess_text)
    df['gpt-4-turbo_response_tokens'] = df['gpt-4-turbo_response'].apply(preprocess_text)

    # Prepare data for insertion into MongoDB
    response_docs = df.to_dict(orient='records')

    # Insert data into 'responses_token' collection
    response_collection.insert_many(response_docs)

    # Print confirmation
    print(f"Inserted {len(response_docs)} documents into the ","dummy_responses_token" if dummy else "responses_token"," collection.")

    # Optional: Save the processed dataset to CSV for local verification
    output_file = 'responses.csv'
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

#evaluation methods


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


@app.route('/api/score', methods=['POST'])
def generateScore():
    payload = request.get_json()
    required = ['question', 'ideal_answer', 'gpt-4-response', 'gpt-3.5-response']
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
