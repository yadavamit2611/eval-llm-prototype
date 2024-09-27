# app.py
from flask import Flask, request, jsonify
from pymongo import MongoClient
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Configure MongoDB connection
mongo_uri = "mongodb://localhost:27017/"
client = MongoClient(mongo_uri)
db = client["eval_db"]


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
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    return jsonify({'message': 'File uploaded successfully', 'filename': file.filename}), 200


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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
