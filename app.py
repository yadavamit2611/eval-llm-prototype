# app.py
from flask import Flask, jsonify
from pymongo import MongoClient

app = Flask(__name__)

# Configure MongoDB connection
mongo_uri = "mongodb://localhost:27017/"
client = MongoClient(mongo_uri)
db = client["eval_db"]


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
    app.run(debug=True)
