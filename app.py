# app.py
from flask import Flask, jsonify
from pymongo import MongoClient

app = Flask(__name__)

# Configure MongoDB connection
mongo_uri = "mongodb://localhost:27017/"
client = MongoClient(mongo_uri)
db = client["eval_db"]
collection = db["dummy_composite_scores"]

@app.route('/api/data', methods=['GET'])
def get_data():
    data = list(collection.find({}))
    for record in data:
        record['_id'] = str(record['_id'])  # Convert ObjectId to string
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
