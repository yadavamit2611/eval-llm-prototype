import pymongo
import json
from bson import json_util

# Connect to the MongoDB server
client = pymongo.MongoClient("mongodb://localhost:27017/")

# Access the specific database and collection
db = client["eval_db"]
collection = db["composite_scores"]

# Open a file to write to (in JSONL format)
with open("output.jsonl", "w") as jsonl_file:
    # Fetch records from the collection
    cursor = collection.find({})
    for document in cursor:
        # Convert MongoDB's BSON format to JSON using json_util for proper serialization
        jsonl_file.write(json_util.dumps(document) + "\n")

print("Data successfully exported to JSONL format.")
