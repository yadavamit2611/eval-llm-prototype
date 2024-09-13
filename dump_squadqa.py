from pymongo import MongoClient
from datasets import load_dataset

# Replace the URI with your MongoDB URI if you're using a cloud service
client = MongoClient("mongodb://localhost:27017/")
db = client["eval_db"] 
collection = db["squad2_qa"] 

# Load the SQuAD v1 dataset
squad = load_dataset('squad')

# Extract 50 questions and answers from the training dataset
train_data = squad['train']

# Create a list to store the questions and answers
qa_pairs = []

# Iterate over the first 50 examples
for i in range(50):
    example = train_data[i]
    question = example['question']
    answer = example['answers']['text'][0]  # The answer is a list, take the first one
    context = example['context']
    
    # Store the question, answer, and context in a dictionary
    qa_pairs.append({
        'question': question,
        'answer': answer,
        'context': context
    })

collection.insert_many(qa_pairs)
print("Data inserted successfully in ", collection)
