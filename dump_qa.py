from pymongo import MongoClient
from datasets import load_dataset
import pandas as pd

# Connect to MongoDB (replace with your MongoDB URI if necessary)
client = MongoClient("mongodb://localhost:27017/")
db = client["eval_db"]

# Ask the user to select the dataset
dataset_choice = input("Which dataset would you like to load? Enter 'squad' or 'truthful_qa': ").strip().lower()

if dataset_choice == "squad":
    # Load the SQuAD dataset
    squad = load_dataset('squad')
    collection = db["squad2_qa"]
    
    # Extract 50 questions and answers from the training dataset
    train_data = squad['train']
    
    # Create a list to store the questions, answers, and context
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
    
    # Insert into MongoDB
    collection.insert_many(qa_pairs)
    print("SQuAD data inserted successfully in MongoDB.")

elif dataset_choice == "truthful_qa":
    # Load the TruthfulQA dataset
    truthful_qa = load_dataset("truthful_qa","generation")
    collection = db["truthful_qa"]
    
    # Extract questions, answers, and metadata from the training split
    train_data = truthful_qa['validation']  # TruthfulQA may only have a validation set
    
    # Create a list to store the questions and answers
    qa_pairs = []
    
    for example in train_data:
        sector = example['category']
        question = example['question']
        answer = example['best_answer']  # This may vary based on dataset structure
        
        # Store the question and answer in a dictionary
        qa_pairs.append({
            'Sector' : sector,
            'Question': question,
            'Answer': answer
        })
    
    # Insert into MongoDB
    collection.insert_many(qa_pairs)
    print("TruthfulQA data inserted successfully in MongoDB.")
    
    # Optionally save to CSV
    save_to_csv = input("Would you like to save the TruthfulQA data to a CSV file? (yes/no): ").strip().lower()
    if save_to_csv == 'yes':
        df = pd.DataFrame(qa_pairs)
        df.to_csv("truthful_qa.csv", index=False)
        print("TruthfulQA data saved to truthful_qa.csv.")

else:
    print("Invalid choice. Please enter either 'squad' or 'truthful_qa'.")
