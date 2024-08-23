import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pymongo import MongoClient
import os

#set value to false when needed to test for big dataset
dummy = True

# MongoDB Configuration
mongo_uri = "mongodb://localhost:27017/"
client = MongoClient(mongo_uri)
db = client["eval_db"]
if(dummy):
    collection = db["dummy_data"]
    response_collection = db["dummy_responses_token"]
else:
    collection = db["question_answers"]
    response_collection = db["responses_token"]

# Tokenization and Normalization
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

from openai import OpenAI
client = OpenAI()

# Function to get LLM response
def get_llm_response(Question,test_model):
    completion = client.chat.completions.create(
    model=test_model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Provide short and concise answers"},
        {"role": "user", "content": Question}
    ]
    )
    return completion.choices[0].message.content

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

df = df.drop(columns=['gpt3.5-turbo_response', 'gpt-4-turbo_response'])

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
