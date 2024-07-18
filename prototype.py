import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

# Load the dataset
data = pd.read_csv('politics-dataset.csv')

# Data Cleaning
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# Standardization
data['question'] = data['question'].str.lower().str.strip()
data['ideal_answer'] = data['ideal_answer'].str.lower().str.strip()

# Tokenization and Normalization
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

data['question_tokens'] = data['question'].apply(preprocess_text)
data['ideal_answer_tokens'] = data['ideal_answer'].apply(preprocess_text)

# Save the preprocessed dataset
data.to_csv('preprocessed_qa.csv', index=False)

from openai import OpenAI

client = OpenAI()

# Function to get LLM response
def get_llm_response(question,test_model):
    completion = client.chat.completions.create(
    model=test_model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Provide short and concise answers"},
        {"role": "user", "content": question}
    ]
    )
    return completion.choices[0].message.content


# Check if the 'Response' column exists, if not create it
if 'gpt3.5-turbo_response' not in data.columns:
    data['gpt3.5-turbo_response'] = ''
if 'gpt-4-turbo_response' not in data.columns:
    data['gpt-4-turbo_response'] = ''

qa_counter = 0
# Iterate through the questions and get responses

for index, row in data.iterrows():
    question = row['question']
    response1 = get_llm_response(question,"gpt-3.5-turbo")
    response2 = get_llm_response(question,"gpt-4-turbo")
    data.at[index, 'gpt3.5-turbo_response'] = response1
    data.at[index, 'gpt-4-turbo_response'] = response2

    if(question != "" and response1 != "" and response2 != ""):
        qa_counter = qa_counter + 1

# Save responses back to Excel
output_file = 'responses.csv'

# Check if the file already exists
if os.path.exists(output_file):
    print(f"The file {output_file} already exists.")
    data.to_csv(output_file, index=False)
    # Decide what to do if it exists (e.g., overwrite, append, create a new file with a different name)
else:
    data.to_csv(output_file, index=False)
    print(f"The file {output_file} has been created.")
print(f"{qa_counter} responses generated")

