import openai
import os

print(os.getenv("OPENAI_API_KEY"))

""" client = OpenAI() """
question = "what is an algorithm?"
claim = "An algorithm is a step-by-step procedure or formula for solving a problem or performing a task in computing"

prompt = f"""
I want you to act as a factual verifier. You will be given a question and an answer.
Your task is to evaluate whether the answer is factually correct based on your knowledge.
Return the factual correctness score only as a numerical value between 0.0 and 1.0. Do not add any additional text after the score.

For the given question:

Question: {question}

Answer: {claim}

On a scale of 0.0 to 1.0, where 1.0 means the answer is completely factually correct and 0.0 means the answer is completely factually incorrect, give a factual correctness score. Provide a brief explanation for your rating.
"""

completion = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "You are a judge who is responds with just one numerical value in between 0.0 to 1.0 based on the prompt provided"},
    {"role": "user", "content": prompt}
  ],
  temperature=0.5,
)

print(completion.choices[0].message.content)