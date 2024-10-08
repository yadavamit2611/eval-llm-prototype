import openai

""" client = OpenAI() """

completion = openai.ChatCompletion.create(
  model="gpt-4-turbo",
  messages=[
    {"role": "system", "content": "You are a judge who is responds with just one numerical value in between 0.0 to 1.0 based on the claim provided"},
    {"role": "user", "content": "lionel messi is the greatest player in the football history"}
  ]
)

print(completion.choices[0].message.content)