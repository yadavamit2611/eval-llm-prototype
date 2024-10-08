from transformers import BertModel, BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel
import torch
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import json
from sklearn.metrics.pairwise import cosine_similarity
import wikipediaapi
import numpy as np

# Load pre-trained BERT model and tokenizer
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BertModel.from_pretrained(bert_model_name)

# Function to compute BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

model_name = "gpt2"
gpt2tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt2model = GPT2LMHeadModel.from_pretrained(model_name)
gpt2model.eval()


# Function to calculate BLEU score
def calculate_bleu(reference, hypothesis):
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    return round(sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=SmoothingFunction().method1),2)

# Function to calculate ROUGE score
def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

# Function to calculate METEOR score
def calculate_meteor(reference, hypothesis):
    reference = ' '.join(reference) if isinstance(reference, list) else reference
    hypothesis = ' '.join(hypothesis) if isinstance(hypothesis, list) else hypothesis
    return round(meteor_score([reference], hypothesis),2)

# 4. BERTScore
def calculate_BERTScore(references,predictions):
    # BERTScore expects lists of strings as input
    references = [references]  # Wrap the reference in a list
    predictions = [predictions]  # Wrap the prediction in a list
    P, R, F1 = score(predictions, references, lang='en', verbose=True)
    return round(F1.mean().item(),2) 


def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs.input_ids

    # Disable gradient calculation to speed up inference
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    # Get the loss (cross-entropy loss) from the output
    loss = outputs.loss
    perplexity = torch.exp(loss)
    
    return round(perplexity.item(),2)

# Initialize Wikipedia API with a proper User-Agent
def get_wikipedia_summary(query):
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent="FactChecker/1.0 (factchecker@example.com)"
    )
    page = wiki_wiki.page(query)
    if page.exists():
        return page.summary
    return None

# Function to compare claim with each sentence in evidence
def compare_claim_with_evidence(claim, evidence):
    claim_embedding = get_bert_embedding(claim)
    sentences = evidence.split('. ')
    max_similarity = -1
    best_sentence = None

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            sentence_embedding = get_bert_embedding(sentence)
            similarity = 1 - compute_cosine_similarity(claim_embedding, sentence_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                best_sentence = sentence

    return best_sentence, max_similarity

# Main function to run fact-checking pipeline
def fact_check_claim(claim, topic):
    #print(f"Claim: {claim}")

    # Retrieve evidence from Wikipedia
    evidence = get_wikipedia_summary(topic)
    if evidence:
        """ print(f"Evidence from Wikipedia:\n{evidence}...\n")  # Limiting output for readability """

        # Compare claim with evidence sentences
        best_sentence, max_similarity = compare_claim_with_evidence(claim, evidence)

        if best_sentence:
            print(f"Best Matching Sentence: {best_sentence}")
            print(f"Similarity Score: {max_similarity:.2f}")
        else:
            print("No relevant sentences found.")
    else:
        print("No relevant evidence found on Wikipedia.")

    return round(max_similarity,2)

# Placeholder functions for factual verification and contextual relevance
def factual_verification(topic, llm_response):
   """  topic = extract_named_entities(sector)
    print(topic, llm_response) """
   result = fact_check_claim(llm_response, topic)
   return result

# Function to compute cosine similarity
def compute_cosine_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return 0.0
    embedding1 = embedding1.reshape(1, -1) if embedding1.ndim == 1 else embedding1
    embedding2 = embedding2.reshape(1, -1) if embedding2.ndim == 1 else embedding2
    result = cosine_similarity(embedding1, embedding2)[0][0]
    return round(result,2)

import openai
#verify from gpt-4
def llmFactualVerification(claim):
    completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a judge who responds with just one numerical value in between 0.0 to 1.0 based on the claim provided"},
        {"role": "user", "content": claim}
    ]
    )
    return round(float(completion.choices[0].message.content),2)

def contextual_relevance(question_embedding, llm_response_embedding):
    cont_rel = compute_cosine_similarity(question_embedding, llm_response_embedding)
    return round(cont_rel,2)

# Weights for the composite score
weights = {
    'semantic': 0.4,
    'factual': 0.4,
    'contextual': 0.2
}

# Function to compute composite score
def compute_composite_score(semantic, factual, contextual, weights):
    return round((weights['semantic'] * semantic +
            weights['factual'] * factual +
            weights['contextual'] * contextual),2)


og_question = "what is an algorithm?"
processed_question = "algorithm"

og_idealanswer = "An algorithm is a step-by-step procedure or formula for solving a problem or performing a task in computing"
processed_idealanswer = "algorithm procedure formula solving problem performing task computing"

og_gpt3response = "An algorithm is a step-by-step procedure or set of rules to be followed in calculations or problem-solving operations."
processed_gpt3response = "An algorithm procedure formula solving problem accomplishing task"

og_gpt4response = "An algorithm is a set of step-by-step instructions or rules designed to perform a specific task or solve a particular problem."
processed_gpt4response = "An algorithm set instructions designed perform specific task solve particular problem"

og_IdealAnswer_perplexity = calculate_perplexity(gpt2model, gpt2tokenizer, og_idealanswer)
og_gpt3response_perplexity = calculate_perplexity(gpt2model, gpt2tokenizer, og_gpt3response)
og_gpt4response_perplexity = calculate_perplexity(gpt2model, gpt2tokenizer, og_gpt4response)

processed_IdealAnswer_perplexity = calculate_perplexity(gpt2model, gpt2tokenizer, processed_idealanswer)
processed_gpt3response_perplexity = calculate_perplexity(gpt2model, gpt2tokenizer, processed_gpt3response)
processed_gpt4response_perplexity = calculate_perplexity(gpt2model, gpt2tokenizer, processed_gpt4response)

og_gpt3bleu = calculate_bleu(og_idealanswer, og_gpt3response)
og_gpt4bleu = calculate_bleu(og_idealanswer, og_gpt4response)

processed_gpt3bleu = calculate_bleu(processed_idealanswer, processed_gpt3response)
processed_gpt4bleu = calculate_bleu(processed_idealanswer, processed_gpt4response)

og_gpt3rouge = calculate_rouge(og_idealanswer, og_gpt3response)
og_gpt4rouge = calculate_rouge(og_idealanswer, og_gpt4response)

processed_gpt3rouge = calculate_rouge(processed_idealanswer, processed_gpt3response)
processed_gpt4rouge = calculate_rouge(processed_idealanswer, processed_gpt4response)

og_gpt3meteor = calculate_meteor(og_idealanswer, og_gpt3response)
og_gpt4meteor = calculate_meteor(og_idealanswer, og_gpt4response)

processed_gpt3meteor = calculate_meteor(processed_idealanswer, processed_gpt3response)
processed_gpt4meteor = calculate_meteor(processed_idealanswer, processed_gpt4response)

og_gpt3bscore = calculate_BERTScore(og_idealanswer, og_gpt3response)
og_gpt4bscore = calculate_BERTScore(og_idealanswer, og_gpt4response)

processed_gpt3bscore = calculate_BERTScore(processed_idealanswer, processed_gpt3response)
processed_gpt4bscore = calculate_BERTScore(processed_idealanswer, processed_gpt4response)

og_gpt3ss = compute_cosine_similarity(get_bert_embedding(og_idealanswer), get_bert_embedding(og_gpt3response))
og_gpt4ss = compute_cosine_similarity(get_bert_embedding(og_idealanswer), get_bert_embedding(og_gpt4response))

processed_gpt3ss = compute_cosine_similarity(get_bert_embedding(processed_idealanswer), get_bert_embedding(processed_gpt3response))
processed_gpt4ss = compute_cosine_similarity(get_bert_embedding(processed_idealanswer), get_bert_embedding(processed_gpt4response))

og_gpt3llmfv = llmFactualVerification(og_gpt3response)
og_gpt4llmfv = llmFactualVerification(og_gpt4response)

processed_gpt3llmfv = llmFactualVerification(processed_gpt3response)
processed_gpt4llmfv = llmFactualVerification(processed_gpt4response)

og_gpt3fv = factual_verification(processed_question,og_gpt3response)
og_gpt4fv = factual_verification(processed_question,og_gpt4response)

processed_gpt3fv = factual_verification(processed_question,processed_gpt3response)
processed_gpt4fv = factual_verification(processed_question,processed_gpt4response)

og_gpt3cr = contextual_relevance(get_bert_embedding(og_idealanswer), get_bert_embedding(og_gpt3response))
og_gpt4cr = contextual_relevance(get_bert_embedding(og_idealanswer), get_bert_embedding(og_gpt4response))

processed_gpt3cr = contextual_relevance(get_bert_embedding(processed_idealanswer), get_bert_embedding(processed_gpt3response))
processed_gpt4cr = contextual_relevance(get_bert_embedding(processed_idealanswer), get_bert_embedding(processed_gpt4response))

og_gpt3ocs = compute_composite_score(og_gpt3ss,og_gpt3fv,og_gpt3cr, weights)
og_gpt4ocs = compute_composite_score(og_gpt4ss,og_gpt4fv,og_gpt4cr, weights)

processed_gpt3ocs = compute_composite_score(processed_gpt3ss,processed_gpt3fv,processed_gpt3cr, weights)
processed_gpt4ocs = compute_composite_score(processed_gpt3ss,processed_gpt4fv,processed_gpt4cr, weights)

og_gpt3ncs = compute_composite_score(og_gpt3ss,og_gpt3llmfv,og_gpt3bscore, weights)
og_gpt4ncs = compute_composite_score(og_gpt4ss,og_gpt4llmfv,og_gpt4bscore, weights)

processed_gpt3ncs = compute_composite_score(processed_gpt3ss,processed_gpt3llmfv,processed_gpt3bscore, weights)
processed_gpt4ncs = compute_composite_score(processed_gpt3ss,processed_gpt4llmfv,processed_gpt4bscore, weights)

""" og_gpt3fv = factual_verification(og_gpt3response)
og_gpt4fv = factual_verification(og_gpt4response)

processed_gpt3fv = factual_verification(processed_gpt3response)
processed_gpt4fv = factual_verification(processed_gpt4response) """


results = {
        "question": {
            "original": og_question,
            "processed": processed_question
        },
        "ideal_answer":{
            "original": og_idealanswer,
            "processed": processed_idealanswer
        },
        "gpt3-5-answer": {
            "original": og_gpt3response,
            "processed": processed_gpt3response
        },
        "gpt4-answer": {
            "original": og_gpt4response,
            "processed": processed_gpt4response
        },
        "perplexity" : {
            "gpt3-5-perplexity": {
                "original" : og_gpt3response_perplexity,
                "processed" : processed_gpt3response_perplexity
            },
            "gpt4-perplexity": {
                "original" : og_gpt4response_perplexity,
                "processed" : processed_gpt4response_perplexity
            }
        },
        "bleu":{
            "gpt3-5-bleu": {
                "original" : og_gpt3bleu,
                "processed" : processed_gpt3bleu
            },
            "gpt4-bleu": {
                "original" : og_gpt4bleu,
                "processed" : processed_gpt4bleu
            }
        },
        "rouge":{
            "gpt3-5-rouge": {
                "original" : og_gpt3rouge,
                "processed" : processed_gpt3rouge
            },
            "gpt4-rouge": {
                "original" : og_gpt4rouge,
                "processed" : processed_gpt4rouge
            }
        },
        "meteor":{
            "gpt3-5-meteor": {
                "original" : og_gpt3meteor,
                "processed" : processed_gpt3meteor
            },
            "gpt4-meteor": {
                "original" : og_gpt4meteor,
                "processed" : processed_gpt4meteor
            }
        },
        "bertscore":{
            "gpt3-5-bscore": {
                "original" : og_gpt3bscore,
                "processed" : processed_gpt3bscore
            },
            "gpt4-bscore": {
                "original" : og_gpt4bscore,
                "processed" : processed_gpt4bscore
            }
        },
        "semantic similartiy":{
            "gpt3-5-ss": {
                "original" : str(og_gpt3ss),
                "processed" : str(processed_gpt3ss)
            },
            "gpt4-ss": {
                "original" : str(og_gpt4ss),
                "processed" : str(processed_gpt4ss)
            }
        },
        "factual verification":{
            "gpt3-5-fv": {
                "original" : og_gpt3fv,
                "processed" : processed_gpt3fv
            },
            "gpt4-fv": {
                "original" : og_gpt4fv,
                "processed" : processed_gpt4fv
            }
        },
        "llm factual verification":{
            "gpt3-5-llmfv": {
                "original" : og_gpt3llmfv,
                "processed" : processed_gpt3llmfv
            },
            "gpt4-llmfv": {
                "original" : og_gpt4llmfv,
                "processed" : processed_gpt4llmfv
            }
        },
        "Contextual relevance":{
            "gpt3-5-cr": {
                "original" : str(og_gpt3cr),
                "processed" : str(processed_gpt3cr)
            },
            "gpt4-cr": {
                "original" : str(og_gpt4cr),
                "processed" : str(processed_gpt4cr)
            }
        },
        "old-Composite-score":{
            "gpt3-5-ocs": {
                "original" : str(og_gpt3ocs),
                "processed" : str(processed_gpt3ocs)
            },
            "gpt4-ocs": {
                "original" : str(og_gpt4ocs),
                "processed" : str(processed_gpt4ocs)
            }
        },
        "new-Composite-score":{
            "gpt3-5-ncs": {
                "original" : str(og_gpt3ncs),
                "processed" : str(processed_gpt3ncs)
            },
            "gpt4-ncs": {
                "original" : str(og_gpt4ncs),
                "processed" : str(processed_gpt4ncs)
            }
        }                                     
}
# Convert to JSON and save to a file
with open('result.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Data successfully written to result.json")

# gpt3response 23.69
# gpt4response 12.85