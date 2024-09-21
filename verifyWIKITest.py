import wikipediaapi
from transformers import RobertaTokenizer, RobertaModel
import torch
from scipy.spatial.distance import cosine

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

# Function to get embeddings
def get_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Function to compare claim with each sentence in evidence
def compare_claim_with_evidence(claim, evidence, tokenizer, model):
    claim_embedding = get_embeddings(claim, tokenizer, model)
    sentences = evidence.split('. ')
    max_similarity = -1
    best_sentence = None

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            sentence_embedding = get_embeddings(sentence, tokenizer, model)
            similarity = 1 - cosine(claim_embedding, sentence_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                best_sentence = sentence

    return best_sentence, max_similarity

# Main function to run fact-checking pipeline
def fact_check_claim(claim, topic):
    print(f"Claim: {claim}")

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    model = RobertaModel.from_pretrained("roberta-large")

    # Retrieve evidence from Wikipedia
    evidence = get_wikipedia_summary(topic)
    if evidence:
        print(f"Evidence from Wikipedia:\n{evidence}...\n")  # Limiting output for readability

        # Compare claim with evidence sentences
        best_sentence, max_similarity = compare_claim_with_evidence(claim, evidence, tokenizer, model)

        if best_sentence:
            print(f"Best Matching Sentence: {best_sentence}")
            print(f"Similarity Score: {max_similarity:.2f}")
        else:
            print("No relevant sentences found.")
    else:
        print("No relevant evidence found on Wikipedia.")

# Example claim
claim = "Tempura is a Japanese cooking technique where seafood or vegetables are coated in a light batter and deep-fried until crispy"
topic = "tempura"

# Run fact-checking
fact_check_claim(claim, topic)
