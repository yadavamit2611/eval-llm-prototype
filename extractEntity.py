import spacy

# Load the pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_named_entities(sentence):
    # Process the sentence with spaCy
    doc = nlp(sentence)
    
    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return entities

# Example sentence
sentence = "Saint Bernadette Soubirous"

# Extract named entities
named_entities = extract_named_entities(sentence)

print("Named Entities and their labels:")
for entity, label in named_entities:
    print(f"{entity}: {label}")