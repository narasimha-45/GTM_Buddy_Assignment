import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import json
import re
import logging
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.DEBUG)

# Load domain knowledge
domain_knowledge_path = "domain_knowledge.json"
with open(domain_knowledge_path, "r") as file:
    domain_knowledge = json.load(file)

# Define regex patterns for NER
regex_patterns = {
    "competitors": r"\bcompetitor\s*(\w+)",
    "pricing_keywords": r"(price|cost|discount|budget|affordable)",
    "features": r"(real-time|data|analytics|fraud detection|automation tools)"
}

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Input and output schema
class SnippetInput(BaseModel):
    snippet: str

class SnippetOutput(BaseModel):
    predicted_labels: list
    extracted_entities: dict
    summary: str

# Helper functions for entity extraction
def dictionary_lookup(text, knowledge_base):
    entities = {key: [] for key in knowledge_base.keys()}
    for key, terms in knowledge_base.items():
        for term in terms:
            if term.lower() in text.lower():
                entities[key].append(term)
    return entities

def regex_ner(text, patterns):
    entities = {key: [] for key in patterns.keys()}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        entities[key].extend(matches)
    return entities

def combine_entities(dict_entities, regex_entities):
    combined = {key: list(set(dict_entities[key] + regex_entities[key])) for key in dict_entities.keys()}
    return combined

label_mapping = {
    0: "competition",
    1: "features",
    2: "objection",
    3: "pricing discussion",
    4: "security"
}

# Define the ensemble prediction function
def predict_labels_ensemble(snippet, vectorizer, log_reg_model, svm_model):
    # Preprocess the input
    X = vectorizer.transform([snippet])
    # Get probabilities from both models
    log_reg_probs = log_reg_model.predict_proba(X)
    svm_probs = svm_model.predict_proba(X)
    # Combine probabilities and return labels
    ensemble_probs = (log_reg_probs + svm_probs) / 2
    threshold = 0.5
    labels = [
        
        label_mapping[i]  # Ensure labels are strings
        for i, prob in enumerate(ensemble_probs[0])
        if prob > threshold
    ]
    print(labels)
    return labels


# Load models and vectorizer
log_reg_model_path = "logistic_regression_model.pkl"
svm_model_path = "svm_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

log_reg_model = joblib.load(log_reg_model_path)
svm_model = joblib.load(svm_model_path)
vectorizer = joblib.load(vectorizer_path)

@app.post("/process", response_model=SnippetOutput)
def process_snippet(snippet_input: SnippetInput):
    snippet = snippet_input.snippet
    print(snippet)
    # Predict labels using ensemble logic
    predicted_labels = predict_labels_ensemble(snippet, vectorizer, log_reg_model, svm_model)
    # Extract entities
    dict_entities = dictionary_lookup(snippet, domain_knowledge)
    regex_entities = regex_ner(snippet, regex_patterns)
    extracted_entities = combine_entities(dict_entities, regex_entities)
    competitors = ', '.join(extracted_entities['competitors']) or "no competitors "
    features = ', '.join(extracted_entities['features']) or "no features "
    pricing_keywords = ', '.join(extracted_entities['pricing_keywords']) or "no pricing details "

    # Adjusted summary
    summary = f"The snippet mentions {competitors} and focuses on {features}.and {pricing_keywords}"
    print(summary)
    x = SnippetOutput(
        predicted_labels=predicted_labels,
        extracted_entities=extracted_entities,
        summary=summary
    )
    print(x)
    return x

