import re
from transformers import BertForQuestionAnswering, BertTokenizerFast
import torch
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json

# Load the JSON file
with open('context.json', 'r') as f:
    dataset = json.load(f)
    
# Load the fine-tuned model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('model')

def generate_answer(question, context, tokenizer, model):
    # Tokenize question and context
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get the model's predictions
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Identify the answer span
    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores) + 1

    # Decode the answer
    answer = tokenizer.decode(input_ids[0][start_idx:end_idx])

    return answer



# Combine all contexts from the dataset
contexts = [entry['context'] for entry in dataset]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
context_matrix = vectorizer.fit_transform(contexts)

def retrieve_context(question, vectorizer, context_matrix, contexts, top_n=1):
    # Transform the question into the same vector space
    question_vec = vectorizer.transform([question])

    # Compute cosine similarity between the question and all contexts
    similarities = (context_matrix @ question_vec.T).toarray().flatten()

    # Get the most relevant context(s)
    best_indices = np.argsort(similarities)[-top_n:][::-1]
    return [contexts[i] for i in best_indices]


def chatbot(question):
    # Step 1: Retrieve the most relevant context
    retrieved_contexts = retrieve_context(question, vectorizer, context_matrix, contexts)
    context = retrieved_contexts[0]  # Use the top context
    print("Retrieved context:",context)

    # Step 2: Generate the answer
    answer = generate_answer(question, context, tokenizer, model)

    # Step 3: Return the answer
    return answer


# Simulate user input
user_question = "What was the income before income taxes for the Financial Services segment in Q3 2024?"

# Get the chatbot's response
response = chatbot(user_question)
print(f"Chatbot: {response}")