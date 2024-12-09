import gradio as gr
from transformers import BertForQuestionAnswering, BertTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
import json

# Load the fine-tuned BERT model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('model')

# Load the structured context JSON
with open("context.json", "r") as file:
    context_sections = json.load(file)

# Combine all contexts into a list
contexts = [section['context'] for section in context_sections]

# Create a TF-IDF vectorizer and fit it on the contexts
vectorizer = TfidfVectorizer(stop_words='english')
context_matrix = vectorizer.fit_transform(contexts)

# Function to retrieve the most relevant context using cosine similarity
def retrieve_context(question, vectorizer, context_matrix, contexts, top_n=1):
    # Transform the question into the same vector space
    question_vec = vectorizer.transform([question])

    # Compute cosine similarity between the question and all contexts
    similarities = (context_matrix @ question_vec.T).toarray().flatten()

    # Get the most relevant context(s)
    best_indices = np.argsort(similarities)[-top_n:][::-1]
    return [contexts[i] for i in best_indices]

# Function to generate an answer using the fine-tuned BERT model
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

# Function to answer a question
def answer_question(question):
    # Retrieve the most relevant context
    retrieved_contexts = retrieve_context(question, vectorizer, context_matrix, contexts)
    relevant_context = retrieved_contexts[0]  # Use the top context

    # Log the selected context
    print("\n--- Selected Context ---")
    print(relevant_context)

    if not relevant_context:
        return "I couldn't determine a relevant context for your question."

    # Generate the answer using the fine-tuned BERT model
    answer = generate_answer(question, relevant_context, tokenizer, model)
    
    print("\n--- Generated Answer ---")
    print(answer)
    
    return answer

# Create Gradio Interface
interface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text",
    title="Canadian Tire Corporation Q3 2024 Chatbot",
    description="Ask questions about the Canadian Tire Corporation Q3 2024 report. The chatbot dynamically selects the relevant context and provides answers.",
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)
