import gradio as gr
from transformers import BertForQuestionAnswering, AutoTokenizer
import torch

# Load the fine-tuned model and tokenizer
model_path = "./bert_fine_tuned"  # Path to the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BertForQuestionAnswering.from_pretrained(model_path)

# Function to handle user input
def answer_question(context, question):
    # Tokenize the inputs
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].tolist()[0]

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the start and end logits
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Find the most likely start and end of the answer
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)

    # Debugging logs
    sequence_ids = inputs["token_type_ids"].tolist()[0]
    context_start = sequence_ids.index(1)  # Start of the context
    context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)  # End of the context
    print(f"Context starts at: {context_start}, ends at: {context_end}")
    print(f"Predicted start index: {start_index}, end index: {end_index}")
    print(f"Start token: {input_ids[start_index]} ({tokenizer.decode([input_ids[start_index]])})")
    print(f"End token: {input_ids[end_index]} ({tokenizer.decode([input_ids[end_index]])})")

    # Clamp indices to ensure they are within the context range
    start_index = max(context_start, start_index)
    end_index = min(context_end, end_index)

    if start_index > end_index:
        return "Unable to find a valid answer."

    # Extract and decode the answer
    answer_tokens = input_ids[start_index:end_index + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    # Avoid returning invalid answers
    if not answer or answer in question:
        return "Unable to find a valid answer."

    return answer

# Gradio Interface
interface = gr.Interface(
    fn=answer_question,  # Function to be called when the user submits input
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter the context here", label="Context"),  # Text input for the context
        gr.Textbox(lines=1, placeholder="Enter your question here", label="Question")  # Text input for the question
    ],
    outputs="text",  # Output will be a text response (the answer)
    title="BERT Question Answering Model",  # Title of the app
    description="Ask any question based on the provided context. The model will extract an answer from the context."
)

if __name__ == "__main__":
    # Launch Gradio's built-in server
    interface.launch(server_name="0.0.0.0", server_port=8080)  