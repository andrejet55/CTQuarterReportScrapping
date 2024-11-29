import gradio as gr
from simpletransformers.question_answering import QuestionAnsweringModel
from transformers import pipeline
import json

# Load your trained QA model
model_type = "bert"
model_name = "model"  # Update this path to where your best model is stored
model = QuestionAnsweringModel(model_type, model_name, use_cuda=False)

# Load the expanded structured context JSON
with open("structured_context.json", "r") as file:
    context_sections = json.load(file)

# Initialize the zero-shot classification pipeline
classification_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

# Function to truncate context for better QA performance
def truncate_context(context, max_length=512):
    return context[:max_length]  # Simple truncation to first max_length characters

# Function to find the most relevant context dynamically
def find_relevant_context(question):
    topics = [section["topic"] for section in context_sections]

    # Perform classification
    classification_result = classification_pipeline(question, topics)

    # Select the best topic
    best_topic = classification_result["labels"][0]
    print(f"\n--- Selected Topic: {best_topic} ---")

    # Retrieve the context corresponding to the best topic
    for section in context_sections:
        if section["topic"] == best_topic:
            return truncate_context(section["context"])
    
    return ""  # Fallback if no topic is matched

def answer_question(question):
    # Find the relevant context
    relevant_context = find_relevant_context(question)
    if not relevant_context:
        return "I couldn't determine a relevant context for your question."
    
    # Log the selected context
    print("\n--- Debug Info: Selected Context ---")
    print(relevant_context)

    # Prepare the prediction input for your trained model
    to_predict = [
        {
            "context": relevant_context,
            "qas": [
                {
                    "id": "1",
                    "question": question,
                }
            ],
        }
    ]
    
    # Get predictions from your QA model
    answers, _ = model.predict(to_predict)

    # Log the model's answers
    print("\n--- Debug Info: Model Answers ---")
    print(answers)

    # Handle multiple answers and filter out empty strings
    filtered_answers = [ans for ans in answers[0]["answer"] if ans.strip()]
    
    # Return the first valid answer or fallback if no valid answer exists
    if filtered_answers:
        return filtered_answers[0]
    else:
        return "I couldn't find an answer to your question."

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
