import gradio as gr
from simpletransformers.question_answering import QuestionAnsweringModel

# Load your trained model
model_type = "bert"
model_name = "model"  # Update this path to where your best model is stored
model = QuestionAnsweringModel(model_type, model_name, use_cuda=False)

# Load static context from the text file
with open("fixed_context.txt", "r") as file:
    static_context = file.read()

# Function to generate answers
def answer_question(question):
    to_predict = [
        {
            "context": static_context,
            "qas": [
                {
                    "id": "1",
                    "question": question,
                }
            ],
        }
    ]
    answers, _ = model.predict(to_predict)
    answer = answers[0]['answer'][0] if answers and 'answer' in answers[0] else "I couldn't find an answer to your question."
    return answer

# Create Gradio Interface
interface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text",
    title="Canadian Tire Corporation Q3 2024 Chatbot",
    description="Ask questions about the Canadian Tire Corporation Q3 2024 report. The chatbot will provide answers based on the provided static context.",
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)
