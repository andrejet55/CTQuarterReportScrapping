import json
from simpletransformers.question_answering import QuestionAnsweringModel

def ask_question_from_file(model):
    """
    Ask a question to the trained QA model with user input for context and question.

    Args:
        model (QuestionAnsweringModel): The trained QA model.

    Returns:
        None: Prints the question and the answer to the terminal.
    """
    context = input("Enter the context (paragraph containing the answer):\n")
    question = input("\nEnter your question:\n")
    
    # Prepare the prediction input
    to_predict = [
        {
            "context": context,
            "qas": [
                {
                    "id": "1",  # Dummy ID
                    "question": question
                }
            ]
        }
    ]
    
    try:
        # Debugging information: Print input to the model
        print("\n--- Debug Info ---")
        print(f"Prediction Input: {to_predict}")
        
        # Get predictions
        answers, _ = model.predict(to_predict)
        
        # Debugging information: Print raw answers
        print(f"Raw Answers: {answers}")
        
        # Extract and display the answer
        answer = answers[0]['answer'][0] if answers and 'answer' in answers[0] else "No answer found"
    except ValueError as e:
        print(f"Error encountered during prediction: {e}")
        answer = "An error occurred while processing the question. Please try again."
    except Exception as e:
        print(f"Unexpected error: {e}")
        answer = "An unexpected error occurred. Please try again."
    
    print("\n--- Model's Answer ---")
    print(f"Question: {question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    # Load the trained model
    model_trained_type = "bert"
    model_trained_name = "model"  # Adjust to the path where your model is stored
    model_trained = QuestionAnsweringModel(
        model_type=model_trained_type, 
        model_name=model_trained_name, 
        use_cuda=False  # Disable GPU for prediction
    )

    print("Question Answering Script")
    print("--------------------------")
    ask_question_from_file(model_trained)
