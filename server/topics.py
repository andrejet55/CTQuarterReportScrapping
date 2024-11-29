from transformers import pipeline

# Initialize zero-shot classification pipeline
classification_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Example question
question = "What is the annual dividend for stakeholders?"

# Full list of topics
topics = [
    "Revenue",
    "Earnings",
    "Dividends",
    "Shares",
    "Customers",
    "Financial",
    "Investments",
    "Capital",
    "Conference",
    "Business",
    "Sales",
    "Income",
    "Stakeholders",
    "Retail",
    "Canadians",
    "Information",
]

# Perform classification
classification_result = classification_pipeline(question, topics)

# Print the results
print("\n--- Classification Results ---")
for label, score in zip(classification_result["labels"], classification_result["scores"]):
    print(f"Topic: {label}, Score: {score}")
