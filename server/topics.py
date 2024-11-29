from transformers import pipeline

# Initialize zero-shot classification pipeline
classification_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Example question
question = "What was the revenue growth in Q3 2024?"

# Full list of topics
topics = [
    "conference",
    "business",
    "sales",
    "quarterly revenue growth",
    "retail",
    "financial performance",
    "capital",
    "dividend",
    "repurchase",
    "forward looking",
    "earnings",
    "dividends",
    "share repurchases",
    "customer engagement",
    "financial services",
    "ct reit",
    "capital allocation",
    "canadian customers motivations",
    "company information"
]

# Perform classification
classification_result = classification_pipeline(question, topics)

# Print the results
print("\n--- Classification Results ---")
for label, score in zip(classification_result["labels"], classification_result["scores"]):
    print(f"Topic: {label}, Score: {score}")
