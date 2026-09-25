import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faq_questions = [
    "How can I reset my password?",
    "Where do I change my account password?",
    "How can I track my delivery?",
    "What is your refund policy?"
]

faq_vectorizer = TfidfVectorizer()

faq_matrix = faq_vectorizer.fit_transform(faq_questions)

query = ["I need to change my password"]

scores = cosine_similarity(
    faq_vectorizer.transform(query),
    faq_matrix
).flatten()

result = pd.DataFrame({
    "question": faq_questions,
    "similarity": scores
}).sort_values(
    "similarity",
    ascending=False
)

print(result)
