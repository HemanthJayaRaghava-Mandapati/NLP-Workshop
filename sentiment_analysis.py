import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

reviews = [
    ("The phone is excellent and very fast", "positive"),
    ("Amazing camera and beautiful display", "positive"),
    ("Battery life is great", "positive"),
    ("I love this affordable phone", "positive"),
    ("The product works perfectly", "positive"),
    ("Excellent quality and quick delivery", "positive"),
    ("Very happy with the purchase", "positive"),
    ("The phone feels premium and reliable", "positive"),
    ("Fantastic performance for the price", "positive"),
    ("The display is bright and sharp", "positive"),
    ("Customer service was helpful", "positive"),
    ("This is a wonderful device", "positive"),

    ("The phone is slow and frustrating", "negative"),
    ("Terrible battery life", "negative"),
    ("The camera quality is poor", "negative"),
    ("I regret buying this phone", "negative"),
    ("The product stopped working", "negative"),
    ("Bad quality and delayed delivery", "negative"),
    ("Very disappointed with this purchase", "negative"),
    ("The device is unreliable", "negative"),
    ("Awful performance for the price", "negative"),
    ("The display is dull and damaged", "negative"),
    ("Customer service was rude", "negative"),
    ("This is a terrible device", "negative")
]

data = pd.DataFrame(reviews, columns=["text", "sentiment"])

print(data["sentiment"].value_counts())

X = data["text"]
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Train:", len(X_train), "| Test:", len(X_test))

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results = pd.DataFrame({
    "text": X_test.values,
    "actual": y_test.values,
    "predicted": y_pred
})

print(results)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=["negative", "positive"],
    cmap="Grays",
    values_format="d"
)

plt.title("Sentiment Classifier Confusion Matrix")
plt.tight_layout()
plt.show()

new_reviews = [
    "The camera is excellent and the battery is great",
    "The phone is slow and the display is terrible",
    "The phone is affordable"
]

for text, label in zip(new_reviews, model.predict(new_reviews)):
    print(f"{label.upper():8} | {text}")

probs = model.predict_proba(new_reviews)

print(
    pd.DataFrame(
        probs,
        columns=model.named_steps["classifier"].classes_
    ).assign(text=new_reviews)
)
