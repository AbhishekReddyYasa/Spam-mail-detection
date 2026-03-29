"""
create_model.py
---------------
Trains the initial spam-detection model and saves:
  - model.pkl
  - vectorizer.pkl
  - feedback_data.csv  (empty seed file for the self-learning loop)

Run once before launching the Streamlit app.
"""

import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# 1. Training data  (expanded from the original 6-sample set)
# ---------------------------------------------------------------------------
spam_samples = [
    "Win money now click here",
    "Congratulations you have won a prize",
    "Claim your free reward today",
    "You are selected for a cash prize",
    "Exclusive deal just for you act now",
    "Free entry in our weekly competition",
    "Get rich quick guaranteed",
    "Earn money fast from home",
    "Your account has been compromised click to verify",
    "Urgent your PayPal account is suspended",
    "Click here to claim your lottery prize",
    "You have been pre-approved for a loan",
    "Limited time offer buy now",
    "Dear winner your prize is waiting",
    "Send us your bank details to receive funds",
    "Hot singles in your area",
    "Make $5000 a week from home",
    "Buy cheap medications online",
    "Increase your revenue with our system",
    "FREE! Call now to claim your voucher",
    "WINNER!! As a valued network customer",
    "Had your mobile 11 months or more? Update for free",
    "SIX chances to win CASH",
    "URGENT! Your mobile No was awarded a prize",
    "You have won a Nokia 3310",
    "Cash prize of 5000 GBP awaits",
    "Double your income overnight",
    "Apply now for payday loan",
    "Your subscription is expiring renew now",
    "Discount on Viagra order online",
]

ham_samples = [
    "Hello how are you doing today",
    "Let us meet tomorrow at the cafe",
    "Are we still on for the project meeting",
    "Can you send me the report by Friday",
    "Happy birthday hope you have a great day",
    "The weather is nice today lets go for a walk",
    "Could you please review the attached document",
    "I will call you back in an hour",
    "Thanks for your help with the presentation",
    "Please find the invoice attached",
    "Good morning hope you slept well",
    "Team lunch is scheduled for noon",
    "I have forwarded your email to the manager",
    "Just checking in to see how things are going",
    "See you at the conference next week",
    "The new library opened downtown",
    "Reminder your dentist appointment is tomorrow",
    "Can we reschedule our call to 3 PM",
    "Loved the book you recommended",
    "Your order has been shipped tracking number below",
    "Mum I will be home late tonight",
    "What time does the match start",
    "Do you want to grab coffee this afternoon",
    "I need help understanding the assignment",
    "Please let me know if you need anything else",
    "Happy new year wishing you all the best",
    "The meeting notes are attached for your reference",
    "My flight lands at 7 PM pick me up",
    "We are having a barbecue on Sunday",
    "Great work on the presentation today",
]

texts  = spam_samples + ham_samples
labels = [1] * len(spam_samples) + [0] * len(ham_samples)

df = pd.DataFrame({"text": texts, "label": labels})

# ---------------------------------------------------------------------------
# 2. Train / test split for evaluation
# ---------------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# ---------------------------------------------------------------------------
# 3. Vectorise
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams + bigrams for better context
    min_df=1,
    stop_words="english",
)
X_train = vectorizer.fit_transform(X_train_raw)
X_test  = vectorizer.transform(X_test_raw)

# ---------------------------------------------------------------------------
# 4. Train – use MultinomialNB because it supports partial_fit (self-learning)
# ---------------------------------------------------------------------------
model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)
print("=== Initial Model Evaluation ===")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# ---------------------------------------------------------------------------
# 6. Persist artifacts
# ---------------------------------------------------------------------------
pickle.dump(model,      open("model.pkl",      "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Seed an empty feedback CSV so the app doesn't error on first run
feedback_path = "feedback_data.csv"
if not os.path.exists(feedback_path):
    pd.DataFrame(columns=["text", "label"]).to_csv(feedback_path, index=False)

print("\nSaved: model.pkl | vectorizer.pkl | feedback_data.csv")
print("Run `streamlit run app.py` to start the application.")
