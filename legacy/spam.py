import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

data = pd.read_csv('spam.csv', encoding='latin-1')
print(data)
data = data[['v1', 'v2']]

data.columns = ['label', 'message']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# train-test-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# naive bayes
from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)
y_pred_nb = nb_classifier.predict(X_test_tfidf)

# Logistic Regression 
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train_tfidf, y_train)
y_pred_lr = lr_classifier.predict(X_test_tfidf)

#SVM
from sklearn.svm import SVC
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)
y_pred_svm = svm_classifier.predict(X_test_tfidf)

# Evaluating
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
def evaluate(y_test, y_pred, model_name):
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n")

# Evaluate the models
evaluate(y_test, y_pred_nb, "Naive Bayes")
evaluate(y_test, y_pred_lr, "Logistic Regression")
evaluate(y_test, y_pred_svm, "Support Vector Machine")





def predict_sms(message, vectorizer, nb_classifier, lr_classifier, svm_classifier):
   
    message_tfidf = vectorizer.transform([message]) # Transform the message using the trained TF-IDF vectorizer
    nb = nb_classifier.predict(message_tfidf)[0] #Naive Bayes
    lr = lr_classifier.predict(message_tfidf)[0] #Logistic Regression
    svm = svm_classifier.predict(message_tfidf)[0]#SVM
    
    # Predictions
    print("Naive Bayes : ", "Spam" if nb == 1 else "No Spam")
    print("Logistic Regression : ", "Spam" if lr == 1 else "No Spam")
    print("SVM : ", "Spam" if svm == 1 else "No spam")


message = input('Enter the message:')
predict_sms(message,
            vectorizer, 
            nb_classifier, 
            lr_classifier, 
            svm_classifier)
