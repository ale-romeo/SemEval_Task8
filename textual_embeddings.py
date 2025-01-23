from SemEval_Task8.utils import label_answer_type
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def textual_embeddings(all_qa):
    data = [
        {"question": qa['question'], "answer_type": label_answer_type(qa['answer'])}
        for qa in all_qa
    ]

    questions = [item['question'] for item in data]
    answer_types = [item['answer_type'] for item in data]

    # Convert answer types to numerical labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(answer_types)

    # Convert questions to TF-IDF vectors
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(questions)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='f1_macro', verbose=1)

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    joblib.dump(best_model, "SemEval_Task8/best_answer_type_model.pkl")
    joblib.dump(vectorizer, "SemEval_Task8/tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, "SemEval_Task8/label_encoder.pkl")
    print("Model, vectorizer, and label encoder saved successfully!")