from typing import Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class TfidMultinominalNb:

    def __init__(self):
        self.encoder = TfidfVectorizer(max_features=1000)  # You can adjust the number of features
        self.classifier = MultinomialNB()


    def fit(self, dataset:dict):
        X = dataset["list_text"]
        Y = dataset["list_label"]
        X_transformed = self.encoder.fit_transform(X)
        self.classifier.fit(X_transformed, Y)

    def classify(self, text:str) -> Dict[str, float]:
        encoding = self.encoder.transform([text])
        y_pred = self.classifier.predict_proba(encoding)
        dict_result = {}
        for i, label in enumerate(self.classifier.classes_):
            dict_result[label] = y_pred[0][i]
        return dict_result