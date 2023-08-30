import os
from queue import Queue

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from arin_text_classifier.classifier.tfid_multinominal_nb import \
    TfidMultinominalNb

if __name__ == "__main__":
    classifier = TfidMultinominalNb()
    classifier.fit("test_data")
    print(classifier.classify("I am a happy person"))