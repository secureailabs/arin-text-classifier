from tools_test import load_test_data

from arin_text_classifier.classifier.tfid_multinominal_nb import TfidMultinominalNb

if __name__ == "__main__":
    dataset = load_test_data()
    classifier = TfidMultinominalNb()
    classifier.fit(dataset)
    print(classifier.classify("I am a happy person"))
    print(classifier.classify("You are great"))
