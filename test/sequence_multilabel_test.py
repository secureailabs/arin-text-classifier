import os

from tools_test import load_test_data

from arin_text_classifier.classifier.sequence_multilabel import \
    SequenceMultilabel

if __name__ == "__main__":
    dataset = load_test_data()
    classifier = SequenceMultilabel()
    if os.path.exists("test_model_ms"):
        classifier.load("test_model_ms")
    else:
        classifier.fit(dataset)
        classifier.save("test_model_ms")
    print(classifier.classify("I am a happy person"))
    print(classifier.classify("You are great"))

    print(classifier.classify_explain("I am a happy person"))
    print(classifier.classify_explain("You are great"))

    print(classifier.classify_explain_html("I am a happy person"))
    print(classifier.classify_explain_html("You are great"))
    print(classifier.classify_explain_html("You are great"))
