import json
import os
import random


def load_test_data():
    list_name_file = os.listdir("test_data")
    dataset = {}
    dataset["list_label"] = ["spam", "eggs"]
    dataset["list_text"] = []
    dataset["list_text_label"] = []
    for name_file in list_name_file:
        with open("test_data/" + name_file, "r") as file:
            instance = json.load(file)
            dataset["list_text"].append(instance["text"])
            dataset["list_text_label"].append(random.choice(dataset["list_label"]))
    return dataset