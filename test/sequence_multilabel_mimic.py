import json
import os
from typing import List

from tools_test import load_test_data

from arin_text_classifier.classifier.sequence_multilabel import SequenceMultilabel


def get_state():
    return "New York"


def get_ssn():
    return "123-45-6789"


def get_country():
    return "USA"


def get_year():
    return "2021"


def get_name():
    return "Mark Antony"


def get_date():
    return "2021-07-21"


def get_month():
    return "July"


def get_telephone():
    return "+1 123-456-7890"


def get_hospital_name():
    return "St. Mary's Hospital"


def get_location():
    return "New York City"


def sanitize_mimic(text: str, block_size=None) -> List[str]:
    with open("mimic_sanitize.txt", "w") as f:
        f.write(text)
    dict_lookup = {}

    text_sanitized = ""
    # find alls tags [**
    index_start = -1
    index_end = -1
    while True:
        index_start = text.find("[**", index_end + 1)
        # add text until mask
        if index_start == -1:
            text_sanitized += text[index_end + 3 :]
            break
        else:
            text_sanitized += text[index_end + 3 : index_start]
        index_end = text.find("**]", index_start)
        # add mask text
        text_mask = text[index_start + 3 : index_end]
        if "Last Name" in text_mask:
            text_sanitized += get_name()
        elif "name" in text_mask:
            text_sanitized += get_name()
        elif "Name" in text_mask:
            text_sanitized += get_name()
        elif "Hospital" in text_mask:
            text_sanitized += get_hospital_name()
        elif "Telephone" in text_mask:
            text_sanitized += get_telephone()
        elif text_mask.count("-") == 2:
            text_sanitized += get_date()
        elif text_mask.count("-") == 1:
            text_sanitized += get_date()
        elif "Date range" in text_mask:
            text_sanitized += get_date() + " " + get_date()
        elif text_mask.isalnum() and len(text_mask) == 4:
            text_sanitized += get_year()
        elif text_mask.isalnum() and len(text_mask) == 2:
            text_sanitized += get_year()
        elif "Month (only)" in text_mask:
            text_sanitized += get_month()
        elif "Location" in text_mask:
            text_sanitized += get_location()
        elif "MD Number" in text_mask:
            text_sanitized += get_telephone()
        elif "Country" in text_mask:
            text_sanitized += get_country()
        elif "Numeric Identifier" in text_mask:
            text_sanitized += get_ssn()
        elif "Job Number" in text_mask:
            text_sanitized += get_telephone()
        elif "State" in text_mask:
            text_sanitized += get_state()
        elif "Month/Day" in text_mask:
            text_sanitized += get_date()
        else:
            print(text_mask, flush=True)
            print(index_start, index_end, flush=True)

    # remove the extra newlines and split in sets of x lines
    list_line = text_sanitized.split("\n")
    list_block = []
    block = ""
    for line in list_line:
        if line.strip() != "":
            if block_size is None:
                block += line.strip() + "\n"
            else:
                if block_size < len(block) + len(line):
                    list_block.append(block)
                    block = ""
                block += line.strip() + "\n"
    if block != "":
        list_block.append(block)
    return list_block


def sanitize_html(text: str) -> str:

    exit()


def load_mimic() -> dict:

    path_dir_dataset_cache = os.environ["PATH_DIR_DATASET_CACHE"]
    path_file_dataset_train = os.path.join(
        path_dir_dataset_cache,
        "dataset-diabetes-10000-4000-20230721-dev",
        "dataset-diabetes-10000-4000-20230721-train-dev.json",
    )
    dataset = {}
    dataset["list_label"] = []
    dataset["list_text"] = []
    dataset["list_text_label"] = []
    dataset["list_instance_id"] = []
    dataset["list_list_block"] = []
    with open(path_file_dataset_train, "r") as f:
        json_data = json.load(f)
        dataset["list_label"] = json_data["list_label"]
        print(len(json_data["list_instance"]))
        for instance in json_data["list_instance"]:
            dataset["list_instance_id"].append(instance["instance_id"])
            list_block = []
            for data_source in instance["list_data_source"]:

                if "Discharge summary" in data_source["list_tag"]:
                    list_block_sanitized = sanitize_mimic(data_source["text"], block_size=1500)
                    for block in list_block_sanitized:
                        dataset["list_text"].append(block)
                        list_block.append(block)
                        dataset["list_text_label"].append(instance["list_label"][0])
                    break
                print("miss")
            dataset["list_list_block"].append(list_block)

    return dataset


if __name__ == "__main__":
    dataset = load_mimic()
    print(len(dataset["list_instance_id"]))
    print(dataset["list_label"], flush=True)
    classifier = SequenceMultilabel()
    if os.path.exists("test_model_mimic"):
        classifier.load("test_model_mimic")
        for instance_id, list_block in zip(dataset["list_instance_id"], dataset["list_list_block"]):
            for i, block in enumerate(list_block):
                path_file = f"./test_model_mimic_out/{instance_id}_{i}.json"
                if os.path.exists(path_file):
                    print("skip")
                    continue
                try:
                    dict_result = classifier.classify_explain(block)

                    json.dump(dict_result, open(path_file, "w"))
                except Exception as e:
                    dict_result = {"error": str(e)}
                    json.dump(dict_result, open(path_file, "w"))
                    continue
    else:
        classifier.fit(dataset)
        classifier.save("test_model_mimic")
