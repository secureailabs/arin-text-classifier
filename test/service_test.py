import requests
from tools_test import load_test_data

base_url = "http://localhost:8000/"
dataset = load_test_data()
dataset_template = {"dict_tag": {"dataset_name": "test"}, "list_label": dataset["list_label"]}
response = requests.post(base_url + "create_dataset/", json=dataset_template).json()
print(response)
dataset_id = response["dataset_id"]
response = requests.get(base_url + "list_dataset/").json()
print(response)
instance_batch_template = {}
instance_batch_template["list_instance"] = []
for text, label in zip(dataset["list_text"], dataset["list_text_label"]):
    instance_template = {"dataset_id":dataset_id, "text": text, "text_label": label}
    instance_batch_template["list_instance"].append(instance_template)

response = requests.post(base_url + "add_instance_batch/", json=instance_batch_template).json()
print(response)
classifier_template = {"dataset_id": dataset_id, "dict_tag":{}, "classifier_type": "tfid_multinominal_nb"}
response = requests.post(base_url + "create_classifier/", json=classifier_template).json()
print(response)
classifier_id = response["classifier_id"]
response = requests.get(base_url + "list_classifiers/").json()
print(response)
prediction_request = {"classifier_id": classifier_id, "text": "I love you"}
response = requests.post(base_url + "predict/", json=prediction_request).json()
print(response)
