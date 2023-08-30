import json
import os
import uuid
from typing import List

from fastapi import FastAPI, HTTPException

app = FastAPI()

dict_classifier = {}
dict_dataset = {}

from arin_text_classifier.classifier.tfid_multinominal_nb import \
    TfidMultinominalNb


@app.get("/list_classifier_types/")
async def list_classifier_types():
    return {"list_classifier_types": ["tfid_multinominal_nb"]}

@app.get("/list_classifiers/")
async def list_classifiers(job_id: str):
    dict_list_classifier = {}
    dict_list_classifier["list_classifier"] = []
    for classfier_id, classifier in dict_classifier.items():
        dict_list_classifier["list_classifier"].append(
            {
                "classfier_id": classfier_id,
                "dict_tag": classifier["dict_tag"],
                "list_label": classifier["list_label"]
            })
    return dict_list_classifier

@app.post("/create_classifier/")
async def create_classifier(classifier_type: str, dict_tag:dict, dataset_id: str):

    if classifier_type not in ["tfid_multinominal_nb"]:
        raise HTTPException(status_code=404, detail=f"Classifier type {classifier_type} not found")
    if dataset_id not in dict_dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with id {dataset_id} not found")
    classifier_id = str(uuid.uuid4())
    classifier = {}
    classifier["classifier_id"] = classifier_id
    classifier["dict_tag"] = dict_tag
    classifier["list_label"] = dict_dataset[dataset_id]["list_label"]
    if classifier_type == "tfid_multinominal_nb":
        classifier["model"] = TfidMultinominalNb()
        classifier["model"].fit(dict_dataset[dataset_id])
    return {"message": "Classfier created"}


@app.get("/create_dataset/")
async def create_dataset(dict_tag:dict, list_label: str):
    dataset_id = str(uuid.uuid4())
    dataset = {}
    dataset["dataset_id"] = dataset_id
    dataset["dict_tag"] = dict_tag
    dataset["list_label"] = list_label
    dataset["list_text"] = []
    dict_dataset[dataset_id] = dataset
    return dataset_id

@app.get("/add_text/")
async def add_text(dataset_id: str, text:str, label:str):
    if dataset_id not in dict_dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with id {dataset_id} not found")
    dataset = dict_dataset[dataset_id]
    if label not in dataset["list_label"]:
        raise HTTPException(status_code=404, detail=f"Label {label} not found")
    dataset.list_text.append({"text": text, "label": label})
    return {"message": "Text added"}

@app.get("/list_dataset_id/")
async def get_transcript(job_id: str):
    dict_list_dataset = {}
    dict_list_dataset["list_dataset"] = []
    for dataset_id, dataset in dict_dataset.items():
        dict_list_dataset["list_dataset"].append(
            {
                "dataset_id": dataset_id,
                "dict_tag": dataset["dict_tag"],
                "list_label": dataset["list_label"]
            })
    return dict_list_dataset

@app.get("/predict_label/")
def predict_label(classifier_id: str, text: str):
    if classifier_id not in dict_classifier:
        raise HTTPException(status_code=404, detail=f"Classfier with id {classifier_id} not found")
    classifier = dict_classifier[classifier_id]
    dict_result = classifier["model"].classify(text)
    return dict_result    dict_result = classifier["model"].classify(text)
    return dict_result