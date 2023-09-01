import json
import os
import uuid
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

dict_classifier = {}
dict_dataset = {}

from arin_text_classifier.classifier.tfid_multinominal_nb import \
    TfidMultinominalNb


class DatasetTemplate(BaseModel):
    dict_tag: Dict[str, str]
    list_label: List[str]


class InstanceTemplate(BaseModel):
    dataset_id: str
    text: str
    text_label: str

class PredictionRequest(BaseModel):
    classifier_id: str
    text: str

class InstanceBatchTemplate(BaseModel):
    list_instance: List[InstanceTemplate]


class ClassifierTemplate(BaseModel):
    classifier_type: str
    dict_tag: Dict[str, str]
    dataset_id: str

@app.get("/list_dataset/")
async def list_dataset():
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

@app.post("/create_dataset/")
async def create_dataset(dataset_template:DatasetTemplate):
    dataset_id = str(uuid.uuid4())
    dataset = {}
    dataset["dataset_id"] = dataset_id
    dataset["dict_tag"] = dataset_template.dict_tag
    dataset["list_label"] = dataset_template.list_label
    dataset["list_instance_id"] = []
    dataset["list_text"] = []
    dataset["list_text_label"] = []
    dict_dataset[dataset_id] = dataset
    return {"dataset_id":dataset_id}

@app.post("/add_instance/")
async def add_instance(instance_template: InstanceTemplate):
    if instance_template.dataset_id not in dict_dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with id {instance_template.dataset_id} not found")
    dataset = dict_dataset[instance_template.dataset_id]
    if instance_template.text_label not in dataset["list_label"]:
        raise HTTPException(status_code=404, detail=f"Label {instance_template.text_label} not found")
    instance_id = str(uuid.uuid4())
    dataset["list_instance_id"].append(instance_id)
    dataset["list_text"].append(instance_template.text)
    dataset["list_text_label"].append(instance_template.text_label)
    return {"instance_id": instance_id}


@app.post("/add_instance_batch/")
async def add_instance_batch(instance_batch_template: InstanceBatchTemplate):
    for instance_template in instance_batch_template.list_instance:
        if instance_template.dataset_id not in dict_dataset:
            raise HTTPException(status_code=404, detail=f"Dataset with id {instance_template.dataset_id} not found")
        dataset = dict_dataset[instance_template.dataset_id]
        if instance_template.text_label not in dataset["list_label"]:
            raise HTTPException(status_code=404, detail=f"Label {instance_template.text_label} not found")
    list_instance_id = []
    for instance_template in instance_batch_template.list_instance:
        dataset = dict_dataset[instance_template.dataset_id]
        instance_id = str(uuid.uuid4())
        dataset["list_instance_id"].append(instance_id)
        dataset["list_text"].append(instance_template.text)
        dataset["list_text_label"].append(instance_template.text_label)
        list_instance_id.append(instance_id)
    return {"list_instance_id": list_instance_id}

@app.get("/list_classifier_types/")
async def list_classifier_types():
    return {"list_classifier_types": ["tfid_multinominal_nb"]}

@app.get("/list_classifiers/")
async def list_classifiers():
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
async def create_classifier(classifier_template: ClassifierTemplate):

    if classifier_template.classifier_type not in ["tfid_multinominal_nb"]:
        raise HTTPException(status_code=404, detail=f"Classifier type {classifier_template.classifier_type} not found")
    if classifier_template.dataset_id not in dict_dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with id {classifier_template.dataset_id} not found")
    classifier_id = str(uuid.uuid4())
    classifier = {}
    classifier["classifier_id"] = classifier_id
    classifier["dict_tag"] = classifier_template.dict_tag
    classifier["list_label"] = dict_dataset[classifier_template.dataset_id]["list_label"]
    if classifier_template.classifier_type == "tfid_multinominal_nb":
        classifier["model"] = TfidMultinominalNb()
        classifier["model"].fit(dict_dataset[classifier_template.dataset_id])
    dict_classifier[classifier_id] = classifier
    return {"classifier_id": classifier_id}


@app.post("/predict/")
def predict(prediction_request: PredictionRequest):
    if prediction_request.classifier_id not in dict_classifier:
        raise HTTPException(status_code=404, detail=f"Classfier with id {prediction_request.classifier_id} not found")
    classifier = dict_classifier[prediction_request.classifier_id]
    dict_result = classifier["model"].classify(prediction_request.text)
    return dict_result
