import json
import os
from typing import Dict

import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from transformers_interpret import MultiLabelClassificationExplainer


class SequenceMultilabel:
    def __init__(self):
        self.list_label = []
        self.tokenizer = None
        self.model = None
        self.cls_explainer = None

    def fit(self, dataset: dict):
        train_texts = dataset["list_text"]
        dict_label = {}
        self.list_label = dataset["list_label"]
        for i, label in enumerate(dataset["list_label"]):
            dict_label[label] = i

        train_labels = []
        for text_label in dataset["list_text_label"]:
            train_labels.append(dict_label[text_label])
        print("loaded", flush=True)

        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True)

        class SmDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        train_dataset = SmDataset(train_encodings, train_labels)
        val_dataset = SmDataset(val_encodings, val_labels)

        training_args = TrainingArguments(
            output_dir="./results",  # output directory
            num_train_epochs=3,  # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir="./logs",  # directory for storing logs
            logging_steps=10,
        )

        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

        trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
        )

        trainer.train()

        self.cls_explainer = MultiLabelClassificationExplainer(self.model, self.tokenizer)

    def save(self, model_id: str):
        dict_header = {}
        dict_header["model_id"] = model_id
        dict_header["classifier_type"] = "sequence_multilabel"
        dict_header["list_label"] = self.list_label
        if not os.path.isdir(f"./{model_id}"):
            os.mkdir(model_id)
        json.dump(dict_header, open(f"./{model_id}/header.json", "w"))
        self.model.save_pretrained(f"./{model_id}")

    def load(self, model_id: str):
        dict_header = json.load(open(f"./{model_id}/header.json", "r"))
        self.list_label = dict_header["list_label"]
        self.model = DistilBertForSequenceClassification.from_pretrained(f"./{model_id}")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.cls_explainer = MultiLabelClassificationExplainer(self.model, self.tokenizer)

    def classify(self, text: str) -> dict:
        self.cls_explainer(text)
        dict_result = {}
        for i in range(len(self.list_label)):
            dict_result[self.list_label[i]] = self.cls_explainer.pred_probs_list[i].item()
        return dict_result

    def classify_explain(self, text: str) -> dict:
        attribution = self.cls_explainer(text)
        dict_result = {}
        for i, label in enumerate(self.list_label):
            dict_result[label] = {}
            dict_result[label]["label"] = label
            dict_result[label]["score"] = self.cls_explainer.pred_probs_list[i].item()
            dict_result[label]["support"] = attribution[f"LABEL_{i}"]
        return dict_result

    def classify_explain_html(self, text: str) -> dict:
        attribution = self.cls_explainer(text)
        self.cls_explainer.visualize("temp.html")
        html_string = open("temp.html", "r").read()
        dict_result = {}
        for i, label in enumerate(self.list_label):
            dict_result[label] = {}
            dict_result[label]["label"] = label
            dict_result[label]["score"] = self.cls_explainer.pred_probs_list[i].item()
            dict_result[label]["support"] = attribution[f"LABEL_{i}"]
            dict_result[label]["html"] = html_string
        return dict_result
        return dict_result
