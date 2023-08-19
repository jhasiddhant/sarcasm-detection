import torch
import pandas as pd
import numpy as np
import evaluate
from sklearn.metrics import classification_report
from datasets import load_dataset, Features, Value, ClassLabel, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding, Trainer

MODEL_CHECKPOINT = "cardiffnlp/twitter-roberta-base-sentiment"

class_names = ["0", "1"]

def preprocess_dataset(filename):
    data_features = Features(
        {
            "text": Value('string'),
            "label": ClassLabel(names=class_names)
        }
    )
    new_dataset = load_dataset("csv", data_files=filename, features=data_features, split='train')
    new_dataset = new_dataset.map(lambda example: {"text": example["text"], "label": str(example["label"])})
    new_dataset = new_dataset.train_test_split(train_size=0.7)

    return new_dataset

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, return_tensors = 'pt')
    return tokenizer

def load_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')
    print(device)
    return device

id2label = {0: "0", 1: "1"}
label2id = {"0": 0, "1": 1}

def preprocess_function(examples):
    tokenizer = load_tokenizer()
    model_inputs = tokenizer(examples["text"], truncation=True, max_length = 128, padding ="max_length")
    return model_inputs

def tokenize_dataset(new_dataset):
    tokenized_dataset = new_dataset.map(preprocess_function, batched = True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label","labels")
    return tokenized_dataset

def load_data_collator():
    data_collator = DataCollatorWithPadding(load_tokenizer(), return_tensors ='pt')
    return data_collator

def set_training_args():
    args = TrainingArguments(
        output_dir = "Path to Store checkpoints of trained model",
        evaluation_strategy = "epoch",
        logging_strategy = "epoch",
        learning_rate = 2e-5,
        do_eval = True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay = 0.01,
        save_strategy = "epoch",
        load_best_model_at_end=True,
    )
    return args

def compute_metric(eval_preds):
    f1_metric = evaluate.load("f1")
    predictions,labels = eval_preds
    predictions = np.argmax(predictions,axis=-1)
    f1_score = f1_metric.compute(predictions=predictions, references=labels, average = "micro")
    return f1_score

def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=2, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    model = model.to(load_device())
    return model

def set_trainer(tokenized_dataset):
    trainer = Trainer(
        load_model(),
        set_training_args(),
        train_dataset = tokenized_dataset['train'],
        eval_dataset = tokenized_dataset['test'],
        tokenizer=load_tokenizer(),
        compute_metrics=compute_metric,
    )
    return trainer

new_dataset = preprocess_dataset("Path to Preprocessed train data")
tokenized_data = tokenize_dataset(new_dataset)
print(tokenized_data['train'][:2])
trainer = set_trainer(tokenized_data)
trainer.train()
predictions = trainer.predict(tokenized_data['test'])
f1_metric = evaluate.load("f1")
preds = np.argmax(predictions.predictions,axis=-1)
results = f1_metric.compute(predictions=preds, references=predictions.label_ids, average = "micro")
result_report = classification_report(preds, predictions.label_ids, target_names = class_names )
print(result_report)


TRAINED_MODEL_CHECKPOINT = "Path of last checkpoint of trained model"

def load_trained_model():
    test_model = AutoModelForSequenceClassification.from_pretrained(TRAINED_MODEL_CHECKPOINT, local_files_only = True, num_labels=2, id2label=id2label, label2id=label2id)
    test_model = test_model.to(load_device())
    return test_model

def load_test_tokenizer():
    test_tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_CHECKPOINT, return_tensors = 'pt')
    return test_tokenizer

def preprocess_test_dataset(filename):
    data_features = Features( {
            "text": Value('string'),
            "label":ClassLabel(names=class_names)
        }
    )
    test_dataset = load_dataset("csv", data_files = filename, features = data_features, split = 'train')
    test_dataset = test_dataset.map(lambda example: {"text": example["text"], "label": str(example["label"])})
    print(test_dataset)

    return test_dataset

def preprocess_test_function(examples):
    test_tokenizer = load_test_tokenizer()
    model_test_inputs = test_tokenizer(examples["text"], truncation=True, max_length = 128, padding ="max_length")
    return model_test_inputs

def tokenize_test_dataset(test_dataset):
    tokenized_test_dataset = test_dataset.map(preprocess_test_function, batched = True)
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(["text"])
    tokenized_test_dataset = tokenized_test_dataset.rename_column("label","labels")

    return tokenized_test_dataset

def set_test_args():
    test_args = TrainingArguments(
        output_dir = "Path to Store checkpoints of test",

        do_train = False,
        do_predict = True,
        do_eval = False,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

    )
    return test_args

def set_test_trainer(tokenized_test_dataset):
    test_trainer = Trainer(
        load_trained_model(),
        set_test_args(),

        eval_dataset = tokenized_test_dataset,
        # data_collator=load_data_collator(),
        tokenizer=load_test_tokenizer(),
        compute_metrics=compute_metric,
    )
    return test_trainer

test_dataset = preprocess_test_dataset("Path to Preprocessed test data")
tokenized_test_data = tokenize_test_dataset(test_dataset)
test_trainer = set_test_trainer(tokenized_test_data)
predictions = test_trainer.predict(tokenized_test_data)
f1_metric = evaluate.load("f1")
preds = np.argmax(predictions.predictions,axis=-1)
results = f1_metric.compute(predictions=preds, references=predictions.label_ids, average = "micro")

result_report = classification_report(preds,predictions.label_ids, target_names = class_names )
print(result_report)

