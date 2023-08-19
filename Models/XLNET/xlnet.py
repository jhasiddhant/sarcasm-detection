import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import XLNetTokenizer, XLNetForSequenceClassification, TrainingArguments, Trainer

train_data = pd.read_csv('Path to preprocesses training data')
test_data = pd.read_csv('Path to preprocesses testing data')

class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SarcasmTestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item
    def __len__(self):
        return len(self.encodings)

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(labels, pred)

    return {"accuracy": accuracy,"f1_score":f1}


X_train = train_data['text']
y_train = train_data['label']
X_test = test_data['text']
y_test = test_data['label']

X_train = X_train.tolist()
X_test = X_test.tolist()
y_train = y_train.tolist()
y_test = y_test.tolist()

model_name = 'detecting-sarcasm'
task='sentiment'
MODEL = 'xlnet-base-cased'

tokenizer = XLNetTokenizer.from_pretrained(MODEL,num_labels=2, loss_function_params={"weight": [0.75, 0.25]})

train_encodings = tokenizer(X_train, truncation=True, padding=True,return_tensors = 'pt')
test_encodings = tokenizer(X_test,truncation=True, padding=True,return_tensors = 'pt')

train_dataset = SarcasmDataset(train_encodings, y_train)
test_dataset = SarcasmDataset(test_encodings, y_test)

training_args = TrainingArguments(
    output_dir='./res', num_train_epochs=5, per_device_train_batch_size=32, warmup_steps=500, weight_decay=0.01,logging_dir='./logs4'
)

model = XLNetForSequenceClassification.from_pretrained(MODEL)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    compute_metrics = compute_metrics,
)
trainer.train()

preds = trainer.predict(test_dataset)
preds = np.argmax(preds.predictions[:, 0:2], axis=-1)

report = classification_report(y_test, preds)
print(report)
