import torch
import numpy as np
import pandas as pd
from transformers import GPT2Model, GPT2Tokenizer
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

train_df = pd.read_csv('Path to preprocessed training data')
test_df = pd.read_csv('Path to preprocessed test data')

# Instantiate the GPT2Tokenizer from the 'gpt2' pre-trained model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# This means that padding tokens will be added to the left side of the sequences. GPT2 uses the last token for prediction so we need to pad to the left.
tokenizer.padding_side = "left"

# Set the pad_token to the end-of-sequence (eos) token
tokenizer.pad_token = tokenizer.eos_token

labels = {0: 0, 1: 1}

class trainDataset(torch.utils.data.Dataset):
    # Initialize the trainDataset class with the dataframe 'df'
    def __init__(self, df):

        # Extract the labels from the 'label' column of the dataframe and map them to their corresponding values
        self.labels = [labels[label] for label in train_df['label']]

        # Tokenize and preprocess each text in the 'text' column of the dataframe
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=128,
                                truncation=True,
                                return_tensors="pt") for text in train_df['text']]

    def classes(self):
        # Return the labels
        return self.labels

    def __len__(self):
        # Return the length of the dataset (number of samples)
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Get a batch of labels given the index 'idx'
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Get a batch of input texts given the index 'idx'
        return self.texts[idx]

    # Get a specific sample from the dataset given the index 'idx'
    def __getitem__(self, idx):

        # Get the batch of input texts for the sample
        batch_texts = self.get_batch_texts(idx)

        # Get the batch of labels for the sample
        batch_y = self.get_batch_labels(idx)

        # Return the batch of input texts and labels
        return batch_texts, batch_y

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Initialize the SimpleGPT2SequenceClassifier class
class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, max_seq_len: int, gpt_model_name: str):
        super(SimpleGPT2SequenceClassifier, self).__init__()

        # Load the GPT2 model from the specified pretrained model name
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)

        # Define a fully connected layer (linear layer) to map the hidden representation to the number of classes
        self.fc1 = nn.Linear(hidden_size * max_seq_len, num_classes)

    def forward(self, input_id, mask):
        """
        Forward pass of the SimpleGPT2SequenceClassifier

        Args:
            input_id: Encoded input IDs of the sentences
            mask: Attention mask indicating the valid tokens in the input sequence
        """

        # Pass the input IDs and attention mask through the GPT2 model
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)

        # Get the batch size from the output shape of the GPT2 model
        batch_size = gpt_out.shape[0]

        # Reshape the GPT2 output to match the input shape of the fully connected layer
        linear_output = self.fc1(gpt_out.view(batch_size, -1))

        # Return the linear output
        return linear_output


"""## Training"""
def train(model, train_data, val_data, learning_rate, epochs):
    """
    Train function for the SimpleGPT2SequenceClassifier model.

    Args:
        model: Instance of the SimpleGPT2SequenceClassifier model
        train_data: Training data
        val_data: Validation data
        learning_rate: Learning rate for the optimizer
        epochs: Number of training epochs
    """
    # Create train and validation datasets
    train_dataset = trainDataset(train_data)
    val_dataset = trainDataset(val_data)

    # Create train and validation data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2)

    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Move the model and loss function to GPU if available
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        all_predictions_train = []
        all_labels_train = []

        # Iterate over the training data batches
        for train_input, train_label in tqdm(train_dataloader):
            # Move input and label tensors to the device (GPU if available)
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)

            # Zero the gradients
            model.zero_grad()

            # Perform forward pass
            output = model(input_id, mask)

            # Calculate the batch loss
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            # Calculate the batch accuracy
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            # Collect predictions and labels for F1 score calculation
            all_predictions_train.extend(output.argmax(dim=1).cpu().tolist())
            all_labels_train.extend(train_label.cpu().tolist())

            # Perform backward pass and optimization
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0
        all_predictions_val = []
        all_labels_val = []

        # Disable gradient calculation during validation
        with torch.no_grad():
            # Iterate over the validation data batches
            for val_input, val_label in val_dataloader:
                # Move input and label tensors to the device (GPU if available)
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                # Perform forward pass
                output = model(input_id, mask)

                # Calculate the batch loss
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                # Calculate the batch accuracy
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

                # Collect predictions and labels for F1 score calculation
                all_predictions_val.extend(output.argmax(dim=1).cpu().tolist())
                all_labels_val.extend(val_label.cpu().tolist())

        # Calculate F1 scores for train and validation sets
        train_f1 = f1_score(all_labels_train, all_predictions_train, average='weighted')
        val_f1 = f1_score(all_labels_val, all_predictions_val, average='weighted')

        # Print training and validation metrics for the current epoch
        print(f"Epoch: {epoch_num + 1} \
              | Train Loss: {total_loss_train / len(train_data):.3f} \
              | Train Accuracy: {total_acc_train / len(train_data):.3f} \
              | Train F1 Score: {train_f1:.3f} \
              | Val Loss: {total_loss_val / len(val_data):.3f} \
              | Val Accuracy: {total_acc_val / len(val_data):.3f} \
              | Val F1 Score: {val_f1:.3f}")

# Define the number of epochs and learning rate
EPOCHS = 5
LR = 1e-5

# Create an instance of the SimpleGPT2SequenceClassifier model
model = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=5, max_seq_len=128, gpt_model_name="gpt2")

# Train the model
train(model, train_df, val_df, LR, EPOCHS)


"""## Evaluation"""
class testDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in test_df['label']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=128,
                                truncation=True,
                                return_tensors="pt") for text in test_df['text']]

    def classes(self):
        # Return the labels
        return self.labels

    def __len__(self):
        # Return the length of the dataset (number of samples)
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Get a batch of labels given the index 'idx'
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Get a batch of input texts given the index 'idx'
        return self.texts[idx]

    # Get a specific sample from the dataset given the index 'idx'
    def __getitem__(self, idx):

        # Get the batch of input texts for the sample
        batch_texts = self.get_batch_texts(idx)

        # Get the batch of labels for the sample
        batch_y = self.get_batch_labels(idx)

        # Return the batch of input texts and labels
        return batch_texts, batch_y

def evaluate(model, test_data):
    """
    Evaluate the model on the test data.

    Args:
        model: The trained model.
        test_data: Test data DataFrame.

    Returns:
        true_labels: List of true labels.
        predictions_labels: List of predicted labels.
    """
    # Create test dataset and dataloader
    test_dataset = testDataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2)

    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    # Tracking variables
    predictions_labels = []
    true_labels = []

    total_acc_test = 0
    with torch.no_grad():
        # Iterate over test data batches
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            # Forward pass
            output = model(input_id, mask)

            # Calculate accuracy
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

            # Add original labels
            true_labels += test_label.cpu().numpy().flatten().tolist()
            # Get predictions as a list
            predictions_labels += output.argmax(dim=1).cpu().numpy().flatten().tolist()

    # Calculate F1 score
    test_f1 = f1_score(true_labels, predictions_labels, average='weighted')

    # Print evaluation metrics
    print(f'Test Accuracy: {total_acc_test / len(test_data):.3f}')
    print(f'Test F1 Score: {test_f1:.3f}')

    return true_labels, predictions_labels

# Evaluate the model on the test dataset
true_labels, pred_labels = evaluate(model, test_df)

print(classification_report(true_labels, pred_labels))