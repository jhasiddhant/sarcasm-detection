import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, TimeDistributed

train_data = pd.read_csv('Path to Preprocessed Train Data')
test_data = pd.read_csv('Path to Preprocessed Test Data')

# Preprocess the data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data['text'])

X_train = tokenizer.texts_to_sequences(train_data['text'])
X_test = tokenizer.texts_to_sequences(test_data['text'])

max_len = 50
vocab_size = len(tokenizer.word_index) + 1

X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

# Define the model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len))

model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))

model.add(LSTM(100, return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

# Evaluate the model on test set
test_loss, test_acc = model.evaluate(X_test, y_test)

# Print test metrics
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Predict on the test set
y_pred = model.predict(X_test)

# Convert continuous predictions to binary
threshold = 0.5
y_pred = np.where(y_pred > threshold, 1, 0)

# Print classification report for test set
print(classification_report(y_test, y_pred))

