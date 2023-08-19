import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, TimeDistributed
from sklearn.metrics import classification_report

train_data = 'Path to preprocessed train data'
test_data = 'Path to preprocessed test data'

# Tokenize the data
max_features = 10000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(train_data['text'])
X = tokenizer.texts_to_sequences(train_data['text'])
X = pad_sequences(X)

# Model Architecture
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(TimeDistributed(Dense(128, activation='relu')))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(TimeDistributed(Dense(128, activation='relu')))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, train_data['label'], batch_size=32, epochs=10, validation_split=0.2)

# Predict on test data
test_sequences = tokenizer.texts_to_sequences(test_data['text'])
test_sequences = pad_sequences(test_sequences, maxlen=X.shape[1])
y_pred = model.predict(test_sequences)

# Convert predictions to labels
y_pred = np.round(y_pred).astype(int)

# Evaluate the model
print(classification_report(test_data['label'], y_pred))