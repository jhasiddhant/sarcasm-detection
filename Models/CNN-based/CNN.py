import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

train_data = pd.read_csv('Path to Preprocessed Train Data')
test_data = pd.read_csv('Path to Preprocessed Test Data')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data['text'], train_data['label'], test_size=0.2)

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(test_data['text'])

# Pad the sequences
max_len = max([len(x) for x in X_train_seq])
X_train_seq = pad_sequences(X_train_seq, maxlen=max_len)
X_val_seq = pad_sequences(X_val_seq, maxlen=max_len)
X_test_seq = pad_sequences(X_test_seq, maxlen=max_len)

# Build the model
vocab_size = len(tokenizer.word_index) + 1
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_len))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val), epochs=5)

# Evaluate the model on the test set
y_pred = model.predict(X_test_seq)
y_pred = [round(x[0]) for x in y_pred]

print(classification_report(test_data['label'], y_pred))
