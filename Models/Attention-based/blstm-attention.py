import os
import pandas as pd
import random
import pandas as pd
from keras import backend as K
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Dense, Lambda, Dot, Activation, Concatenate, Layer

debug_flag = int(os.environ.get('KERAS_ATTENTION_DEBUG', 0))

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def dataset_embedding(dataset_path, tokenizer, batch_size=32):
    dataset = pd.read_csv(dataset_path)[["text", "label"]]
    dataset = dataset[dataset['text'].notna()]

    tokenized_texts = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)) for text in dataset['text']]

    texts_with_len = [[text, dataset['label'].iloc[i], len(text)] for i, text in enumerate(tokenized_texts)]
    random.Random(42).shuffle(texts_with_len)

    texts_with_len.sort(key=lambda x: x[2])
    sorted_texts_labels = [(text_lab[0], text_lab[1]) for text_lab in texts_with_len]
    processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_texts_labels, output_types=(tf.int32, tf.int32))

    return processed_dataset.padded_batch(batch_size, padded_shapes=((None, ), ()))

def prepare_datasets(train_path, test_path):
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

    dataset_train = dataset_embedding(train_path, tokenizer)
    dataset_test = dataset_embedding(test_path, tokenizer)

    return dataset_train, dataset_test, tokenizer

class Attention(object if debug_flag else Layer):
    def __init__(self, units=128, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        with K.name_scope(self.name if not debug_flag else 'attention'):
            self.attention_score_vec = Dense(input_dim, use_bias=False, name='attention_score_vec')
            self.h_t = Lambda(lambda x: x[:, -1, :], output_shape=(input_dim,), name='last_hidden_state')
            self.attention_score = Dot(axes=[1, 2], name='attention_score')
            self.attention_weight = Activation('softmax', name='attention_weight')
            self.context_vector = Dot(axes=[1, 1], name='context_vector')
            self.attention_output = Concatenate(name='attention_output')
            self.attention_vector = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')
        if not debug_flag:
            super(Attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def __call__(self, inputs, training=None, **kwargs):
        if debug_flag:
            return self.call(inputs, training, **kwargs)
        else:
            return super(Attention, self).__call__(inputs, training, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        if debug_flag:
            self.build(inputs.shape)
        score_first_part = self.attention_score_vec(inputs)
        h_t = self.h_t(inputs)
        score = self.attention_score([h_t, score_first_part])
        attention_weights = self.attention_weight(score)
        context_vector = self.context_vector([inputs, attention_weights])
        pre_activation = self.attention_output([context_vector, h_t])
        attention_vector = self.attention_vector(pre_activation)
        return attention_vector

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({'units': self.units})
        return config


train_path = 'Path to preprocessed train data'
test_path = 'Path to preprocessed test data'
train_data, test_data, tokenizer = prepare_datasets(train_path, test_path)

att_Blstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.vocab), 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    Attention(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
att_Blstm.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy', f1_m])
print(att_Blstm.summary())

att_Blstm.fit(train_data, epochs=10, validation_data = test_data, class_weight={1:4, 0:1})

loss, accuracy, f1_score = att_Blstm.evaluate(test_data)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')
print(f'Test F1 score: {f1_score:.4f}')

y_true = []
for batch in test_data:
    labels = batch[1].numpy()
    y_true.extend(labels)

predictions = att_Blstm.predict(test_data)
y_pred = (predictions > 0.5).astype('int32')

report = classification_report(y_true, y_pred)
print(report)