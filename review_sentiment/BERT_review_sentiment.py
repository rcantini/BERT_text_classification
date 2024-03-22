from transformers import TFBertModel
seed_value = 42
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)
import tensorflow_addons as tfa
import keras
import keras.layers as layers
from keras.callbacks import ModelCheckpoint
from BERT_text_classification.utils import bert_utils, plots


MAX_SEQ_LEN = 128
# class labels
LABELS = {"NEG": 0, "POS": 1}
BERT_NAME = 'bert-base-uncased'
classes = list(LABELS.keys())


def get_data(n_train=5000, n_val=500, n_test=512):  # per-class number of samples
    sentences_train, y_train = [], []
    sentences_val, y_val = [], []
    sentences_test, y_test = [], []
    neg = np.array(os.listdir("review_data//train//neg"))
    pos = np.array(os.listdir("review_data//train//pos"))
    neg_test = np.array(os.listdir("review_data//test//neg"))
    pos_test = np.array(os.listdir("review_data//test//pos"))
    p = np.random.permutation(len(neg))
    neg_train, pos_train = neg[p][:n_train], pos[p][:n_train]
    neg_val, pos_val = neg[p][n_train:n_train+n_val], pos[p][n_train:n_train+n_val]
    p_test = np.random.permutation(len(neg_test))
    neg_test, pos_test = neg_test[p_test][:n_test], pos_test[p_test][:n_test]
    for file in neg_train:
        with open("review_data//train//neg//" + file, "r", encoding="utf-8") as input:
            line = input.readline()
            sentences_train.append(line)
            y_train.append(LABELS["NEG"])
    for file in pos_train:
        with open("review_data//train//pos//" + file, "r", encoding="utf-8") as input:
            line = input.readline()
            sentences_train.append(line)
            y_train.append(LABELS["POS"])
    for file in neg_val:
        with open("review_data//train//neg//" + file, "r", encoding="utf-8") as input:
            line = input.readline()
            sentences_val.append(line)
            y_val.append(LABELS["NEG"])
    for file in pos_val:
        with open("review_data//train//pos//" + file, "r", encoding="utf-8") as input:
            line = input.readline()
            sentences_val.append(line)
            y_val.append(LABELS["POS"])
    for file in neg_test:
        with open("review_data//test//neg//" + file, "r", encoding="utf-8") as input:
            line = input.readline()
            sentences_test.append(line)
            y_test.append(LABELS["NEG"])
    for file in pos_test:
        with open("review_data//test//pos//" + file, "r", encoding="utf-8") as input:
            line = input.readline()
            sentences_test.append(line)
            y_test.append(LABELS["POS"])
    sentences_train, y_train = np.asarray(sentences_train), np.asarray(y_train)
    sentences_val, y_val = np.asarray(sentences_val), np.asarray(y_val)
    sentences_test, y_test = np.asarray(sentences_test), np.asarray(y_test)
    p_train = np.random.permutation(len(sentences_train))
    sentences_train, y_train = sentences_train[p_train], y_train[p_train]
    p_val = np.random.permutation(len(sentences_val))
    sentences_val, y_val = sentences_val[p_val], y_val[p_val]
    p_test = np.random.permutation(len(sentences_test))
    sentences_test, y_test = sentences_test[p_test], y_test[p_test]
    return sentences_train, y_train, sentences_val, y_val, sentences_test, y_test


def create_model():
    input_ids = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='input_ids')
    input_mask = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='attention_mask')
    input_type = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='token_type_ids')
    inputs = [input_ids, input_mask, input_type]
    bert = TFBertModel.from_pretrained(BERT_NAME)
    bert_outputs = bert(inputs)
    last_hidden_states = bert_outputs.last_hidden_state
    avg = layers.GlobalAveragePooling1D()(last_hidden_states)
    output = layers.Dense(1, activation="sigmoid")(avg)
    model = keras.Model(inputs=inputs, outputs=output)
    model.summary()
    return model


def fine_tune(model, X_train, x_val, y_train, y_val):
    max_epochs = 4
    batch_size = 32
    opt = tfa.optimizers.RectifiedAdam(learning_rate=3e-5)
    loss = keras.losses.BinaryCrossentropy()
    best_weights_file = "weights.h5"
    m_ckpt = ModelCheckpoint(best_weights_file, monitor='val_auc', mode='max', verbose=2,
                             save_weights_only=True, save_best_only=True)
    model.compile(loss=loss, optimizer=opt, metrics=[keras.metrics.AUC(multi_label=True, curve="ROC"),
                                                     keras.metrics.BinaryAccuracy()])
    model.fit(
        X_train, y_train,
        validation_data=(x_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[m_ckpt],
        verbose=2
    )


def load_model():
    model = create_model()
    loss = keras.losses.BinaryCrossentropy()
    best_weights_file = "weights.h5"
    model.load_weights(best_weights_file)
    opt = tfa.optimizers.RectifiedAdam(learning_rate=3e-5)
    model.compile(loss=loss, optimizer=opt, metrics=[keras.metrics.AUC(multi_label=True, curve="ROC"),
                                                     keras.metrics.BinaryAccuracy()])
    return model


######### BERT SENTIMENT CLASSIFICATION #########

# 1) get an extract of the IMDB Reviews Dataset
sentences_train, y_train, sentences_val, y_val, sentences_test, y_test = get_data()

# 2) encode sentences following the BERT specifications
X_train = bert_utils.prepare_bert_input(sentences_train, MAX_SEQ_LEN, BERT_NAME)
X_val = bert_utils.prepare_bert_input(sentences_val, MAX_SEQ_LEN, BERT_NAME)
X_test = bert_utils.prepare_bert_input(sentences_test, MAX_SEQ_LEN, BERT_NAME)

# 3) create the classification model
model = create_model()

# 4) full model fine tuning
fine_tune(model, X_train, X_val, y_train, y_val)

# 5) evaluate on test set
model = load_model()
predictions = model.predict(X_test)
model.evaluate(X_test, y_test, batch_size=32)
plots.plot_roc_auc(y_test, predictions, classes)
# 6) predict sentiment for some test sentences
sentences = np.asarray(["This is the best film of the year!", "A colossal waste of time!",
                        "If you are sad, watching this film will surely cheer you up.", "It's a must watch!",
                        "Can't wait for it's next part!", "Just wow!", "Save your self from this 90 mins disaster!",
                        "It fell short of expectations.", "The right movie if you like to waste your money."])
enc_sentences = bert_utils.prepare_bert_input(sentences, MAX_SEQ_LEN, BERT_NAME)
predictions = model.predict(enc_sentences)
for s, p in zip(sentences, predictions):
    sentiment = classes[round(p[0])]
    print('review: '+s.replace("\n", "").strip()+' -- sentiment: '+sentiment+', score: '+str(p[0]))
