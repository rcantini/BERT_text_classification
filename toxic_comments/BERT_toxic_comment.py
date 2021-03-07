import pandas as pd
from transformers import TFBertModel
seed_value = 42
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
np.set_printoptions(precision=2)
import tensorflow as tf
tf.random.set_seed(seed_value)
import tensorflow_addons as tfa
import keras
import keras.layers as layers
from keras.callbacks import ModelCheckpoint
from BERT_text_classification.utils import bert_utils, plots


N_CLASSES = 6
MAX_SEQ_LEN = 128
BERT_NAME = 'bert-base-uncased'
classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def get_data(train_n=5000, val_n=500, test_n=1024):
    labeled_data = pd.read_csv('comment_data//train.csv')
    labeled_data = labeled_data.sample(frac=1)
    labels = labeled_data[classes].values.astype(np.float32)
    lab_sentences = labeled_data["comment_text"].fillna("fillna").str.lower()
    train_sentences = lab_sentences[:train_n]
    y_train = labels[:train_n]
    val_sentences = lab_sentences[train_n:train_n+val_n]
    y_val = labels[train_n:train_n+val_n]
    test_sentences = lab_sentences[train_n+val_n:train_n+val_n+test_n]
    y_test = labels[train_n+val_n:train_n+val_n+test_n]
    return train_sentences, y_train, val_sentences, y_val, test_sentences, y_test


def create_model():
    input_ids = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='input_ids')
    input_type = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='token_type_ids')
    input_mask = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='attention_mask')
    inputs = [input_ids, input_type, input_mask]
    bert = TFBertModel.from_pretrained(BERT_NAME)
    bert_outputs = bert(inputs)
    last_hidden_states = bert_outputs.last_hidden_state
    avg = layers.GlobalAveragePooling1D()(last_hidden_states)
    output = layers.Dense(N_CLASSES, activation="sigmoid")(avg)
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


######### BERT TOXICITY CLASSIFICATION #########

# 1) get an extract of the Toxicity Comments Dataset
train_sentences, y_train, val_sentences, y_val, test_sentences, y_test = get_data()

# 2) encode sentences following the BERT specifications
X_train = bert_utils.prepare_bert_input(train_sentences, MAX_SEQ_LEN, BERT_NAME)
X_val = bert_utils.prepare_bert_input(val_sentences, MAX_SEQ_LEN, BERT_NAME)
X_test = bert_utils.prepare_bert_input(test_sentences, MAX_SEQ_LEN, BERT_NAME)

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
s1 = "You are a f***ing idiot!"
s2 = "I will find you, and I will kill you."
s3 = "Shut the f**k up, you stupid junkie!"
s4 = "I completely disagree with your previous statement."
sentences = np.asarray([s1, s2, s3, s4])
enc_sentences = bert_utils.prepare_bert_input(sentences, MAX_SEQ_LEN, BERT_NAME)
predictions = model.predict(enc_sentences)
classes = np.array(classes)
for sentence, pred in zip(sentences, predictions):
    mask = (pred > 0.5).astype(bool)
    tox = str(classes[mask]) if any(mask) else "not toxic"
    print('comment: '+sentence.replace("\n", "").strip()+' -- toxicity: '+tox+', scores:'+str(pred[mask]))
