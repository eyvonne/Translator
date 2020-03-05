import pandas as pd
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.utils import tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


def nlp_to_nums(df):
    '''this function can be used to process training data. It takes in a dataframe of all the phrases to be included
    and returns the english and french dictionary as well as the augmented dataframe

    A future release of this will include the ability to pass in dictionary objects'''

    # tokenize and normalize the data
    df['french_tokens'] = df['french'].apply(
        tokenize, lower=True, deacc=False).apply(lambda x: list(x))
    df['english_tokens'] = df['english'].apply(tokenize, lower=True).apply(lambda x: list(x))
    print('Tokens Created')

    # create the dictionaries
    id2fren = Dictionary(df['french_tokens'])
    id2eng = Dictionary(df['english_tokens'])
    print('Dictionaries Built')

    # transform the words into numbers
    def en_doc2num(tokens):
        nums = []
        for token in tokens:
            nums.append(id2eng.token2id[token])
        return nums

    def fr_doc2num(tokens):
        nums = []
        for token in tokens:
            nums.append(id2fren.token2id[token])
        return nums

    df['english_bow'] = df['english_tokens'].apply(en_doc2num)
    df['french_bow'] = df['french_tokens'].apply(fr_doc2num)
    print('BOWs built')

    # pad the sequences
    df['english_padded'] = pad_sequences(
        df['english_bow'],
        maxlen=60,
        dtype='int32',
        padding='post',
        value=-1
    ).tolist()

    df['french_padded'] = pad_sequences(
        df['french_bow'],
        maxlen=60,
        dtype='int32',
        padding='post',
        value=-1
    ).tolist()

    return df, id2fren, id2eng


def decode_french(nums, id2fren):
    ''' this function takes in a dictionary and a sentence that has been predicted by the model and returns it in french'''
    return [id2fren[num] for num in nums if num != -1]


def encode_english(sentence, id2eng):
    ''' this function takes in a sentence in natural language and a dictionary and returns a padded encoding of the sentence.'''
    words = tokenize(sentence, lower=True)
    nums = [id2eng.token2id[token] for token in words]
    padded = pad_sequences(
        [nums],
        maxlen=50,
        dtype='int32',
        padding='post',
        truncating='post',
        value=-1
    )
    return padded


def str_to_int(data):
    data = data.strip('[]').split(', ')
    data = [int(x) for x in data]
    return np.asarray(data)


def load_data():
    '''this function loads up the already processed data with all of the nested lists properly reformatted as lists, and loads up the dictionaries'''
    df = pd.read_csv('data/processed_full.tsv', sep='\t')
    df['english_tokens'] = df['english_tokens'].apply(lambda x: x.strip("['']").split("', '"))
    df['french_tokens'] = df['french_tokens'].apply(lambda x: x.strip("['']").split("', '"))
    df['english_bow'] = df['english_bow'].apply(str_to_int)
    df['french_bow'] = df['french_bow'].apply(str_to_int)
    df['english_padded'] = df['english_padded'].apply(str_to_int)
    df['french_padded'] = df['french_padded'].apply(str_to_int)
    df = df.drop('Unnamed: 0', axis=1)

    eng = Dictionary.load('data/Dictionaries/eng')
    fren = Dictionary.load('data/Dictionaries/fren')

    # create ML data
    X_eng = np.vstack(df['english_padded'].values)
    y_fren = np.vstack(df['french_padded'].values)

    y_fren = y_fren.reshape(*y_fren.shape, 1)
    X_eng = X_eng.reshape(*X_eng.shape, 1)

    return df, eng, fren, X_eng, y_fren


def simple_model(input_shape, output_length, eng_vocab_size, french_vocab_size):
    # hyperparameters
    learning_rate = .005

    # TODO: Build the layers
    model = Sequential()
    model.add(GRU(256, input_shape=input_shape[1:], return_sequences=True))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))

    # Compile model
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    print('loading data')
    data, id2eng, id2fren, X_eng, y_fren = load_data()
    print('splitting data')
    X_train_eng, X_test_eng, y_train_fren, y_test_fren = train_test_split(X_eng, y_fren)
    print('creating model')
    simple_rnn = simple_model(X_train_eng.shape, 60, len(id2eng), len(id2fren))
    print('starting training')
    simple_rnn.fit(X_train_eng, y_train_fren, batch_size=1024, epochs=10, validation_split=.2)
    print(simple_rnn.evaluate(X_test_eng, y_test_fren, batch_size=1024))
