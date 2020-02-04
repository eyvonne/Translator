import pandas as pd
from gensim.corpora.dictionary import Dictionary 
from gensim.utils import tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

def nlp_to_nums(df):
    '''this function can be used to process training data. It takes in a dataframe of all the phrases to be included
    and returns the english and french dictionary as well as the augmented dataframe
    
    A future release of this will include the ability to pass in dictionary objects'''
    
    #tokenize and normalize the data
    df['french_tokens'] = df['french'].apply(tokenize, lower=True, deacc=False).apply(lambda x: list(x))
    df['english_tokens'] = df['english'].apply(tokenize, lower=True).apply(lambda x: list(x))
    print('Tokens Created')
    
    #create the dictionaries
    id2fren = Dictionary(df['french_tokens'])
    id2eng = Dictionary(df['english_tokens'])
    print('Dictionaries Built')
    
    #transform the words into numbers
    def en_doc2num(tokens):
        nums=[]
        for token in tokens:
            nums.append(id2eng.token2id[token])
        return nums
    
    def fr_doc2num(tokens):
        nums=[]
        for token in tokens:
            nums.append(id2fren.token2id[token])
        return nums
    
    df['english_bow'] = df['english_tokens'].apply(en_doc2num)
    df['french_bow'] = df['french_tokens'].apply(fr_doc2num)
    print('BOWs built')
    
    #pad the sequences 
    processed_data['english_padded'] = pad_sequences(
        processed_data['english_bow'],
        maxlen=50,
        dtype='int32',
        padding='post',
        value=-1
    ).tolist()
    
    processed_data['french_padded'] = pad_sequences(
        processed_data['french_bow'],
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


