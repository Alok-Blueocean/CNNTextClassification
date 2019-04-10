import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk.corpus import stopwords
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation


class CNNModel:

    MAX_NB_WORDS = 200000
    MAX_SEQUENCE_LENGTH = 15
    EMBEDDING_DIM = 300
    VALIDATION_SPLIT = 0.2
    EMBEDDING_FILE = "/home/alok.r/CNN/GoogleNews-vectors-negative300.bin"
    input_file = "/home/alok.r/CNN/Annotations.xlsx"
    x_train=''
    y_train=''
    x_val=''
    y_val=''
    model=''
    STOPWORDS = set(stopwords.words("english"))

    def __init__(self):

        self.tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS)

    def preprocess(self,text):
        text = text.strip().lower().split()
        text = filter(lambda word: word not in self.STOPWORDS, text)
        return " ".join(text)

    def data_preparation(self):

        df = pd.read_excel(self.input_file, sheetname="Label H")
        df.dropna()
        titleh = df["Title"].tolist()
        labelh = df["LABEL"].tolist()

        df = pd.read_excel(self.input_file, sheetname="Lable S")
        df.dropna()
        titles = df["Title"].tolist()
        labels = df["LABEL"].tolist()

        df = pd.read_excel(self.input_file, sheetname="Label P")
        df.dropna()
        titlep = df["Title"].tolist()
        labelp = df["LABEL"].tolist()

        df = pd.read_excel(self.input_file, sheetname="Label A")
        df.dropna()
        titlea = df["Title"].tolist()
        labela = df["LABEL"].tolist()

        title = titleh + titles + titlep + titlea
        label = labelh + labels + labelp + labela

        unique_labels = [u'ONLY SYMP', u'ONLY DRUGONLY DRUG', u'ONLY DRUG', u'OTHERS',
                         u'DRUG_DIS_SYMP', u'DRUG_ADE_ADR', u'DRUG_DIS_SYM', u'ONLY ADE']

        title_list = []
        label_list = []
        for tit, lab in zip(title, label):

            try:
                if len(tit) > 50:
                    if lab.__contains__(","):
                        lab = lab.split(",")[0].strip()
                    label_list.append(unique_labels.index(lab.strip()))
                    title_list.append(self.preprocess(tit))

            except Exception as e:
                print e

        return title_list,label_list

    def train_test_data(self,data,category):
        nb_validation_samples = int(self.VALIDATION_SPLIT * data.shape[0])
        self.x_train = data[:-nb_validation_samples]
        self.y_train = category[:-nb_validation_samples]
        self.x_val = data[-nb_validation_samples:]
        self.y_val = category[-nb_validation_samples:]

    def data_embedding(self):

        word2vec = KeyedVectors.load_word2vec_format(self.EMBEDDING_FILE, binary=True)
        word_index = self.tokenizer.word_index
        nb_words = min(self.MAX_NB_WORDS, len(word_index)) + 1


        embedding_matrix = np.zeros((nb_words, self.EMBEDDING_DIM))
        for word, i in word_index.items():

            if word in word2vec.vocab:
                embedding_matrix[i] = word2vec.word_vec(word)

        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

        embedding_layer = Embedding(embedding_matrix.shape[0],  # or len(word_index) + 1
                                    embedding_matrix.shape[1],  # or EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=self.MAX_SEQUENCE_LENGTH,
                                    trainable=False)

        return embedding_layer

    def data_vector(self, title_list,label_list):

        all_texts = title_list

        self.tokenizer.fit_on_texts(all_texts)

        clothing_sequences =  self.tokenizer.texts_to_sequences(all_texts)
        clothing_data = pad_sequences(clothing_sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        word_index =  self.tokenizer.word_index

        data = clothing_data
        category = to_categorical(label_list)

        indices = np.arange(data.shape[0])  # get sequence of row index
        np.random.shuffle(indices)  # shuffle the row indexes
        data = data[indices]  # shuffle data/product-titles/x-axis
        category = category[indices]  # shuffle labels/category/y-axis
        return data,category

    def model_building(self,embedding_layer):

        self.model = Sequential()
        self.model.add(embedding_layer)
        self.model.add(Dropout(0.2))
        self.model.add(Conv1D(300, 3, padding='valid', activation='relu', strides=2))
        self.model.add(Conv1D(150, 3, padding='valid', activation='relu', strides=2))
        self.model.add(Conv1D(75, 3, padding='valid', activation='relu', strides=2))
        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(150, activation='sigmoid'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(8, activation='sigmoid'))

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

        self.model.summary()

        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), epochs=2, batch_size=128)

    def model_evaluation(self):
        return self.model.evaluate(self.x_val, self.y_val, verbose=0)

if __name__=="__main__":
    cnnmodel = CNNModel()
    title_list,label_list = cnnmodel.data_preparation()
    data,category = cnnmodel.data_vector(title_list,label_list)
    cnnmodel.train_test_data(data,category)
    embedding_layer = cnnmodel.data_embedding()
    cnnmodel.model_building(embedding_layer)