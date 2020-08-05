from keras.models import Sequential
from keras.models import Model
import numpy as np
from keras.layers import Dense, Input, Flatten, Lambda
from keras.layers import Conv1D, MaxPooling1D, Embedding, merge, Dropout, LSTM, GRU, Bidirectional,SimpleRNN
from keras import layers
from keras.models import load_model
from keras.backend import max
import keras
'''
def buildModel_RNN(word_index, embeddings_index, nClasses, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
word_index in word index , 
embeddings_index is embeddings index, look at data_helper.py 
nClasses is number of classes, 
MAX_SEQUENCE_LENGTH is maximum lenght of text sequences, 
EMBEDDING_DIM is an int value for dimention of word embedding look at data_helper.py 
'''


def Max_layer(tensor):
    def max(tensor):
        return keras.backend.max(tensor, axis=1)
    return Lambda(max)(tensor)


def buildModel_RNN(word_index, embeddings_index, nClasses, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
    # construct model

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be random 0-1之间的浮点数
            embedding_matrix[i] = embedding_vector
    x1_input = Input((MAX_SEQUENCE_LENGTH,), dtype='int32', name='x1_input')
    embedded_x1 = layers.Embedding(input_dim=len(word_index) + 1,
                                    output_dim=EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False,
                                    mask_zero=True)(x1_input)
    gru_out_x1 = Bidirectional(GRU(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(embedded_x1)
    sent_gru_out_x1 = Bidirectional(GRU(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(gru_out_x1)   # 多层gru  batch_size,50,100
    sent_gru_out_x1 = Max_layer(sent_gru_out_x1)
    # print(sent_gru_out_x1)
    sent_feat_x1 = layers.Dense(200, activation='tanh')(sent_gru_out_x1)
    sent_feat_x1 = layers.Dropout(rate=0.2)(sent_feat_x1)

    x2_input = Input((MAX_SEQUENCE_LENGTH,), dtype='int32', name='x2_input')
    embedded_x2 = layers.Embedding(input_dim=len(word_index) + 1,
                                    output_dim=EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False,
                                    mask_zero=True)(x2_input)
    gru_out_x2 =Bidirectional(GRU(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(embedded_x2)  # 多层gru
    sent_gru_out_x2 = Bidirectional(GRU(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(gru_out_x2)
    sent_gru_out_x2 = Max_layer(sent_gru_out_x2)
    # print(sent_gru_out_x2)
    sent_feat_x2 = layers.Dense(200, activation='tanh')(sent_gru_out_x2)
    sent_feat_x2 = layers.Dropout(rate=0.2)(sent_feat_x2)
    merged = layers.add([sent_feat_x1, sent_feat_x2])
    # concatenated = Flatten()(concatenated)
    output = layers.Dense(1, activation='sigmoid')(merged)
    print("class:", nClasses)
    model = Model([x1_input, x2_input], output)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    return model


def buildModel_RNN_layer2(word_index, embeddings_index, nClasses, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
    # construct model


    model = load_model('best_model_1_100_nopos_add_3m29.h5', custom_objects={'keras': keras})
    model1 = Model(inputs=model.input, outputs=model.get_layer('lambda_1').output)
    model2 = Model(inputs=model.input, outputs=model.get_layer('lambda_2').output)
    for layer in model1.layers:
        layer.trainable = False  #原来的不训练
    for layer in model2.layers:
        layer.trainable = False
    output1 = model1.output
    output2 = model2.output
    sent_feat_x1 = layers.Dense(200, activation='tanh')(output1)
    sent_feat_x1 = layers.Dropout(rate=0.2)(sent_feat_x1)

    sent_feat_x2 = layers.Dense(200, activation='tanh')(output2)
    sent_feat_x2 = layers.Dropout(rate=0.2)(sent_feat_x2)
    merged = layers.add([sent_feat_x1, sent_feat_x2])
    output = layers.Dense(1, activation='sigmoid')(merged)
    print("class:" , nClasses)
    self_model = Model(model1.inputs, output)
    self_model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    self_model.summary()
    return self_model