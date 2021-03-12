# coding=utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
# from keras.models import Sequential
from keras.models import Model
import keras
from keras import backend as K
import tensorflow as tf
import Data_helper_Tweet_npyload_nopos as Data_helper
import BuildModelForTweet_nopos as BuildModel
from keras.callbacks import ModelCheckpoint

if __name__ == "__main__":

    checkpoint1 = ModelCheckpoint(filepath='best_model_1_sub.h5', monitor='val_acc', mode='auto', save_best_only='True')
    checkpoint2 = ModelCheckpoint(filepath='best_model_2_sub.h5', monitor='val_acc', mode='auto', save_best_only='True')
    MEMORY_MB_MAX = 1600000 # maximum memory you can use
    MAX_SEQUENCE_LENGTH = 50 # Maximum sequance lenght 500 words
    MAX_NB_WORDS = 50000 # Maximum number of unique words
    EMBEDDING_DIM = 100 #embedding dimension you can change it to {25, 100, 150, and 300} but need to change glove version
    batch_size_L1 = 64 # batch size in Level 1
    batch_size_L2 = 64 # batch size in Level 2
    epochs = 50

    L1_model =2 # 0 is DNN, 1 is CNN, and 2 is RNN for Level 1
    L2_model =2 # 0 is DNN, 1 is CNN, and 2 is RNN for Level 2

    np.set_printoptions(threshold=np.inf)
    '''
      Tokenizer that is using GLOVE
    '''

    X_train1, X_train2, y_train, X_test1, X_test2, y_test, content1_L2_Train, content2_L2_Train, L2_Train, content1_L2_Test, content2_L2_Test, L2_Test, number_of_classes_L2,word_index, embeddings_index,number_of_classes_L1 = \
        Data_helper.loadData_Tokenizer(MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)
    print("Loading Data is Done")
    print(X_train1.shape)
    print(X_train2.shape)
    print(X_test1.shape)
    print(X_test2.shape)

    if L1_model == 2:
        callback_lists = [checkpoint1]
        print('Create model of RNN')
        model = BuildModel.buildModel_RNN(word_index, embeddings_index,number_of_classes_L1,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM)
        model.fit({'x1_input': X_train1, 'x2_input': X_train2}, y_train[:,0],
                  epochs=epochs,
                  verbose=2,
                  validation_data=({'x1_input': X_test1, 'x2_input': X_test2}, y_test[:,0]),
                  batch_size=batch_size_L1,
                  callbacks=callback_lists)

        print("save embedding layer model success")
        print("Saving model to disk \n")
        mp = "model_1_sub.h5"
        model.save(mp)

    # RNN Level 2
    if L2_model == 2:
        callback_lists = [checkpoint2]
         # i = i + 1
        print('Create Sub model of ', 1)
        model2 = BuildModel.buildModel_RNN_layer2(word_index, embeddings_index,number_of_classes_L2[1],MAX_SEQUENCE_LENGTH,EMBEDDING_DIM)
        model2.fit({'x1_input': content1_L2_Train[1], 'x2_input': content2_L2_Train[1]}, L2_Train[1],
                      epochs=epochs,
                      verbose=2,
                      validation_data=({'x1_input': content1_L2_Test[1], 'x2_input': content2_L2_Test[1]}, L2_Test[1]),
                      batch_size=batch_size_L2,
                      callbacks=callback_lists)
        print("Saving model to disk \n")
        mp = "model_2_sub.h5"
        model2.save(mp)







