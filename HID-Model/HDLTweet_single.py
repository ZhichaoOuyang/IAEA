"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
RMDL: Random Multimodel Deep Learning for Classification

* Copyright (C) 2018  Kamran Kowsari <kk7nc@virginia.edu>
* Last Update: Oct 26, 2018
* This file is part of  HDLTex project, University of Virginia.
* Free to use, change, share and distribute source code of RMDL
* Refrenced paper : HDLTex: Hierarchical Deep Learning for Text Classification
* Link: https://doi.org/10.1109/ICMLA.2017.0-134
* Comments and Error: email: kk7nc@virginia.edu
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# coding=utf-8
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
# from keras.models import Sequential
from keras.models import Model
import keras
from keras import backend as K
import tensorflow as tf
import Data_helper_Tweet_single as Data_helper
import BuildModelForTweet_single as BuildModel
from keras.callbacks import ModelCheckpoint
if __name__ == "__main__":
    # num_cores = 4
    # config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, \
    #                         inter_op_parallelism_threads=num_cores, allow_soft_placement=True, \
    #                         device_count={'CPU': 1, 'GPU': 1})
    # session = tf.Session(config=config)
    # K.set_session(session)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    checkpoint1 = ModelCheckpoint(filepath='best_model_single_3m29.h5', monitor='val_acc', mode='auto', save_best_only='True')
    MEMORY_MB_MAX = 1600000 # maximum memory you can use
    MAX_SEQUENCE_LENGTH = 50 # Maximum sequance lenght 500 words
    MAX_NB_WORDS = 50000 # Maximum number of unique words
    EMBEDDING_DIM = 100 #embedding dimension you can change it to {25, 100, 150, and 300} but need to change glove version
    batch_size_L1 = 64 # batch size in Level 1
    batch_size_L2 = 64 # batch size in Level 2
    epochs = 100

    L1_model =2 # 0 is DNN, 1 is CNN, and 2 is RNN for Level 1
    L2_model =2 # 0 is DNN, 1 is CNN, and 2 is RNN for Level 2

    np.set_printoptions(threshold=np.inf) # 打印数组所有元素
    '''
    Tokenizer that is using GLOVE
    '''

    X_train1, X_train2, y_train, X_test1, X_test2, y_test,word_index, embeddings_index,number_of_classes_L1 = \
        Data_helper.loadData_Tokenizer(MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)
    checkpoint1 = ModelCheckpoint(filepath='best_model_single_nopos.h5', monitor='val_acc', mode='auto', save_best_only='True')
    print("Loading Data is Done")
    # RNN Level 1

    if L1_model == 2:
        callback_lists = [checkpoint1]
        print('Create model of RNN')
        model = BuildModel.buildModel_RNN(word_index, embeddings_index,number_of_classes_L1,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM)
        model.fit({'x1_input': X_train1, 'x2_input': X_train2}, y_train,
                  epochs=epochs,
                  verbose=2,
                  validation_data=({'x1_input': X_test1, 'x2_input': X_test2}, y_test),
                  batch_size=batch_size_L1,
                  callbacks=callback_lists)
        # embedding1_layer_model = Model(inputs=model.input, outputs=model.get_layer('embedding_1').output)
        # embedding2_layer_model = Model(inputs=model.input, outputs=model.get_layer('embedding_2').output)
        # mp = "embedding1_layer_model_lastest_poslab_epch100_true.h5"
        # embedding1_layer_model.save(mp)
        # mp = "embedding2_layer_model_lastest_poslab_epch100_true.h5"
        # embedding2_layer_model.save(mp)
        # print("save embedding layer model success")
        print("Saving model to disk \n")
        mp = "model_single_nopos_3m29.h5"
        model.save(mp)





