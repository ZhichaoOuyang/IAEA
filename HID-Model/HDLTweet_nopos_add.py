# coding=utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
# from keras.models import Sequential
from keras.models import Model
import keras
from keras import backend as K
import tensorflow as tf
import Data_helper_Tweet_npyload_nopos as Data_helper
import BuildModelForTweet_nopos_add as BuildModel
from keras.callbacks import ModelCheckpoint


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    将输入的Session保存为静态的计算图结构.
    创建一个新的计算图，其中的节点以及权重和输入的Session相同. 新的计算图会将输入Session中不参与计算的部分删除。
    @param session 需要被保存的Session.
    @param keep_var_names 一个记录了需要被保存的变量名的list，若为None则默认保存所有的变量.
    @param output_names 计算图相关输出的name list.
    @param clear_devices 若为True的话会删除不参与计算的部分，这样更利于移植，否则可能移植失败
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


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
    checkpoint1 = ModelCheckpoint(filepath='best_model_1_100_nopos_add.h5', monitor='val_acc', mode='auto', save_best_only='True')
    checkpoint2 = ModelCheckpoint(filepath='best_model_22_100_nopos_add.h5', monitor='val_acc', mode='auto', save_best_only='True')
    MEMORY_MB_MAX = 1600000 # maximum memory you can use
    MAX_SEQUENCE_LENGTH = 50 # Maximum sequance lenght 500 words
    MAX_NB_WORDS = 55000 # Maximum number of unique words
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
        # embedding1_layer_model = Model(inputs=model.input, outputs=model.get_layer('embedding_1').output)
        # embedding2_layer_model = Model(inputs=model.input, outputs=model.get_layer('embedding_2').output)
        # mp = "embedding1_model.h5"
        # embedding1_layer_model.save(mp)
        # mp = "embedding2_model.h5"
        # embedding2_layer_model.save(mp)
        #
        # embedding1_layer_model = Model(inputs=model.input, outputs=model.get_layer('bidirectional_3').output)
        # embedding2_layer_model = Model(inputs=model.input, outputs=model.get_layer('bidirectional_4').output)
        # mp = "bi1_model.h5"
        # embedding1_layer_model.save(mp)
        # mp = "bi2_model.h5"
        # embedding2_layer_model.save(mp)
        #
        # embedding1_layer_model = Model(inputs=model.input, outputs=model.get_layer('dropout_1').output)
        # embedding2_layer_model = Model(inputs=model.input, outputs=model.get_layer('dropout_2').output)
        # mp = "dp1_model.h5"
        # embedding1_layer_model.save(mp)
        # mp = "dp2_model.h5"
        # embedding2_layer_model.save(mp)


        print("save embedding layer model success")
        print("Saving model to disk \n")
        mp = "model1_100_nopos_add.h5"
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
        mp = "model22_100_nopos_add.h5"
        model2.save(mp)







