# from tensorflow.python.platform import gfile
# import tensorflow as tf
# import numpy as np
#
# def load_model(pb_filename,x1,x2,old_num, new_num):
#     # 初始化TensorFlow的session
#
#     result = np.zeros([old_num], dtype=np.int32)
#
#     with tf.Session() as sess:
#         # 读取得到的pb文件加载模型
#         with gfile.FastGFile("selector/model.pb", 'rb') as f:
#             graph_def = tf.GraphDef()
#             graph_def.ParseFromString(f.read())
#             # 把图加到session中
#             tf.import_graph_def(graph_def, name='')
#
#         # 获取当前计算图
#         graph = tf.get_default_graph()
#
#         # 从图中获输出那一层
#         pred = graph.get_tensor_by_name("dense_1/Sigmoid:0")
#
#         # 运行并预测输入的x1,x2 总共有old_num * new_num对
#         label = sess.run(pred, feed_dict={"x1_input:0": x1, "x2_input:0": x2})
#         flag = 0
#         while flag < old_num:
#             layer2_X1 = []
#             layer2_X2 = []
#             f = False
#             for i in range((new_num * flag),(new_num * (flag+1))):
#                 if label[i] > 0.5:
#                     layer2_X1.append(x1[i])
#                     layer2_X2.append(x2[i])   # 进入第二层判断
#                     f = True
#             if f==False:
#                 result[flag] = 0   # 遍历完没有冲突
#                 continue
#             with gfile.FastGFile("selector/model2.pb", 'rb') as f:
#                 graph_def = tf.GraphDef()
#                 graph_def.ParseFromString(f.read())
#                 # 把图加到session中
#                 tf.import_graph_def(graph_def, name='')
#
#             # 获取当前计算图
#             graph = tf.get_default_graph()
#
#             # 从图中获输出那一层
#             pred = graph.get_tensor_by_name("dense_2/Sigmoid:0")
#
#             # 运行并预测输入的x1,x2 总共有old_num * new_num对
#             label2 = sess.run(pred, feed_dict={"x1_input:0": layer2_X1, "x2_input:0": layer2_X2})
#             for i in range(len(layer2_X1)):
#                 if label[i] > 0.5:
#                     result[flag] = 1
#                     f = True
#             if f == False:
#                 result[flag] = 0
#             flag += 1
#         return result
