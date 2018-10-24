'''Created on 23/10/2018 by 
Sai Krishna Dyavarashetty'''

import os, sys
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# change this as you see fit
image_path = sys.argv[1]

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("./trained_model/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("./trained_model/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    class_id = top_k[0]
    class_name = label_lines[class_id]
    print(top_k[0])
    #print(len(top_k))
    print('%s (score = %.5f)' % (class_name, predictions[0][class_id]))
    '''for node_id in top_k:
        #print(node_id)
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))'''