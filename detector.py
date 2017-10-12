import tensorflow as tf
from keras.models import load_model, Model
from keras import backend as K

import time

sess = tf.Session()
K.set_session(sess)

time.sleep(1)

model = load_model('./model4b.10-0.68.hdf5')

time.sleep(2)

x = tf.placeholder(tf.float32, shape=model.get_input_shape_at(0))

time.sleep(1)

y = model(x)

import numpy as np
import matplotlib.pyplot as plt

time.sleep(1)

img = plt.imread('sushi.png')

def preprocess_input(x):
    x_copy = np.copy(x)
    x_copy -= 0.5
    x_copy *= 2.
    return x_copy

time.sleep(1)

img_processed = preprocess_input(img)
time.sleep(1)
imgs = np.expand_dims(img_processed, 0)
time.sleep(1)
orig_scores = sess.run(y, feed_dict={x: imgs, K.learning_phase(): False})

def find_top_pred(scores):
    top_label_ix = np.argmax(scores) # label 95 is Sushi
    confidence = scores[0][top_label_ix]
    print('Label: {}, Confidence: {}'.format(top_label_ix, confidence))
    
find_top_pred(orig_scores)