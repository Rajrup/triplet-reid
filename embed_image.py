# %%
import json
import os
from importlib import import_module

import cv2
import tensorflow as tf
import numpy as np

# %%
sess = tf.Session()

# Read Config
config = json.loads(open(os.path.join('./', 'args.json'), 'r').read())

net_input_size = (config['net_input_height'], config['net_input_width'])
img_read = "cv2"
if img_read == "tf":
    # Read Image via TF
    image_encoded = tf.read_file(os.path.join(config['image_root'],'query', '0001_c1s1_001051_00.jpg'))
    image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
    image_resized = tf.image.resize_images(image_decoded, net_input_size)
    img = tf.expand_dims(image_resized, axis=0)
else:
    # Read Image using OpenCV
    raw_img = cv2.imread(os.path.join(config['image_root'],'query', '0001_c1s1_001051_00.jpg'))
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    raw_img = cv2.resize(raw_img, (net_input_size[1], net_input_size[0]))
    raw_img = np.expand_dims(raw_img, axis=0)
    img = tf.placeholder(tf.float32, (None, net_input_size[0], net_input_size[1], 3))

# %%
# Create the model and an embedding head.
model = import_module('nets.' + config['model_name'])
head = import_module('heads.' + config['head_name'])

endpoints, _ = model.endpoints(img, is_training=False)
with tf.name_scope('head'):
    endpoints = head.head(endpoints, config['embedding_dim'], is_training=False)

tf.train.Saver().restore(sess, os.path.join(config['experiment_root'],'models','checkpoint-25000') )

# %%
if img_read == "tf":
    emb = sess.run(endpoints['emb'])[0]
else:
    emb = sess.run(endpoints['emb'],  feed_dict={img: raw_img})[0]
