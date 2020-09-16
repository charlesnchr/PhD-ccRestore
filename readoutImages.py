import os
from skimage import io
import sys

# fix to make it tf 2 compatible (replaces import tensorflow as tf)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def save_images_from_event(fn, tag, prefix, output_dir='./', maxexport = 0):
    os.makedirs(output_dir, exist_ok=True)
    assert(os.path.isdir(output_dir))


    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/{}_{:04d}.png'.format(output_dir, prefix, count))
                    print("Saving '{}'".format(output_fn))
                    io.imsave(output_fn, im)
                    count += 1  
                    if count == maxexport: return

# 'events.out.tfevents.1597287793.wizion-turtle.357022.0','sr/0'

for i in range(20):
    save_images_from_event(sys.argv[1],'sr/%d' % i, '%d' % i, output_dir='C:/temp/sr_%d' % i, maxexport=0)



# from collections import defaultdict, namedtuple
# from typing import List
# import tensorflow as tf


# TensorBoardImage = namedtuple("TensorBoardImage", ["topic", "image", "cnt"])


# def extract_images_from_event(event_filename: str, image_tags: List[str]):
#     topic_counter = defaultdict(lambda: 0)

#     serialized_examples = tf.data.TFRecordDataset(event_filename)
#     for serialized_example in serialized_examples:
#         event = event_pb2.Event.FromString(serialized_example.numpy())
#         for v in event.summary.value:
#             if v.tag in image_tags:

#                 if v.HasField('tensor'):  # event for images using tensor field
#                     s = v.tensor.string_val[2]  # first elements are W and H

#                     tf_img = tf.image.decode_image(s)  # [H, W, C]
#                     np_img = tf_img.numpy()

#                     topic_counter[v.tag] += 1

#                     cnt = topic_counter[v.tag]
#                     tbi = TensorBoardImage(topic=v.tag, image=np_img, cnt=cnt)

#                     yield tbi

# extract_images_from_event('events.out.tfevents.1597287793.wizion-turtle.357022.0',['hr/0'])