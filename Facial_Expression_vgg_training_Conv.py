#!/usr/bin/env python
# -*-coding:utf-8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import copy, os, time

""" -------------------------------------------------------------
### Facial Expression Recognition
### ------------------------------------------------------------- """

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
flags.DEFINE_integer('batch_size', 16, 'batch_size')
flags.DEFINE_integer('test_size', 1, 'test_batch_size')
flags.DEFINE_integer('iteration', 1194, 'the number of iteration for 1 epoch')
flags.DEFINE_integer('epochs', 100, 'the numbFer of epochs')
flags.DEFINE_integer('NoC', 4, 'the number of classes')
flags.DEFINE_boolean('is_training', True, 'True : training / False : test')
flags.DEFINE_string('root', '/media/austin/D_drive/Embedded/', 'root directory for training')
flags.DEFINE_string('ckpt', 'model/vggVersion/', 'ckpt directory')
flags.DEFINE_string('training_set', './inputList/Facial.txt', 'img list')

def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name = name)

def batch_norm(x, n_out, phase_train, CN_FC):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.zeros([n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.ones([n_out]),
                            name='gamma', trainable=True)
        if CN_FC :
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        else :
            batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update,
                            lambda:(ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def model(X, w1a, w1b, w2a, w2b, w3a, w3b, w3c, w3d, w4a, w4b, w4c, w4d, w5a, w5b, w5c, w5d, w6a, w6b, w_o, phase_train):
    conv1a = tf.nn.conv2d(X, w1a, strides=[1, 1, 1, 1], padding='SAME')        # conv1 shape=(?, widtdh, height, kernels) // 224,224, 64
    conv1a_bn = batch_norm(conv1a, 64, phase_train, True)
    conv1a_out = tf.nn.relu(conv1a_bn)
    conv1b = tf.nn.conv2d(conv1a_out, w1b, strides=[1, 1, 1, 1], padding='SAME')
    conv1b_bn = batch_norm(conv1b, 64, phase_train, True)
    conv1b_out = tf.nn.relu(conv1b_bn)
    pool1 = tf.nn.max_pool(conv1b_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # pool1 shape=(?, width, height, kernels) // 112, 112, 64


    conv2a = tf.nn.conv2d(pool1, w2a, strides=[1, 1, 1, 1], padding='SAME')  # 112,112,128
    conv2a_bn = batch_norm(conv2a, 128, phase_train, True)
    conv2a_out = tf.nn.relu(conv2a_bn)
    conv2b = tf.nn.conv2d(conv2a_out, w2b, strides=[1, 1, 1, 1], padding='SAME') # 112,112,128
    conv2b_bn = batch_norm(conv2b, 128, phase_train, True)
    conv2b_out = tf.nn.relu(conv2b_bn)
    pool2 = tf.nn.max_pool(conv2b_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #56,56,128

    conv3a = tf.nn.conv2d(pool2, w3a, strides=[1, 1, 1, 1], padding='SAME')      # 56,56,256
    conv3a_bn = batch_norm(conv3a, 256, phase_train, True)
    conv3a_out = tf.nn.relu(conv3a_bn)
    conv3b = tf.nn.conv2d(conv3a_out, w3b, strides=[1, 1, 1, 1], padding='SAME')
    conv3b_bn = batch_norm(conv3b, 256, phase_train, True)
    conv3b_out = tf.nn.relu(conv3b_bn)
    conv3c = tf.nn.conv2d(conv3b_out, w3c, strides=[1, 1, 1, 1], padding='SAME')
    conv3c_bn = batch_norm(conv3c, 256, phase_train, True)
    conv3c_out = tf.nn.relu(conv3c_bn)
    conv3d = tf.nn.conv2d(conv3c_out, w3d, strides=[1, 1, 1, 1], padding='SAME')
    conv3d_bn = batch_norm(conv3d, 256, phase_train, True)
    conv3d_out = tf.nn.relu(conv3d_bn)
    pool3 = tf.nn.max_pool(conv3d_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 28,28,256

    conv4a = tf.nn.conv2d(pool3, w4a, strides=[1, 1, 1, 1], padding='SAME') #28,28,512
    conv4a_bn = batch_norm(conv4a, 512, phase_train, True)
    conv4a_out = tf.nn.relu(conv4a_bn)
    conv4b = tf.nn.conv2d(conv4a_out, w4b, strides=[1, 1, 1, 1], padding='SAME')
    conv4b_bn = batch_norm(conv4b, 512, phase_train, True)
    conv4b_out = tf.nn.relu(conv4b_bn)
    conv4c = tf.nn.conv2d(conv4b_out, w4c, strides=[1, 1, 1, 1], padding='SAME')
    conv4c_bn = batch_norm(conv4c, 512, phase_train, True)
    conv4c_out = tf.nn.relu(conv4c_bn)
    conv4d = tf.nn.conv2d(conv4c_out, w4d, strides=[1, 1, 1, 1], padding='SAME')
    conv4d_bn = batch_norm(conv4d, 512, phase_train, True)
    conv4d_out = tf.nn.relu(conv4d_bn)
    pool4 = tf.nn.max_pool(conv4d_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #14,14,512.

    conv5a = tf.nn.conv2d(pool4, w5a, strides=[1, 1, 1, 1], padding='SAME') # 14,14,512
    conv5a_bn = batch_norm(conv5a, 512, phase_train, True)
    conv5a_out = tf.nn.relu(conv5a_bn)
    conv5b = tf.nn.conv2d(conv5a_out, w5b, strides=[1, 1, 1, 1], padding='SAME')
    conv5b_bn = batch_norm(conv5b, 512, phase_train, True)
    conv5b_out = tf.nn.relu(conv5b_bn)
    conv5c = tf.nn.conv2d(conv5b_out, w5c, strides=[1, 1, 1, 1], padding='SAME')
    conv5c_bn = batch_norm(conv5c, 512, phase_train, True)
    conv5c_out = tf.nn.relu(conv5c_bn)
    conv5d = tf.nn.conv2d(conv5c_out, w5d, strides=[1, 1, 1, 1], padding='SAME')
    conv5d_bn = batch_norm(conv5d, 512, phase_train, True)
    conv5d_out = tf.nn.relu(conv5d_bn)
    pool5 = tf.nn.max_pool(conv5d_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #7,7,512

########### Dense Layer
    dense1 = tf.nn.conv2d(pool5, w6a, strides=[1, 1, 1, 1], padding='VALID')         # 1, 1, 4096
    dense1_bn = batch_norm(dense1, 4096, phase_train, True)
    dense1_out = tf.nn.relu(dense1_bn)

    dense2 = tf.nn.conv2d(dense1_out, w6b, strides=[1, 1, 1, 1], padding='SAME')    # 1, 1, 4096
    dense2_bn = batch_norm(dense2, 4096, phase_train, True)
    dense2_out = tf.nn.relu(dense2_bn)

    dense3 = tf.nn.conv2d(dense2_out, w_o, strides=[1, 1, 1, 1], padding='SAME') # 1, 1, 4
    dense3_bn = batch_norm(dense3, FLAGS.NoC, phase_train, True)
    dense3_out = tf.nn.relu(dense3_bn)
  #  print(dense1_out.get_shape().as_list())
    pyx = tf.reshape(dense3_out, [-1, FLAGS.NoC])

    return pyx

def load_img(img_queue, label, batch_size):                                          
    reader = tf.WholeFileReader()
    filename, content = reader.read(img_queue)
    image = tf.image.decode_jpeg(content, channels=1)
    image.set_shape([224,224,1])
    image_batch, image_label = tf.train.batch([image, label], batch_size=batch_size)    # make batch from queue or something else.
    return image_batch, image_label, filename  # return img batch, img label



""" -------------------------------------------------------------------------------------------
    for train set,
    read file name and label data from a text file that has sequence of the image & label list
    --------------------------------------------------------------------------------------------"""
def read_labeled_image_list(image_list_file):
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    arr = [0 for _ in range(FLAGS.NoC)]
    for line in f :
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        arr[int(label)] = 1
        tmp = copy.deepcopy(arr)
        labels.append(tmp)
        arr[int(label)] = 0
    f.close()
    return filenames, labels

""" ---------------------------------------------------------
    for train set,
    take images and labels tensor from a queue
    ---------------------------------------------------------"""
def read_images_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=1)
  #  example = tf.cast(example, tf.float32)
    example.set_shape([224,224,1])
    return example, label

def load_test_set(path, label):
    files = os.listdir(FLAGS.root+path)
    labels = [0 for _ in range(FLAGS.NoC)]
    labels[label] = 1
    test = tf.train.string_input_producer([(FLAGS.root+path+'%s'%name) for name in files])
    return load_img(test, labels, FLAGS.test_size)

def training(_y, y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_y, y))
    train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)
    # train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost)
    cor = tf.equal(tf.argmax(_y, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(cor, tf.float32))
    return acc, train_op, cost

if FLAGS.is_training:
    image_list, label_list = read_labeled_image_list(FLAGS.training_set)
    images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)
    input_queue = tf.train.slice_input_producer([images, labels], num_epochs=None, shuffle=True)
    image, label = read_images_from_disk(input_queue)
    image_batch = tf.train.batch([image, label], batch_size=FLAGS.batch_size)  # image_batch[0] = image, [1] = label
    ### sequence of the training set is already shuffled in the text file, so I don't need to use shuffle_batch

    ### load test set
    print('load validation set')
    happy_batch = load_test_set("cropped/h/", 1)
    angry_batch = load_test_set("cropped/a/", 0)
    sad_batch = load_test_set("cropped/s/", 3)
    neutral_batch = load_test_set("cropped/n/", 2)
else :
    print('load test set')
    test_batch = load_test_set("Test/", 1)                             # [0] = img, [1] = label

### set tensors
X = tf.placeholder("float", [None, 224, 224, 1], name='X')
Y = tf.placeholder("float", [None, FLAGS.NoC], name='Y')

w1a = init_weights([3, 3, 1, 64], 'w1a')                                                  
w1b = init_weights([3, 3, 64, 64], 'w1b')
w2a = init_weights([3, 3, 64, 128], 'w2a')
w2b = init_weights([3, 3, 128, 128], 'w2b')
w3a = init_weights([3, 3, 128, 256], 'w3a')
w3b = init_weights([3, 3, 256, 256], 'w3n')                                                 
w3c = init_weights([3, 3, 256, 256], 'w3c')
w3d = init_weights([3, 3, 256, 256], 'w3d')
w4a = init_weights([3, 3, 256, 512], 'w4a')
w4b = init_weights([3, 3, 512, 512], 'w4b')
w4c = init_weights([3, 3, 512, 512], 'w4c')
w4d = init_weights([3, 3, 512, 512], 'w4d')
w5a = init_weights([3, 3, 512, 512], 'w5a')
w5b = init_weights([3, 3, 512, 512], 'w5b')
w5c = init_weights([3, 3, 512, 512], 'w5c')
w5d = init_weights([3, 3, 512, 512], 'w5d')
w6a = init_weights([7, 7, 512, 4096], 'w6a')
w6b = init_weights([1, 1, 4096, 4096], 'w6b')
w_o = init_weights([1, 1, 4096, FLAGS.NoC], 'w_o')

phase_train = tf.placeholder(tf.bool, name='phase_train')

py_x = model(X, w1a, w1b, w2a, w2b, w3a, w3b, w3c, w3d,
             w4a, w4b, w4c, w4d, w5a, w5b, w5c, w5d, w6a, w6b, w_o, phase_train)

output = training(py_x, Y)
predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver()

with tf.Session() as sess :                                                             

    if FLAGS.is_training :
        tf.global_variables_initializer().run(feed_dict={phase_train:True})
        # saver.restore(sess, FLAGS.root + FLAGS.ckpt)
        # print('model restored')
    else:
        saver.restore(sess, FLAGS.root + FLAGS.ckpt)
        print('model restored')
    start_time2 = time.time()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)                      
    
    start = time.time()

    print ('start')
    if FLAGS.is_training:

        start_time = time.time()
        happy = sess.run(happy_batch)
        angry = sess.run(angry_batch)
        neutral = sess.run(neutral_batch)
        sad = sess.run(sad_batch)

        activate = True
        for i in range(FLAGS.epochs*FLAGS.iteration):

            image_tensor = sess.run(image_batch)
    #        for j in range(20):
    #            plt.imshow(image_tensor[0][j])
    #            print (image_tensor[1][j])
    #            plt.show()

            accuracy, train_op, cost = sess.run(output, feed_dict={X: image_tensor[0], Y: image_tensor[1],
                                          phase_train: activate})

            if i % FLAGS.iteration == 0 :
                running_time = time.time() - start_time
                start_time = time.time()
                activate = False
                print('===================================================')
                print("cost:", cost)
                print('%d accuracy : '%int(i/FLAGS.iteration), accuracy)
                print('[happy batch', i/FLAGS.iteration, 'validation][', running_time, 's][accuracy : %.4f' % np.mean(np.argmax(happy[1], axis=1)==
                            sess.run(predict_op, feed_dict={X: happy[0],
                                                            phase_train: activate})), ']')
                print('[angry batch', i / FLAGS.iteration, 'validation][', running_time,
                      's][accuracy : %.4f' % np.mean(np.argmax(angry[1], axis=1) ==
                                                     sess.run(predict_op, feed_dict={X: angry[0],
                                                                                     phase_train: activate})), ']')
                print('[sad batch', i / FLAGS.iteration, 'validation][', running_time,
                      's][accuracy : %.4f' % np.mean(np.argmax(sad[1], axis=1) ==
                                                     sess.run(predict_op, feed_dict={X: sad[0],
                                                                                     phase_train: activate})), ']')
                print('[neutral batch', i / FLAGS.iteration, 'validation][', running_time,
                      's][accuracy : %.4f' % np.mean(np.argmax(neutral[1], axis=1) ==
                                                     sess.run(predict_op, feed_dict={X: neutral[0],
                                                                                     phase_train: activate})), ']')
                print('===================================================')
                activate = True

        run_time = time.time() - start
        hour = int(run_time) / 3600
        minute = int(run_time % 3600) / 60
        second = (run_time % 3600) % 60

        save_path = saver.save(sess, FLAGS.root+FLAGS.ckpt)
        print (int(hour), 'h ', int(minute), 'm ', second, 's')
        print ("model save in dir : %s" % save_path)

    else :

        activate = False


        for i in range(30) :
            happy_test = sess.run(test_batch)
            result = sess.run(predict_op, feed_dict={X: happy_test[0], phase_train: activate})
            if result[0] == 0 :
                exp = 'Anger'
            elif result[0] == 1 :
                exp = 'Happy'
            elif result[0] == 2 :
                exp = 'Neutral'
            else :
                exp = 'Sadness'
            print('[happy_test][decision : ', happy_test[2], exp, ']')
        running_time2 = time.time() - start_time2
        print (running_time2)

    coord.request_stop()
    coord.join(threads)
