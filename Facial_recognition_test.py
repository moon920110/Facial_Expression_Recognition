#!/usr/bin/env python
# -*-coding:utf-8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import tensorflow as tf
import numpy as np 
""" -------------------------------------------------------------
### Facial Expression Recognition
### ------------------------------------------------------------- """

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('test_size', 1, 'test_batch_size')
flags.DEFINE_integer('NoC', 4, 'the number of classes')
flags.DEFINE_string('root', '/media/austin/D_drive/Embedded/', 'root directory for training')
flags.DEFINE_string('ckpt', 'model/vgg_SGD/', 'ckpt directory')

cascPath = '/home/austin/tmp/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
svPath = '/media/austin/D_drive/Embedded/Test/'

faceCascade = cv2.CascadeClassifier(cascPath)
cap = cv2.VideoCapture(0)

def hom(img):
    rows = img.shape[0]
    cols = img.shape[1]

    imgLog = np.log1p(np.array(img, dtype="float") / 255)

    M = 2 * rows + 1
    N = 2 * cols + 1
    sigma = 10
    (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
    Xc = np.ceil(N / 2)
    Yc = np.ceil(M / 2)
    gaussianNumerator = (X - Xc) ** 2 + (Y - Yc) ** 2

    LPF = np.exp(-gaussianNumerator / (2 * sigma * sigma))
    HPF = 1 - LPF

    LPF_shift = np.fft.ifftshift(LPF.copy())
    HPF_shift = np.fft.ifftshift(HPF.copy())

    img_FFT = np.fft.fft2(imgLog.copy(), (M, N))
    img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N)))
    img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N)))

    gamma1 = 0.3
    gamma2 = 1.5
    img_adjusting = gamma1*img_LF[0:rows, 0:cols] + gamma2*img_HF[0:rows, 0:cols]

    img_exp = np.expm1(img_adjusting)
    img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp))
    img_out = np.array(255 * img_exp, dtype="uint8")

    return img_out

def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name = name)

def batch_norm(x, n_out):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.zeros([n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.ones([n_out]),
                            name='gamma', trainable=True)

        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(tf.less(tf.constant(5), tf.constant(2)), mean_var_with_update,
                            lambda:(ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def model(X, w1a, w1b, w2a, w2b, w3a, w3b, w3c, w3d, w4a, w4b, w4c, w4d, w5a, w5b, w5c, w5d, w6a, w6b, w_o):
    conv1a = tf.nn.conv2d(X, w1a, strides=[1, 1, 1, 1], padding='SAME')        # conv1 shape=(?, widtdh, height, kernels) // 224,224, 64
    conv1a_bn = batch_norm(conv1a, 64)
    conv1a_out = tf.nn.relu(conv1a_bn)
    conv1b = tf.nn.conv2d(conv1a_out, w1b, strides=[1, 1, 1, 1], padding='SAME')
    conv1b_bn = batch_norm(conv1b, 64)
    conv1b_out = tf.nn.relu(conv1b_bn)
    pool1 = tf.nn.max_pool(conv1b_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # pool1 shape=(?, width, height, kernels) // 112, 112, 64


    conv2a = tf.nn.conv2d(pool1, w2a, strides=[1, 1, 1, 1], padding='SAME')  # 112,112,128
    conv2a_bn = batch_norm(conv2a, 128)
    conv2a_out = tf.nn.relu(conv2a_bn)
    conv2b = tf.nn.conv2d(conv2a_out, w2b, strides=[1, 1, 1, 1], padding='SAME') # 112,112,128
    conv2b_bn = batch_norm(conv2b, 128)
    conv2b_out = tf.nn.relu(conv2b_bn)
    pool2 = tf.nn.max_pool(conv2b_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #56,56,128

    conv3a = tf.nn.conv2d(pool2, w3a, strides=[1, 1, 1, 1], padding='SAME')      # 56,56,256
    conv3a_bn = batch_norm(conv3a, 256)
    conv3a_out = tf.nn.relu(conv3a_bn)
    conv3b = tf.nn.conv2d(conv3a_out, w3b, strides=[1, 1, 1, 1], padding='SAME')
    conv3b_bn = batch_norm(conv3b, 256)
    conv3b_out = tf.nn.relu(conv3b_bn)
    conv3c = tf.nn.conv2d(conv3b_out, w3c, strides=[1, 1, 1, 1], padding='SAME')
    conv3c_bn = batch_norm(conv3c, 256)
    conv3c_out = tf.nn.relu(conv3c_bn)
    conv3d = tf.nn.conv2d(conv3c_out, w3d, strides=[1, 1, 1, 1], padding='SAME')
    conv3d_bn = batch_norm(conv3d, 256)
    conv3d_out = tf.nn.relu(conv3d_bn)
    pool3 = tf.nn.max_pool(conv3d_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 28,28,256

    conv4a = tf.nn.conv2d(pool3, w4a, strides=[1, 1, 1, 1], padding='SAME') #28,28,512
    conv4a_bn = batch_norm(conv4a, 512)
    conv4a_out = tf.nn.relu(conv4a_bn)
    conv4b = tf.nn.conv2d(conv4a_out, w4b, strides=[1, 1, 1, 1], padding='SAME')
    conv4b_bn = batch_norm(conv4b, 512)
    conv4b_out = tf.nn.relu(conv4b_bn)
    conv4c = tf.nn.conv2d(conv4b_out, w4c, strides=[1, 1, 1, 1], padding='SAME')
    conv4c_bn = batch_norm(conv4c, 512)
    conv4c_out = tf.nn.relu(conv4c_bn)
    conv4d = tf.nn.conv2d(conv4c_out, w4d, strides=[1, 1, 1, 1], padding='SAME')
    conv4d_bn = batch_norm(conv4d, 512)
    conv4d_out = tf.nn.relu(conv4d_bn)
    pool4 = tf.nn.max_pool(conv4d_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #14,14,512.

    conv5a = tf.nn.conv2d(pool4, w5a, strides=[1, 1, 1, 1], padding='SAME') # 14,14,512
    conv5a_bn = batch_norm(conv5a, 512)
    conv5a_out = tf.nn.relu(conv5a_bn)
    conv5b = tf.nn.conv2d(conv5a_out, w5b, strides=[1, 1, 1, 1], padding='SAME')
    conv5b_bn = batch_norm(conv5b, 512)
    conv5b_out = tf.nn.relu(conv5b_bn)
    conv5c = tf.nn.conv2d(conv5b_out, w5c, strides=[1, 1, 1, 1], padding='SAME')
    conv5c_bn = batch_norm(conv5c, 512)
    conv5c_out = tf.nn.relu(conv5c_bn)
    conv5d = tf.nn.conv2d(conv5c_out, w5d, strides=[1, 1, 1, 1], padding='SAME')
    conv5d_bn = batch_norm(conv5d, 512)
    conv5d_out = tf.nn.relu(conv5d_bn)
    pool5 = tf.nn.max_pool(conv5d_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #7,7,512

########### Dense Layer
    dense1 = tf.nn.conv2d(pool5, w6a, strides=[1, 1, 1, 1], padding='VALID')         # 1, 1, 4096
    dense1_bn = batch_norm(dense1, 4096)
    dense1_out = tf.nn.relu(dense1_bn)

    dense2 = tf.nn.conv2d(dense1_out, w6b, strides=[1, 1, 1, 1], padding='SAME')    # 1, 1, 4096
    dense2_bn = batch_norm(dense2, 4096)
    dense2_out = tf.nn.relu(dense2_bn)

    dense3 = tf.nn.conv2d(dense2_out, w_o, strides=[1, 1, 1, 1], padding='SAME') # 1, 1, 4
    dense3_bn = batch_norm(dense3, FLAGS.NoC)
    dense3_out = tf.nn.relu(dense3_bn)
  #  print(dense1_out.get_shape().as_list())
    pyx = tf.reshape(dense3_out, [-1, FLAGS.NoC])

    return pyx

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

py_x = model(X, w1a, w1b, w2a, w2b, w3a, w3b, w3c, w3d,
             w4a, w4b, w4c, w4d, w5a, w5b, w5c, w5d, w6a, w6b, w_o)

predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver()
with tf.Session() as sess :
 #   tf.global_variables_initializer().run(feed_dict={name: 'Test/'})
    saver.restore(sess, FLAGS.root + FLAGS.ckpt)
    print('model restored')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    check = True

    while check:
        _, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = faceCascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in face:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                homed = hom(img[y:y + h, x:x + w])
                blurred = cv2.GaussianBlur(homed, (3, 3), 0)
                eq = cv2.equalizeHist(blurred)
                resized = cv2.resize(eq, (224, 224))
                a = resized.reshape(-1, 224, 224, 1)
                result = sess.run(predict_op, feed_dict={X: a})
                if result[0] == 0:
                    exp = 'Anger'
                elif result[0] == 1:
                    exp = 'Happy'
                elif result[0] == 2:
                    exp = 'Neutral'
                else:
                    exp = 'Sadness'
                print('[test_decision : ', exp, ']')

            elif cv2.waitKey(1)&0xFF == ord('c'):
                check = False

            cv2.imshow('face', img[y:y + h, x:x + w])
    cv2.destroyAllWindows()

    coord.request_stop()
    coord.join(threads)

#if __name__ == "__main__" :
#    tf.app.run()
