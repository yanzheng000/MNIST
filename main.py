
import numpy as np
import tensorflow as tf
import pandas as pd
from pandas import DataFrame, read_csv

# parameters
LABEL = 10 # 0~9
SIZE = 28 # image pixel
COLOR = 1 # grey image

TRAIN = 20000  
VALID = 40000 - TRAIN 
 
RUN = 5000 # total run time
BATCH = 100 # batch size
KS = 5 # kernel size
DEPTH = 12 # kernel depth size
HIDE = 100 

def Accuracy(pred, labels):
    return 100.0 * np.mean(np.float32(np.argmax(pred, axis=1) == np.argmax(labels, axis=1)), axis=0)

def shuffle(data, labels):
    st = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(st)
    np.random.shuffle(labels)


# import data
data = pd.read_csv('train.csv') # read csv 
labels = np.array(data.pop('label')) # remove the label
labels = np.array([np.arange(LABEL) == label for label in labels])
data = np.array(data, dtype=np.float32)/255.0-1.0# convert 
data = data.reshape(len(labels), SIZE, SIZE, COLOR) # reshape
td = data[:TRAIN] # train data
tl = labels[:TRAIN] # train label
vd = data[TRAIN:] # valid data
vl = labels[TRAIN:] # valid label
teda = np.array(pd.read_csv('test.csv'), dtype=np.float32)/255.0-1.0 #test data
teda = teda.reshape(teda.shape[0], SIZE, SIZE, COLOR)

shuffle(td, tl) 


# tensorflow

tf_td = tf.placeholder(tf.float32, shape=(BATCH, SIZE, SIZE, COLOR))
tf_tl = tf.placeholder(tf.float32, shape=(BATCH, LABEL))
tf_vd = tf.constant(vd)
tf_teda = tf.constant(teda)
# parameter in tensorflow
global_step = tf.Variable(0)
w1 = tf.Variable(tf.truncated_normal([KS, KS, COLOR, DEPTH], stddev=0.1))
b1 = tf.Variable(tf.zeros([DEPTH]))
w2 = tf.Variable(tf.truncated_normal([KS, KS, DEPTH, 2*DEPTH], stddev=0.1))
b2 = tf.Variable(tf.constant(1.0, shape=[2*DEPTH]))
w3 = tf.Variable(tf.truncated_normal([SIZE // 4 * SIZE // 4 * 2*DEPTH, HIDE], stddev=0.1))
b3 = tf.Variable(tf.constant(1.0, shape=[HIDE]))
w4 = tf.Variable(tf.truncated_normal([HIDE, LABEL], stddev=0.1))
b4 = tf.Variable(tf.constant(1.0, shape=[LABEL]))

def logits(data):
    conv = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding='SAME')
    pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    hide = tf.nn.relu(pool + b1)
    conv = tf.nn.conv2d(hide, w2, [1, 1, 1, 1], padding='SAME')
    pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    hide = tf.nn.relu(pool + b2)
    reshape = tf.reshape(hide, (-1, SIZE // 4 * SIZE // 4 * 2*DEPTH))
    hide = tf.nn.relu(tf.matmul(reshape, w3) + b3)
    return tf.matmul(hide, w4) + b4

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits(tf_td), tf_tl))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

tp = tf.nn.softmax(logits(tf_td)) # train prediction
vp = tf.nn.softmax(logits(tf_vd)) # valid
tep = tf.nn.softmax(logits(tf_teda)) # test

session = tf.Session()
tf.global_variables_initializer().run(session=session)

fstep = 0
for step in np.arange(RUN):
    fstep += 1
    if fstep*BATCH > TRAIN: 
        shuffle(td, tl)
        fstep = 0
    start = (step * BATCH) % (TRAIN - BATCH); stop = start + BATCH
    batch_data = td[start:stop]
    batch_labels = tl[start:stop, :]

    feed_dict = {tf_td:batch_data, tf_tl:batch_labels}
    opt, bloss, batch_prediction = session.run([optimizer, loss, tp], feed_dict=feed_dict)
    if (step % 200 == 0):
        ba = Accuracy(batch_prediction, batch_labels) # batch accuracy
        va = Accuracy(vp.eval(session=session), vl) # valid accuracy
        print('step %i'%step)
        print('loss = %.2f'%bloss)
        print('batch accuracy = %.1f'%ba)
        print('valid accuracy = %.1f'%va)
        
   
test_labels = np.argmax(tep.eval(session=session), axis=1)

submission = pd.DataFrame(data={'ImageId':(np.arange(test_labels.shape[0])+1), 'Label':test_labels})
submission.to_csv('temp_result.csv', index=False)
submission.head()
