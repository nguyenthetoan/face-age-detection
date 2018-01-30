import tensorflow as tf
import cv2,os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

path_pos='/home/nguyen.the.toan/Downloads/dataset/gender-feret/male/training_set'
path_neg='/home/nguyen.the.toan/Downloads/dataset/gender-feret/female/training_set'

def read_data(path,label):
    images = []
    labels = []
    dirs = os.listdir( path )
    for files in dirs:
        file_name=path+"/"+files
        image = cv2.imread(file_name,0)
        image=np.reshape(image, 256*384)
        images.append(image)
        labels.append(label)
    return images, labels


images_pos,labels_pos=read_data(path_pos,1)
images_neg,labels_neg=read_data(path_neg,0)

images=images_pos+images_neg
labels=labels_pos+labels_neg

x_train, x_test,y_train,y_test = train_test_split(images, labels,test_size=0.1, random_state=41)

x_train=np.asarray(x_train)
y_train=np.asarray(y_train)

print(x_train.shape)

learning_rate = 0.1
batch_size = 7
display_step = 10
num_steps=100
n_input = 98304
num_classes=2

x = tf.placeholder(tf.float32, [None, n_input],name="x")
y = tf.placeholder(tf.int32, [None],name="y")

def random_batch(x_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(x_train), batch_size)
    x_batch = x_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return x_batch, y_batch


def conv_net(x):
    x = tf.reshape(x, shape=[-1, 256, 384, 1])
    conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu,padding="SAME")
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2,padding="SAME")
    conv2 = tf.layers.conv2d(conv1, 54, 5, activation=tf.nn.relu,padding="SAME")
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2,padding="SAME")

    fc1 = tf.contrib.layers.flatten(conv2)
    fc1 = tf.layers.dense(fc1, 512,activation=tf.nn.relu)
    out = tf.layers.dense(fc1, num_classes,name="output")

    return out


pred = conv_net(x)

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(xentropy)

optimizer = tf.train.AdamOptimizer()
training_op=optimizer.minimize(cost)

correct = tf.nn.in_top_k(pred, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess= tf.Session()

sess.run(init)
for step in range(1, num_steps+1):
    batch_x, batch_y = random_batch(x_train, y_train, batch_size)
    sess.run(training_op, feed_dict={x: batch_x, y: batch_y})
    if step % display_step == 0 or step == 1:
        acc = sess.run( accuracy, feed_dict={x: x_train,y: y_train})
        print('Step:',step, ', Accuracy:',acc)

print("Optimization Finished!")

save_path = saver.save(sess, "models/face_cnn.ckpt")

test_acc=sess.run(accuracy, feed_dict={x: x_test,y: y_test})
print("Testing Accuracy:", test_acc)
