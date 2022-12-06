#import tensorflow as tf
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import data_helpers
import os
import datetime
import time
#from WideAndDeepModel import WideAndDeepModel #同一文件中进行了定义，不再进行import


# 构建Wide&Deep模型
class WideAndDeepModel:
    def __init__(self, wide_length, deep_length, deep_last_layer_len, softmax_label):
        # 首先定义输入部分, 包括 wide 部分、Deep 部分及标签信息
        self.input_wide_part = tf.compat.v1.placeholder(tf.float32, shape=[None, wide_length], name='input_wide_part')
        self.input_deep_part = tf.compat.v1.placeholder(tf.float32, shape=[None, deep_length], name='input_deep_part')
        self.input_y = tf.compat.v1.placeholder(tf.float32, shape=[None, softmax_label], name='input_y')
        # 定义 Deep 部分的网络结构
        with tf.name_scope('deep_part'):
            w_x1 = tf.Variable(tf.compat.v1.random_normal([wide_length, 256], stddev=0.03), name='w_x1')
            b_x1 = tf.Variable(tf.compat.v1.random_normal([256]), name='b_x1')
            w_x2 = tf.Variable(tf.compat.v1.random_normal([256, deep_last_layer_len], stddev=0.03), name='w_x2')
            b_x2 = tf.Variable(tf.compat.v1.random_normal([deep_last_layer_len]), name='b_x2')

            z1 = tf.add(tf.matmul(self.input_wide_part, w_x1), b_x1)
            a1 = tf.nn.relu(z1)
            self.deep_logits = tf.add(tf.matmul(a1, w_x2), b_x2)

            # 定义 wide 那分的网络结构
            with tf.name_scope('wide_part'):
                weights = tf.Variable(tf.compat.v1.truncated_normal([deep_last_layer_len + wide_length, softmax_label]))
            biases = tf.Variable(tf.zeros([softmax_label]))

            self.wide_and_deep = tf.concat([self.deep_logits, self.input_wide_part], axis=1)
            self.wide_and_deep_logits = tf.add(tf.matmul(self.wide_and_deep, weights), biases)
            self.predictions = tf.argmax(self.wide_and_deep_logits, 1, name="prediction")
            # 定义损失函数
            with tf.name_scope('loss'):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.wide_and_deep_logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
            # 定义准确率
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.compat.v1.float32), name="accuracy")

# 读取训练数据和标签
def load_data_and_labels(path):
    data = []
    y = []
    total_q = []

    with open(path, "r") as f:

        rdr = csv.reader(f, delimiter=',', quotechar='"')
        for row in rdr:
            emb_val = row[4].split(':')
            emb_val_f = [float(i) for i in emb_val]

            cate_emb = row[5].split(':')
            cate_emb_val_f = [float(i) for i in cate_emb]

            total_q.append(int(row[3]))
            data.append(emb_val_f + cate_emb_val_f)
            y.append(float(row[1]))
    data = np.asarray(data)
    total_q = np.asarray(total_q)
    y = np.asarray(y)

    y = y + np.random.normal(0, 1e-3, y.shape)

    bins = pd.qcut(y, 50, retbins=True)

    # 将标签转换为数值区间
    def convert_label_to_interval(y):
        gmv_bins = []
        for i in range(len(y)):
            interval = int(y[i] / 20000)
            if interval < 1000:
                gmv_bins.append(interval)
            elif interval >= 1000:
                gmv_bins.append(1000)

        gmv_bins = np.asarray(gmv_bins)
        return gmv_bins

    y = convert_label_to_interval(y)

    # 将标签转换为One-hot encoding
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels_count = 1001
    labels = dense_to_one_hot(y, labels_count)
    labels = labels.astype(np.uint8)

    def dense_to_one_hot2(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel() - 1] - 1
        return labels_one_hot

    total_q_classes = np.unique(total_q).shape[0]
    total_q = dense_to_one_hot2(total_q, total_q_classes)

    data = np.concatenate((data, total_q), axis=1)
    return data, labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    # 根据训练欺据大小生成 batch
    data = np.array(data, dtype=object)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# 模型训练数据路径
tf.compat.v1.flags.DEFINE_string("train_dir", "dataset/train_20w.csv", "Path of train data")
tf.compat.v1.flags.DEFINE_integer("wide_length", 133, "Path of train data")
tf.compat.v1.flags.DEFINE_integer("deep_length", 133, "Path of train data")
tf.compat.v1.flags.DEFINE_integer("deep_last_layer_len", 128, "Path of train data")
tf.compat.v1.flags.DEFINE_integer("softmax_label", 1001, "Path of train data")

# 设定模型训练参数
tf.compat.v1.flags.DEFINE_integer("batch_size", 32, "Batch Size")
tf.compat.v1.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs")
tf.compat.v1.flags.DEFINE_integer("display_every", 100, "Number of iterations to display training info.")
tf.compat.v1.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with.")
tf.compat.v1.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.compat.v1.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps")

# 定义辅助参数
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.compat.v1.flags.FLAGS


def train():
    with tf.device('/cpu:0'):
        # 读取训练数据
        #x, y = data_helpers.load_data_and_labels(FLAGS.train_dir)
        x, y = load_data_and_labels(FLAGS.train_dir)

    print("-" * 120)
    print(x.shape)
    print("-" * 120)

    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.compat.v1.Session(config=session_conf)

        with sess.as_default():
            model = WideAndDeepModel(
                wide_length=FLAGS.wide_length,
                deep_length=FLAGS.deep_length,
                deep_last_layer_len=FLAGS.deep_last_layer_len,
                softmax_label=FLAGS.softmax_label
            )

            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.compat.v1.train.AdamOptimizer(FLAGS.learning_rate).minimize(model.loss, global_step=global_step)

            # 输出模型文件和临时checkpoint
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # 初始化所有变量
            sess.run(tf.compat.v1.global_variables_initializer())

            # 为每一次的新训练都生成 batch、size
            batches = batch_iter(
                list(zip(x, y)), FLAGS.batch_size, FLAGS.num_epochs)
            for batch in batches:
                x_batch, y_batch = zip(*batch)

                feed_dict = {
                    model.input_wide_part: x_batch,
                    model.input_deep_part: x_batch,
                    model.input_y: y_batch
                }

                _, step, loss, accuracy = sess.run([train_op, global_step, model.loss, model.accuracy], feed_dict)

                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc{:g}".format(time_str, step, loss, accuracy))
                # 保存 check-point
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))
            save_path = saver.save(sess, checkpoint_prefix)

def main():
    train()

if __name__ == "__main__":
    #load_data_and_labels("dataset/train_20w.csv")
    #tf.compat.v1.app.run()
    main()
