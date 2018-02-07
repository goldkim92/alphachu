import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class DQN:

    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self.update_target_rate = 200
        self.dis = 0.9
        self._build_network()

    def _build_network(self, l_rate=1e-3):
        with tf.variable_scope(self.net_name):
            r, c, n  = self.input_size
            self._X = tf.placeholder(
                tf.float32, [None, r, c, n], name="input_x")

            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                              activation_fn=tf.nn.relu,
                              weights_initializer=tf.contrib.layers.xavier_initializer()):
                              # weights_regularizer=slim.l2_regularizer(r_rate)
                net = self._X
                net = slim.conv2d(net, 32, [8, 8], stride = 4, scope='conv1')
                net = slim.conv2d(net, 32, [6, 6], stride = 3, scope='conv2')
                net = slim.conv2d(net, 64, [4, 4], stride = 2, scope='conv3')
                net = slim.conv2d(net, 64, [3, 3], stride = 1, scope='conv4')
                net = slim.flatten(net)
                net = slim.fully_connected(net, 512, scope='fc1')
                net = slim.fully_connected(net, self.output_size, activation_fn=None, scope='fc2')
                self._Qpred = net
                # net = slim.repeat(net, 3, slim.conv2d, 16, [3, 3], scope='conv1')
                # net = slim.max_pool2d(net, [2, 2], scope='pool1')
                # net = slim.repeat(net, 3, slim.conv2d, 32, [3, 3], scope='conv2')
                # net = slim.max_pool2d(net, [2, 2], scope='pool2')
                # net = slim.repeat(net, 3, slim.conv2d, 64, [3, 3], scope='conv3')
                # net = slim.max_pool2d(net, [2, 2], scope='pool3')
                # net = slim.repeat(net, 3, slim.conv2d, 64, [3, 3], scope='conv4')
                # net = slim.max_pool2d(net, [2, 2], scope='pool4')
                # net = slim.repeat(net, 3, slim.conv2d, 64, [3, 3], scope='conv5')
                # net = slim.max_pool2d(net, [2, 2], scope='pool5')
                # net = slim.dropout(net, dropout_rate, scope='dropout6')
                # net = slim.fully_connected(net, 256, scope='fc7')
                # net = slim.dropout(net, dropout_rate, scope='dropout7')
        self._Y = tf.placeholder(
            shape=[None, self.output_size], dtype=tf.float32)

        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        self._train = tf.train.AdamOptimizer(
            learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        r, c, n = self.input_size
        x = np.reshape(state, [1, r, c, n])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={
            self._X: x_stack, self._Y: y_stack})


def replay_train(mainDQN, targetDQN, train_batch):
    r, c, n = mainDQN.input_size

    x_stack = np.empty(0).reshape(0, r, c, n)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + mainDQN.dis * np.max(targetDQN.predict(next_state))
            # Q[0, action] = reward + mainDQN.dis * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])
    return mainDQN.update(x_stack, y_stack)

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder = []
    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

# def bot_play(mainDQN):
#     s = env.reset()
#     reward_sum = 0
#     while True:
#         env.render()
#         a = np.argmax(mainDQN.predict(s))
#         s, reward, done, _ = env.step(a)
#         reward_sum += reward
#         if done:
#             print("Total score: {}".format(reward_sum))
#             break
