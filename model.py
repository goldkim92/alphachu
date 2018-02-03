import tensorflow as tf
import numpy as np

class DQN:

    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self.dis = 0.9
        self._build_network()

    def _build_network(self, h1_size=200, h2_size=200, dropout_rate = 0.0, l_rate=1e-2):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(
                tf.float32, [None, self.input_size], name="input_x")

            W1 = tf.get_variable("W1", shape=[self.input_size, h1_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.tanh(tf.matmul(self._X, W1))
            # layer1 = tf.nn.dropout(layer1, dropout_rate)
            W2 = tf.get_variable("W2", shape=[h1_size, h2_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            layer2 = tf.nn.tanh(tf.matmul(layer1, W2))
            # layer2 = tf.nn.dropout(layer2, dropout_rate)
            W3 = tf.get_variable("W3", shape=[h2_size, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            self._Qpred = tf.matmul(layer2, W3)

        self._Y = tf.placeholder(
            shape=[None, self.output_size], dtype=tf.float32)

        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        self._train = tf.train.AdamOptimizer(
            learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={
            self._X: x_stack, self._Y: y_stack})


def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, mainDQN.input_size)
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
